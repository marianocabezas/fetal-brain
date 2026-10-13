import os
import math
import argparse

import numpy as np
from scipy.ndimage import binary_fill_holes

from tqdm.auto import tqdm

import pydicom as dicom

import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

import torch

from utils import load_xcf


def project_points(points, theta):
    norm_x, norm_y = points
    c, s = torch.cos(theta), torch.sin(theta)

    # First we project the points to the new rotated axes
    # (centered on the mask)
    proj_u = norm_x * c + norm_y * s
    proj_v = - norm_x * s + norm_y * c

    return proj_u, proj_v


def ratio_loss(points, theta):
    proj_u, proj_v = project_points(points, theta)

    # Then we compute the mask.
    # This is useful for "elongated" segmentations only.
    l_axis1 = proj_u.max() - proj_u.min()
    l_axis2 = proj_v.max() - proj_v.min()
    if l_axis1 < l_axis2:
        ratio = l_axis1 / l_axis2
    else:
        ratio = l_axis2 / l_axis1

    return ratio


def find_cerebellum_axes(
        x, y, n_init=32, n_iters=1000, lr=5e-2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    torch_x = torch.as_tensor(
        x, dtype=torch.float32, device=device
    ).requires_grad_(True)
    torch_y = torch.as_tensor(
        y, dtype=torch.float32, device=device
    ).requires_grad_(True)

    norm_x = torch_x - torch_x.mean()
    norm_y = torch_y - torch_y.mean()

    v1 = None
    v2 = None
    best_ratio = torch.inf
    init_angles = torch.linspace(0, math.pi / 2, n_init + 1)[:-1]

    for theta0 in init_angles:
        theta = theta0.clone().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=lr)

        for i in range(n_iters):
            optimizer.zero_grad()

            ratio = ratio_loss((norm_x, norm_y), theta)

            ratio.backward()
            optimizer.step()

        with torch.no_grad():
            ratio = ratio_loss((norm_x, norm_y), theta)
            if ratio < best_ratio:
                c, s = torch.cos(theta), torch.sin(theta)

                proj_u, proj_v = project_points((norm_x, norm_y), theta)
                l_axis1 = proj_u.max() - proj_u.min()
                l_axis2 = proj_v.max() - proj_v.min()
                if l_axis1 >= l_axis2:
                    v1 = np.array([c.item(), s.item()])
                    v2 = np.array([s.item(), - c.item()])
                else:
                    v2 = np.array([c.item(), s.item()])
                    v1 = np.array([s.item(), - c.item()])

                best_ratio = ratio

                extreme_u = torch.as_tensor(
                    [proj_u.max(), proj_u.min()],
                    dtype=torch.float32, device=device
                )
                extreme_v = torch.as_tensor(
                    [proj_v.max(), proj_v.min()],
                    dtype=torch.float32, device=device
                )

                extreme_x, extreme_y = project_points(
                    (extreme_u, extreme_v), - theta
                )
                denorm_e_x = extreme_x.mean().cpu() + torch_x.mean().detach().cpu()
                denorm_e_y = extreme_y.mean().cpu() + torch_y.mean().detach().cpu()

                center = (denorm_e_x.numpy(), denorm_e_y.numpy())

    return v1, v2, center


def line_mask_intersections(
        center, direction, mask, eps=1e-9
):
    cx, cy = center
    dx, dy = direction

    # vertical edges: boundary between column c and c+1 -> x = c+0.5,
    # y spans [r-0.5, r+0.5]
    vdiff = mask[:, 1:] != mask[:, :-1]
    rows, cols = np.where(vdiff)
    vx = cols.astype(float) + 0.5
    vy_lo = rows.astype(float) - 0.5
    vy_hi = rows.astype(float) + 0.5

    # horizontal edges: boundary between row r and r+1 -> y = r+0.5,
    # x spans [c-0.5, c+0.5]
    hdiff = mask[1:, :] != mask[:-1, :]
    rows2, cols2 = np.where(hdiff)
    hy = rows2.astype(float) + 0.5
    hx_lo = cols2.astype(float) - 0.5
    hx_hi = cols2.astype(float) + 0.5

    ts, pts = [], []

    if abs(dx) > eps:
        t = (vx - cx) / dx
        y_at = cy + t * dy
        ok = (y_at >= vy_lo - 1e-6) & (y_at <= vy_hi + 1e-6)
        ts.append(t[ok])
        pts.append(np.stack([vx[ok], y_at[ok]], axis=1))

    if abs(dy) > eps:
        t = (hy - cy) / dy
        x_at = cx + t * dx
        ok = (x_at >= hx_lo - 1e-6) & (x_at <= hx_hi + 1e-6)
        ts.append(t[ok])
        pts.append(np.stack([x_at[ok], hy[ok]], axis=1))

    ts = np.concatenate(ts)
    pts = np.concatenate(pts, axis=0)

    valid = np.abs(ts) > eps

    p_pos = pts[valid][np.argmax(ts[valid])].tolist() if valid.any() else None
    p_neg = pts[valid][np.argmin(ts[valid])].tolist() if valid.any() else None
    return p_neg, p_pos


def sample_arc(a, b, n=100):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    if a @ b < 0:
        b = -b
    u = b - (a @ b) * a
    s = np.linalg.norm(u)
    if s < 1e-12:
        return np.tile(a, (n, 1))
    u /= s
    omega = np.arctan2(s, a @ b)
    info = np.finfo(np.double)
    phi = np.random.uniform(0, omega, size=(n, 1))
    return np.cos(phi) * a + np.sin(phi) * u


def compute_tcd(
        cerebellum, spacing, n_init=32, n_iters=1000, lr=5e-2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    y, x = np.where(cerebellum)
    com_x, com_y = np.mean(x), np.mean(y)

    # Find the axes (minimise the ratio between the short and long edges of the
    # bounding box). In our case, we can reduce the box to a set of rotated axes
    # located in the center of mass.
    v1, v2, (c_vx, c_vy) = find_cerebellum_axes(x, y, n_init, n_iters, lr, device)

    ((v1_x0, v1_y0), (v1_xf, v1_yf)) = line_mask_intersections(
        (com_x, com_y), v1, cerebellum
    )
    ((v2_x0, v2_y0), (v2_xf, v2_yf)) = line_mask_intersections(
        (com_x, com_y), v2, cerebellum
    )

    v1_x0_w = v1_x0 * spacing[0]
    v1_xf_w = v1_xf * spacing[0]
    v1_y0_w = v1_y0 * spacing[1]
    v1_yf_w = v1_yf * spacing[1]

    v2_x0_w = v2_x0 * spacing[0]
    v2_xf_w = v2_xf * spacing[0]
    v2_y0_w = v2_y0 * spacing[1]
    v2_yf_w = v2_yf * spacing[1]
    axis1 = ([v1_x0, v1_xf], [v1_y0, v1_yf])
    axis2 = ([v2_x0, v2_xf], [v2_y0, v2_yf])

    length1 = np.sqrt((v1_xf_w - v1_x0_w) ** 2 + (v1_yf_w - v1_y0_w) ** 2)
    length2 = np.sqrt((v2_xf_w - v2_x0_w) ** 2 + (v2_yf_w - v2_y0_w) ** 2)

    tcd = length2 if length2 > length1 else length1
    long_axis = axis2 if length2 > length1 else axis1
    short_axis = axis2 if length2 <= length1 else axis1

    return tcd, long_axis, short_axis, v1, v2, (c_vx, c_vy)


def compute_section_distribution(mask, v_array, center, spacing, plot=False):
    segments = []
    if plot:
        plt.figure(figsize=(40, 20))
        plt.imshow(mask, cmap='grey')

    for v_i, v in enumerate(v_array):
        p0, pf = line_mask_intersections(
            center, v, mask
        )
        if plot:
            ax = plt.gca()
            ax.scatter(
                [center[0]], [center[1]],
                s=100, c='orange', label='Cerebellum CoM'
            )
            ax.quiver(
                center[0], center[1], v[0] * 100, v[1] * 100,
                scale_units='xy', scale=1,
                color='teal', angles='xy'
            )

        if p0 is not None and pf is not None:
            p0_sp = np.array(p0) * np.array(spacing)
            pf_sp = np.array(pf) * np.array(spacing)
            if plot:
                ax = plt.gca()
                ax.plot(
                    [p0[0], pf[0]], [p0[1], pf[1]],
                    'r--'
                )
            segments.append(np.linalg.norm(pf_sp - p0_sp))

    return segments


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Check data related to the MAGE - MRI relationship'
    )

    # Mode selector
    parser.add_argument(
        '-x', '--xcf-path',
        dest='xcf_path', default='/home/mariano/Datasets/GIMPXCF-Anna/',
        help='Path to the GIMP file annotations.'
    )
    parser.add_argument(
        '-d', '--dicom-path',
        dest='dicom_path', default='/home/mariano/Datasets/DCM/',
        help='Path to the DICOM image files.'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=1000,
        help='Number of epochs for tha axis optimisation.'
    )
    parser.add_argument(
        '-l', '--learning-rate',
        dest='learning_rate',
        type=float, default=5e-2,
        help='Learning rate for the axis optimisation.'
    )
    parser.add_argument(
        '-n', '--init_angles',
        dest='n_init',
        default=32, action='store_true',
        help='Number of axis optimisation tries.'
    )
    options = vars(parser.parse_args())

    return options


def main():
    # Init
    options = parse_inputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xcf_path = options['xcf_path']
    dicom_path = options['dicom_path']
    lr = options['learning_rate']
    n_init = options['n_init']
    n_epochs = options['epochs']

    layer_list = [
        'cavum', 'cerebel', 'cisterna magna', 'plec nucal', 'midline', 'silvio'
    ]

    xcf_files = sorted(
        [f for f in os.listdir(xcf_path) if f.endswith('.xcf')],
        key=lambda x: int(x.split('.')[0])
    )
    xcf_loop = tqdm(
        enumerate(xcf_files), total=len(list(xcf_files)), leave=False,
        desc='XCF and DCM file reading',
    )

    tcds = []

    for i, xcf_f in xcf_loop:
        f_split = xcf_f.split('.')
        im_id = '.'.join(f_split[:-1])
        dicom_f = im_id + '.dcm'
        dicom_filepath = os.path.join(dicom_path, dicom_f)
        try:
            xcf_filepath = os.path.join(xcf_path, xcf_f)
            names, data = load_xcf(xcf_filepath)
            layer_dict = {
                name.lower(): d_i
                for name, d_i in zip(names, data)
                if '.dcm' not in name
            }
            dcm_data = dicom.dcmread(dicom_filepath)
            spacing = dcm_data[0x00280030].value

            #### Data preparation
            # > Area segmentations
            mask = np.max([
                (i + 1) * binary_fill_holes(np.array(layer_dict[name])[..., -1] > 0)
                for i, name in enumerate(layer_list[:-2])
            ], axis=0)

            #### Cerebellum biometrics
            cerebellum = mask == 2
            tcd, _, _, _, vcer, cereb_c = compute_tcd(
                cerebellum, spacing, n_init, n_epochs, lr, device
            )
            tcds.append(tcd)

            #### CSP biometrics
            csp = mask == 1
            csp_com = np.mean(np.where(csp), axis=1)[::-1]

            vc2c = np.array([csp_com[0] - cereb_c[0], csp_com[1] - cereb_c[1]])
            v_array = sample_arc(vc2c, vcer, 200)

            #### Cisterna magna biometrics
            cm_segments = compute_section_distribution(
                mask == 3, v_array, cereb_c, spacing, plot=True
            )
            ax = plt.gca()
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    '/home/mariano/Datasets/USFoetalSorted-Anna/Debugging',
                    f'{im_id}_cm.png'
                )
            )
            plt.close()

            #### Nuchal fold biometrics
            nf_segments = compute_section_distribution(
                mask == 4, v_array, cereb_c, spacing, plot=True
            )
            ax = plt.gca()
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    '/home/mariano/Datasets/USFoetalSorted-Anna/Debugging',
                    f'{im_id}_nf.png'
                )
            )
            plt.close()

            print(
                f'{im_id:<5} | [TCD: {tcd:6.3f} mm] |',
                f'[Cisterna magna: {np.median(cm_segments):6.3f} mm] |',
                f'[Nuchal fold: {np.median(nf_segments):6.3f} mm] |',
                f'{sorted(layer_dict.keys())}'
            )

        except (IndexError, FileNotFoundError) as e:
            print('ERROR loading', im_id, e)

    print(
        f'TCDs range [{np.min(tcds):6.3f} - {np.max(tcds):6.3f}] mm',
        f'({np.mean(tcds):6.3f} ± {np.std(tcds):6.3f} mm)'
    )

if __name__ == '__main__':
    main()