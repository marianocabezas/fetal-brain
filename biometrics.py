import os
import math
import argparse

import numpy as np
from scipy.ndimage import binary_fill_holes

from tqdm.auto import tqdm

import pydicom as dicom

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
    torch_x = torch.as_tensor(x, dtype=torch.float32, device=device)
    torch_y = torch.as_tensor(y, dtype=torch.float32, device=device)

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
                v1 = np.array([c.item(), s.item()])
                v2 = np.array([s.item(), - c.item()])
                best_ratio = ratio

    return v1, v2


def line_mask_intersections(center, direction, vert_edges, horiz_edges, eps=1e-9):
    cx, cy = center
    dx, dy = direction
    vx, vy_lo, vy_hi = vert_edges
    hy, hx_lo, hx_hi = horiz_edges

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

    pos = ts > eps
    neg = ts < -eps

    p_pos = pts[pos][np.argmin(ts[pos])] if pos.any() else None
    p_neg = pts[neg][np.argmax(ts[neg])] if neg.any() else None
    return p_neg.tolist(), p_pos.tolist()


def compute_tcd(
        cerebellum, spacing, n_init=32, n_iters=1000, lr=5e-2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    y, x = np.where(cerebellum)
    c_x, c_y = np.mean(x), np.mean(y)

    # Find the axes (minimise the ratio between the short and long edges of the
    # bounding box). In our case, we can reduce the box to a set of rotated axes
    # located in the center of mass.
    v1, v2 = find_cerebellum_axes(x, y, n_init, n_iters, lr, device)

    # vertical edges: boundary between column c and c+1 -> x = c+0.5,
    # y spans [r-0.5, r+0.5]
    vdiff = cerebellum[:, 1:] != cerebellum[:, :-1]
    rows, cols = np.where(vdiff)
    vx = cols.astype(float) + 0.5
    vy_lo = rows.astype(float) - 0.5
    vy_hi = rows.astype(float) + 0.5

    # horizontal edges: boundary between row r and r+1 -> y = r+0.5,
    # x spans [c-0.5, c+0.5]
    hdiff = cerebellum[1:, :] != cerebellum[:-1, :]
    rows2, cols2 = np.where(hdiff)
    hy = rows2.astype(float) + 0.5
    hx_lo = cols2.astype(float) - 0.5
    hx_hi = cols2.astype(float) + 0.5

    ((v1_x0, v1_y0), (v1_xf, v1_yf)) = line_mask_intersections(
        (c_x, c_y), v1,
        (vx, vy_lo, vy_hi),
        (hy, hx_lo, hx_hi)
    )
    ((v2_x0, v2_y0), (v2_xf, v2_yf)) = line_mask_intersections(
        (c_x, c_y), v2,
        (vx, vy_lo, vy_hi),
        (hy, hx_lo, hx_hi)
    )

    v1_x0_w = v1_x0 * spacing[0]
    v1_xf_w = v1_xf * spacing[0]
    v1_y0_w = v1_y0 * spacing[1]
    v1_yf_w = v1_yf * spacing[1]

    v2_x0_w = v2_x0 * spacing[0]
    v2_xf_w = v2_xf * spacing[0]
    v2_y0_w = v2_y0 * spacing[1]
    v2_yf_w = v2_yf * spacing[1]

    length1 = np.sqrt((v1_xf_w - v1_x0_w) ** 2 + (v1_yf_w - v1_y0_w) ** 2)
    length2 = np.sqrt((v2_xf_w - v2_x0_w) ** 2 + (v2_yf_w - v2_y0_w) ** 2)

    tcd = length2 if length2 > length1 else length1
    return tcd


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
            tcd = compute_tcd(cerebellum, spacing, n_init, n_epochs, lr, device)
            tcds.append(tcd)

            print(
                f'{im_id:<5} | [tcd: {tcd:6.3f} mm] | {sorted(layer_dict.keys())}'
            )

        except (IndexError, FileNotFoundError) as e:
            print('ERROR loading', im_id, e)

    print(
        f'TCDs range [{np.min(tcds):6.3f} - {np.max(tcds):6.3f}] mm',
        f'({np.mean(tcds):6.3f} ± {np.std(tcds):6.3f} mm)'
    )

if __name__ == '__main__':
    main()