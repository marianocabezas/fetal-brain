import os
import numpy as np
import torch
import torch.nn.functional as func
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from utils import load_xcf, robust_fit_ellipse, ellipse_to_mask


"""
> Similarity-based losses
"""


def xcor_loss(fixed, moved, mask=None):
    if mask is None:
        fixed_norm = fixed - torch.mean(fixed)
        moved_norm = moved - torch.mean(moved)
    else:
        valid_fixed = fixed[mask]
        valid_moved = moved[mask]
        fixed_norm = valid_fixed - torch.mean(valid_fixed)
        moved_norm = valid_moved - torch.mean(valid_moved)
    fixed_sq = torch.sum(fixed_norm ** 2)
    moved_sq = torch.sum(moved_norm ** 2)

    den = torch.sqrt(fixed_sq * moved_sq)
    num = torch.sum(fixed_norm * moved_norm)

    xcor = num / den if den > 0 else 0

    return 1. - xcor


def xcor_patch_loss(fixed, moved, mask=None, k=8):
    if len(fixed.shape) < 4:
        unsqueeze = (1,) * (4 - len(fixed.shape))
        fixed = torch.reshape(fixed, unsqueeze + fixed.shape)
    if len(moved.shape) < 4:
        unsqueeze = (1,) * (4 - len(moved.shape))
        moved = torch.reshape(moved, unsqueeze + moved.shape)
    fixed_mean = func.interpolate(
        func.avg_pool2d(fixed, k),
        fixed.shape[2:]
    )
    moved_mean = func.interpolate(
        func.avg_pool2d(moved, k),
        moved.shape[2:]
    )
    fixed_norm = fixed - fixed_mean
    moved_norm = moved - moved_mean
    fixed_sq = func.avg_pool2d(fixed_norm ** 2, k)
    moved_sq = func.avg_pool2d(moved_norm ** 2, k)

    den = torch.sqrt(fixed_sq * moved_sq)
    num = func.avg_pool2d(fixed_norm * moved_norm, k)

    xcor = torch.mean(num / den)

    return 1. - xcor


def mse_loss(fixed, moved, mask=None):
    if mask is None:
        mse_val = func.mse_loss(moved, fixed)
    else:
        valid_fixed = fixed[mask]
        valid_moved = moved[mask]
        mse_val = func.mse_loss(valid_moved, valid_fixed)

    return mse_val


"""
> Registration code
"""


def get_affine_matrix(
    theta, height, width, a, b, center_x, center_y
):
    # Init
    half_x = width / 2
    half_y = height / 2
    # Center removal
    cx = - center_x
    cy = - center_y
    # Scaling
    sx = half_x / a
    sy = half_y / b
    # Rotation + scaling
    rxx = sx * np.cos(theta)
    rxy = - sx * np.sin(theta)
    ryx = sy * np.sin(theta)
    ryy = sy * np.cos(theta)
    # Translation
    tx = half_x
    ty = half_y

    to_center = torch.tensor(
        [
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1],
        ], dtype=torch.float64
    )
    ellipse_affine = torch.tensor(
        [
            [rxx, rxy, tx],
            [ryx, ryy, ty],
            [0, 0, 1],
        ], dtype=torch.float64
    )

    affine = ellipse_affine @ to_center

    return affine


def registration2d(
    fixed, moving, mask=None, init_affine=None, flip=False,
    scales=None, epochs=500, patience=100, init_lr=1e-3, loss_f=xcor_loss,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
):
    if scales is None:
        scales = [8, 4, 2, 1]

    best_fit = np.inf
    final_e = 0
    final_fit = np.inf

    fixed_tensor = torch.from_numpy(fixed).view(
        (1, 1) + fixed.shape
    ).to(device)
    if mask is not None:
        mask_tensor = torch.from_numpy(mask).view(
            (1, 1) + fixed.shape
        ).to(device)
    learnable_affine = torch.tensor(
        init_affine[:2, :], device=device,
        requires_grad=True, dtype=torch.float64
    )
    fixed_affine = torch.tensor(
        init_affine[2:, :], device=device,
        requires_grad=False, dtype=torch.float64
    )

    lr = init_lr
    best_affine = torch.tensor(init_affine)

    for s in scales:
        optimizer = torch.optim.SGD([learnable_affine], lr=lr)
        no_improv = 0
        for e in range(epochs):
            affine = torch.cat([learnable_affine, fixed_affine])
            moved = resample(moving, fixed, affine, flip)
            moved_tensor = moved.view((1, 1) + fixed.shape)
            moved_s = func.avg_pool2d(moved_tensor, s)
            fixed_s = func.avg_pool2d(fixed_tensor, s)
            if mask is None:
                loss = loss_f(moved_s, fixed_s)
            else:
                mask_s = func.max_pool2d(
                    mask_tensor.to(torch.float32), s
                ) > 0
                loss = loss_f(moved_s, fixed_s, mask_s)
            loss_value = loss.detach().cpu().numpy().tolist()
            if loss_value < best_fit:
                final_e = e
                final_fit = loss_value
                best_fit = loss_value
                best_affine = learnable_affine.detach()
            else:
                no_improv += 1
                if no_improv == patience:
                    break
            optimizer.zero_grad()
            loss.backward()
            if e == 0:
                print('Epoch {:03d} [scale {:02d}]: {:8.4f}'.format(
                    e + 1, s, loss_value
                ))
            optimizer.step()
        learnable_affine = torch.tensor(
            best_affine.cpu().numpy(), device=device, requires_grad=True,
            dtype=torch.float64
        )
        print('Epoch {:03d} [scale {:02d}]: {:8.4f}'.format(
            final_e + 1, s, final_fit
        ))
        best_fit = np.inf
        # lr = lr / 5
    best_affine = torch.cat([learnable_affine, fixed_affine.detach()])
    return best_affine, final_e, final_fit


def resample(moving, fixed, affine, flip=False, mode='bilinear'):
    m_height, m_width = moving.shape
    f_height, f_width = fixed.shape
    image_tensor = torch.from_numpy(
        moving.astype(np.float32)
    ).view(
        (1, 1) + moving.shape
    ).to(affine.device)
    if f_width == m_width:
        x_step = 1
    else:
        x_step = m_width / f_width
    x = torch.arange(
        start=0, end=m_width, step=x_step
    ).to(dtype=torch.float64, device=affine.device)
    if f_height == m_height:
        y_step = 1
    else:
        y_step = m_height / f_height
    if flip:
        y = torch.arange(
            start=m_height - 1, end=m_height - f_height - 1, step=-y_step
        ).to(dtype=torch.float64, device=affine.device)
    else:
        y = torch.arange(
            start=0, end=m_height, step=y_step
        ).to(dtype=torch.float64, device=affine.device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        torch.ones_like(grid_x.flatten())
    ], dim=0)
    scales = torch.tensor(
        [[m_width], [m_height]],
        dtype=torch.float64, device=affine.device
    )
    affine_grid = 2 * (affine @ grid)[:2, :] / scales - 1

    tensor_grid = torch.swapaxes(affine_grid, 0, 1).view(
        1, f_height, f_width, 2
    )

    moved = func.grid_sample(
        image_tensor, tensor_grid.to(dtype=torch.float32),
        align_corners=True, mode=mode
    ).view(fixed.shape)

    return moved


def skull_registration(path, subject, net, reference=None, ref_mask=None):
    sub_code = '.'.join(subject.split('.')[:-1])

    image, data, labels, points = load_xcf(
        os.path.join(path, subject)
    )
    height, width = image.shape

    with torch.no_grad():
        data_tensor = torch.from_numpy(data).unsqueeze(0).to(net.device)
        seg = torch.sigmoid(net(data_tensor)).detach().cpu().squeeze().numpy()
    a, b, x0, y0, theta = robust_fit_ellipse(seg > 0.5)

    if reference is None:
        affine = get_affine_matrix(
            - theta, height, width, a, b, x0, y0
        )
        flip = False
        reg_image = resample(
            image, image, torch.inverse(affine), flip
        ).view(height, width).cpu().detach().numpy()
        mask = ellipse_to_mask(
            torch.tensor(a), torch.tensor(b), torch.tensor(theta),
            torch.tensor(x0), torch.tensor(y0), height, width
        ).numpy() < 0
        title_s = 'Reference [{:}]'.format(sub_code)
    else:
        r_height, r_width = reference.shape
        affine = get_affine_matrix(
            - theta, r_height, r_width, a, b, x0, y0
        )
        # Mask
        new_mask = ellipse_to_mask(
            torch.tensor(a), torch.tensor(b), torch.tensor(theta),
            torch.tensor(x0), torch.tensor(y0), height, width
        ).numpy() < 0
        if ref_mask.shape == new_mask.shape:
            mask = np.logical_and(ref_mask, new_mask).astype(np.float32)
        else:
            t_mask = resample(
                ref_mask, new_mask, torch.inverse(affine), False, 'nearest'
            ).view(height, width).cpu().detach().numpy()
            mask = np.logical_and(t_mask, new_mask).astype(np.float32)

        # Forward transformation (no flip)
        reg_fwd = resample(
            image, reference, torch.inverse(affine), False
        ).view(r_height, r_width).cpu().detach().numpy()

        # Backward transformation (with flip)
        bck_affine = get_affine_matrix(
            - (np.pi + theta), height, width, a, b, x0, y0
        )
        reg_bck = resample(
            image, reference, torch.inverse(bck_affine), True
        ).view(r_height, r_width).cpu().detach().numpy()

        # Similarity metrics
        ssim_values = [
            ssim(reg_fwd, reference),
            ssim(reg_bck, reference)
        ]
        mse_values = [
            mse(reg_fwd, reference),
            mse(reg_bck, reference)
        ]
        orientations = ['FWD', 'BCK']
        results_s = '(SSIM: {:} - MSE {:})'.format(
            orientations[np.argmax(ssim_values)],
            orientations[np.argmin(mse_values)]
        )
        flip = mse_values[-1] < mse_values[0]

        if flip:
            title_s = '[{:}] BCK {:}'.format(
                sub_code, results_s
            )
            reg_image = reg_bck
            theta = np.pi + theta
            affine = bck_affine
        else:
            title_s = '[{:}] FWD {:}'.format(
                sub_code, results_s
            )
            reg_image = reg_fwd

    e_params = (theta, a, b, x0, y0)
    point_data = (points, labels)
    image_data = (reg_image, image, mask, affine, flip)

    return image_data, point_data, e_params, title_s


def classic_registration(
        path, subject, reference, ref_mask, flip=False,
        init_affine=None, lr=1e-3, loss_f=xcor_loss
):
    sub_code = '.'.join(subject.split('.')[:-1])
    image, data, labels, points = load_xcf(
        os.path.join(path, subject)
    )

    if init_affine is None:
        init_affine = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        try:
            init_affine = torch.inverse(
                init_affine
            ).detach().cpu().numpy()
        except TypeError:
            init_affine = np.linalg.inv(init_affine)

    affine, _, final_fit = registration2d(
        reference, image, ref_mask, init_affine, flip=flip,
        init_lr=lr, loss_f=loss_f,
    )
    height, width = reference.shape
    reg_img = resample(
        image, reference, affine, flip
    ).view(height, width).cpu().detach().numpy()

    image_data = (reg_img, image, torch.inverse(affine))
    point_data = (points, labels)
    title_s = '[{:}] classic registration {:5.3f}'.format(
            sub_code, final_fit
        )

    return image_data, point_data, title_s


