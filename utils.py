import torch
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize
import torch.nn.functional as func


def warp_ellipse(image, theta, height, width, a, b, center_x, center_y):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    theta = torch.tensor(theta, device=device, requires_grad=True)
    half_x = width / 2
    half_y = height / 2
    offset_x = center_x - half_x
    offset_y = center_y - half_y
    x = torch.arange(start=0, end=width, step=1)
    y = torch.arange(start=0, end=height, step=1)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid_x_c = (grid_x - center_x).to(device)
    grid_y_c = (grid_y - center_y).to(device)
    new_x = grid_x_c * torch.cos(theta) - grid_y_c * torch.sin(theta) + offset_x
    new_y = grid_y_c * torch.cos(theta) + grid_x_c * torch.sin(theta) + offset_y

    image_tensor = torch.from_numpy(image).float().view(1, 1, height, width).to(device)
    grid_tensor = torch.stack([
        new_x.float().to(device) * (2 * a) / (half_x ** 2),
        new_y.float().to(device) * (2 * b) / (half_y ** 2)
    ], dim=-1).unsqueeze(0)

    registered_image = func.grid_sample(image_tensor, grid_tensor)

    return registered_image.view(height, width).cpu().detach().numpy()


def robust_fit_ellipse(mask, n_iter=100, multiloop=5, lr=1):
    # Init
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    current_mask = mask

    # Tensor setup
    points = np.where(skeletonize(current_mask))
    points_x = points[1].astype(np.float32)
    points_y = points[0].astype(np.float32)
    best_a = 0.5 * (np.max(points_x) - np.min(points_x)).astype(np.float32)
    best_b = 0.5 * (np.max(points_y) - np.min(points_y)).astype(np.float32)
    best_x0 = np.mean(points_x).astype(np.float32)
    best_y0 = np.mean(points_y).astype(np.float32)
    best_theta = 0.
    best_fit = np.inf
    a = torch.tensor(best_a, device=device, requires_grad=True)
    b = torch.tensor(best_b, device=device, requires_grad=True)
    x0 = torch.tensor(best_x0, device=device, requires_grad=True)
    y0 = torch.tensor(best_y0, device=device, requires_grad=True)
    theta = torch.tensor(best_theta, device=device, requires_grad=True)

    for i in range(multiloop):
        optimizer = torch.optim.Adam([x0, y0, a, b, theta], lr=lr)
        for _ in range(n_iter):
            x = torch.from_numpy(points_x).to(device)
            y = torch.from_numpy(points_y).to(device)
            # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
            best_fit, best_a, best_b, best_x0, best_y0, best_theta = ellipse_step(
                optimizer, a, b, theta, x0, y0, x, y,
                best_fit, best_a, best_b, best_x0, best_y0, best_theta
            )
        a = torch.tensor(best_a, device=device, requires_grad=True)
        b = torch.tensor(best_b, device=device, requires_grad=True)
        x0 = torch.tensor(best_x0, device=device, requires_grad=True)
        y0 = torch.tensor(best_y0, device=device, requires_grad=True)
        theta = torch.tensor(best_theta, device=device, requires_grad=True)
        A, B, C, D, E, F = ellipse_parameters(a, b, theta, x0, y0)
        dist = torch.abs(
            A * (x ** 2) + B * x * y + C * (y ** 2) + D * x + E * y + F
        ).detach().cpu().numpy()
        point_mask = (dist < (np.mean(dist) + 3 * np.std(dist)))
        if np.sum(point_mask) > len(points[0]) / 2:
            idx = np.argsort(np.argsort(dist))
            point_mask = (idx + 1) < len(points[0]) / 2
        points_y = points_y[point_mask]
        points_x = points_x[point_mask]

    if best_b > best_a:
        best_a, best_b = best_b, best_a
        best_theta = np.clip(best_theta - np.pi / 2, - np.pi / 2, np.pi / 2)
    return best_a, best_b, best_x0, best_y0, best_theta


def fit_ellipse(mask, lr=1):
    points = np.where(binary_dilation(mask, iterations=2))
    points_x = points[1].astype(np.float32)
    points_y = points[0].astype(np.float32)
    # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    best_a = 0.5 * (np.max(points_x) - np.min(points_x)).astype(np.float32)
    best_b = 0.5 * (np.max(points_y) - np.min(points_y)).astype(np.float32)
    best_x0 = np.mean(points[1]).astype(np.float32)
    best_y0 = np.mean(points[0]).astype(np.float32)
    best_theta = 0.
    best_fit = np.inf
    n_iter = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    a = torch.tensor(best_a, device=device, requires_grad=True)
    b = torch.tensor(best_b, device=device, requires_grad=True)
    x0 = torch.tensor(best_x0, device=device, requires_grad=True)
    y0 = torch.tensor(best_y0, device=device, requires_grad=True)
    theta = torch.tensor(best_theta, device=device, requires_grad=True)

    x = torch.from_numpy(points[1]).to(device)
    y = torch.from_numpy(points[0]).to(device)

    optimizer = torch.optim.Adam([a, b, theta], lr=lr)
    for i in range(n_iter):
        best_fit, best_a, best_b, best_x0, best_y0, best_theta = ellipse_step(
            optimizer, a, b, theta, x0, y0, x, y,
            best_fit, best_a, best_b, best_x0, best_y0, best_theta
        )
    return best_a, best_b, best_x0, best_y0, best_theta


def ellipse_step(
    optimizer, a, b, theta, x0, y0, x, y,
    best_fit, best_a, best_b, best_x0, best_y0, best_theta
):
    new_theta = torch.clamp(theta, - np.pi / 2, np.pi / 2)
    A, B, C, D, E, F = ellipse_parameters(a, b, new_theta, x0, y0)
    loss = torch.norm(
        A * (x ** 2) + B * x * y + C * (y ** 2) + D * x + E * y + F
    )
    loss_value = loss.detach().cpu().numpy().tolist()
    if loss_value < best_fit:
        best_fit = loss_value
        best_a = a.detach().cpu().numpy().tolist()
        best_b = b.detach().cpu().numpy().tolist()
        best_x0 = x0.detach().cpu().numpy().tolist()
        best_y0 = y0.detach().cpu().numpy().tolist()
        best_theta = new_theta.detach().cpu().numpy().tolist()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return best_fit, best_a, best_b, best_x0, best_y0, best_theta


def ellipse_to_mask(a, b, theta, x0, y0, height, width):
    # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    A, B, C, D, E, F = ellipse_parameters(a, b, theta, x0, y0)

    x, y = np.meshgrid(range(width), range(height))
    values = A * (x ** 2) + B * x * y + C * (y ** 2) + D * x + E * y + F
    return values


def ellipse_parameters(a, b, theta, x0, y0):
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    a_2 = a ** 2
    b_2 = b ** 2
    x0_2 = x0 ** 2
    y0_2 = y0 ** 2
    cos2_theta = cos_theta ** 2
    sin2_theta = sin_theta ** 2
    A = a_2 * sin2_theta + b_2 * cos2_theta
    B = 2 * (b_2 - a_2) * sin_theta * cos2_theta
    C = a_2 * cos2_theta + b_2 * sin2_theta
    D = -2 * A * x0 - B * y0
    E = -2 * C * y0 - B * x0
    F = A * x0_2 + B * x0 * y0 + C * y0_2 - a_2 * b_2

    return A, B, C, D, E, F
