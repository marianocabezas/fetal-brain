import cv2
import torch
import numpy as np
from gimpformats.gimpXcfDocument import GimpDocument
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize
from matplotlib.image import imread


"""
Image reading functions
"""


def load_atlas_sample(path):
    """
    This function is created to interact with US images from Girona. As such,
    we need to remove certain UX data.

    :param path: Path to the image.
    :return: data containing a numpy array ready for a network and image with
     the raw imaging data.
    """

    image = clean_ux(imread(path).copy()[..., 0])
    data = normalise_image(image)

    return data, image


def load_xcf(path):
    """
    Given the name of a xcf file, this functions reads it using GimpDocument,
    extract a list of images along with a list of layer names and returns them
    as two lists.
    :param path: Path to the image.
    :return: list of the layer names and their data (images).
    """

    # List data on groups followed by the direct children of a gimp xcf
    # document.
    project = GimpDocument(path)
    layers = project.layers
    # CAUTION: We are including the image. It's important if we only want the
    # coordinates of the annotatons, as we will need to ignore it!
    names, data = zip(*[
        (layer.name, layer.image) for layer in layers
        if not layer.isGroup
    ])
    labels = list(names)[:-1]
    annotations = list(data)[:-1]
    points = [get_points(np.array(x)) for x in annotations]
    image = clean_ux(np.array(data[-1])[..., 0])
    data = normalise_image(image)

    return image, data, labels, points


def load_vol(path):
    """
    Given the name of a vol file, this functions reads the Kretzfile step by
    step, extract a list of images along with a list of layer names and returns
    them as two lists.
    :param path: Path to the image.
    :return: list of the layer names and their data (images).
    """
    with open(path, 'rb') as f:
        images = []
        resolution = None
        f.seek(16)
        while True:
            tag_shorts = np.frombuffer(f.read(4), dtype=np.ushort)
            if len(tag_shorts) < 2:
                break
            group, element = tag_shorts
            tag_length = np.frombuffer(f.read(4), dtype=np.uint32)
            data = f.read(tag_length[0])

            if group == 0x0110 and element == 0x0002:
                print(hex(group), hex(element), str(data))
            elif group == 0xd100 and element == 0x0001:
                print(hex(group), hex(element), str(data))
            elif group == 0xc000 and element == 0x0001:
                print(
                    'Dimension X', np.frombuffer(data, dtype=np.uint16)
                )
                dim_x = np.frombuffer(data, dtype=np.uint16)[0]
            elif group == 0xc000 and element == 0x0002:
                print(
                    'Dimension Y', np.frombuffer(data, dtype=np.uint16)
                )
                dim_y = np.frombuffer(data, dtype=np.uint16)[0]
            elif group == 0xc000 and element == 0x0003:
                print(
                    'Dimension Z', np.frombuffer(data, dtype=np.uint16)
                )
                dim_z = np.frombuffer(data, dtype=np.uint16)[0]
            elif group == 0xc100 and element == 0x0001:
                resolution = np.frombuffer(data, dtype=np.float64)
                print(
                    'Resolution', resolution
                )
            elif group == 0xc200 and element == 0x0001:
                print(
                    'Offset1', np.frombuffer(data, dtype=np.float64)
                )
            elif group == 0xc200 and element == 0x0002:
                print(
                    'Offset2', np.frombuffer(data, dtype=np.float64)
                )
            elif group == 0xc300 and element == 0x0001:
                angles = np.frombuffer(data, dtype=np.float64)
                amin = angles[0]
                amax = angles[-1]
                b_centre = (amin + amax) / 2
                angles = angles - b_centre
                print(
                    r'Angles $\phi$', angles
                )
            elif group == 0xc300 and element == 0x0002:
                angles = np.frombuffer(data, dtype=np.float64)
                amin = angles[0]
                amax = angles[-1]
                b_centre = (amin + amax) / 2
                angles = angles - b_centre
                print(
                    r'Angles $\theta$', angles
                )
            elif group == 0xd000 and element != 0xffff:
                print(
                    'Image', hex(group), hex(element), tag_length / (800 * 600)
                )
                images.append(data)

    image = np.reshape(
        np.frombuffer(images[0], dtype=np.uint8), (dim_z, dim_y, dim_x)
    )

    return image, resolution


def normalise_image(image):
    """
    Function to normalise the image to its intensity z-scores. No region
    of interest is assumed.
    :param image: Ultrasound image
    :return: a normalised array ready for training
    """
    # Data normalisation and preparation for the network(s).
    image_norm = (image - image.mean()) / image.std()
    data = np.repeat(
        np.expand_dims(image_norm, 0), 3, axis=0
    ).astype(np.float32)

    return data


def clean_ux(image):
    """
    Function designed to clean the UX from a US image from Girona.

    :param image: Image with the UX.
    :return: a clean image without UX.
    """
    new_image = image.copy()
    new_image[:50, ...] = 0
    new_image[:150, -100:] = 0
    new_image[:, :40] = 0

    return new_image

def makeDict(names,points):
    """
    Given two lists of layers and another list with the lists of points in each layer at every position,
    create a dictionary with layer names as keys
    and the lists of points as values
    """
    return { names[i]: points[i] for i in range(len(names)) }


def get_points(image):
    """
    Function designed to take an image with manually drawn points and return
    the set of coordinates for their centroid.

    :param image: RGB image with the point annotations.
    :return:  numpy aray with the centroids (points x coordinates).
    """
    # We us OpenCV to binarize the image and detect the points.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray > 0] = 255

    # We use OpenCV to compute the centroids of the annotated points.
    _, _, _, centroids = cv2.connectedComponentsWithStats(gray)

    # We ignore the background centroid (first and largest one).
    return centroids[1:]


"""
Ellipse functions
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


def warp_points_ellipse(x, y, affine):
    points = torch.stack([
        x, y, torch.ones_like(x)
    ]).to(affine.device, torch.float32)
    new_points = affine.to(torch.float32) @ points
    new_x = new_points[0, :].detach().cpu()
    new_y = new_points[1, :].detach().cpu()

    return new_x, new_y


def warp_points(points, affine):
    norm_points = torch.cat([
        points, torch.ones(len(points), 1)
    ], dim=-1).to(affine.device, torch.float32)
    warped_points = affine.to(torch.float32) @ norm_points.transpose(0, 1)

    return warped_points[:2, :].transpose(0, 1).detach().cpu()


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
        mean_out_dist = np.mean(dist[dist > 0])
        std_out_dist = np.std(dist[dist > 0])
        threshold = mean_out_dist + std_out_dist
        point_mask = (dist < threshold)

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
