"""
Skull segmentation + principal axis pipeline.

Trains SimpleUNet on HC18 (downloading the dataset if needed), then applies
it to a folder of Girona .xcf files to compute and plot each skull's
principal axis.

Reuses existing repo code as-is:
    - datasets.SkullUSDataset to load (image, (boundary, brain, params))
    - base.BaseModel.fit() / save_model() / load_model() to train/save/load
    - utils.robust_fit_ellipse for the ellipse-fitting step (both for
      building the HC18 training CSV and for the principal-axis fit)
    - utils.load_xcf_points to read a .xcf image

Note: utils.load_xcf_points is used here instead of utils.load_xcf, because
load_xcf only returns (names, data) while registration.py's
skull_registration() unpacks 4 values (image, data, labels, points) -- that
matches load_xcf_points, not load_xcf. registration.py itself still calls
load_xcf and will raise on unpacking; that's a pre-existing bug independent
of this file.
"""
import os
import sys
import csv
import zipfile
import urllib.request

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.image import imread
from matplotlib.patches import Ellipse
from torch.utils.data import DataLoader

from models import SimpleUNet
from utils import robust_fit_ellipse, load_xcf_points, ellipse_parameters
from datasets import SkullUSDataset

DATA_URL = 'https://zenodo.org/record/1327317/files/training_set.zip?download=1'
TRAINING_PATH = 'training_set'
ZIP_NAME = 'training_set.zip'
IM_SUFFIX = 'HC.png'
CSV_NAME = 'parameters.csv'
MODEL_DIR = 'models'
MODEL_NAME = 'unet.pt'

OUTPUT_ELLIPSE_DIR = './output_ellipse'
OUTPUT_PRINCIPAL_AXIS = './output_principal_axis'
OUTPUT_SKULL_IMAGE_DIR = './output_skull_image'
OUTPUT_SKULL_DIR = './output_skull'

"""
> HC18 training helpers
"""

def download_dataset():
    if os.path.exists(TRAINING_PATH):
        print('Dataset already present at "{:}", skipping download.'.format(
            TRAINING_PATH
        ))
        return
    print('Downloading HC18 dataset from Zenodo...')
    urllib.request.urlretrieve(DATA_URL, ZIP_NAME)
    print('Extracting...')
    with zipfile.ZipFile(ZIP_NAME, 'r') as zf:
        zf.extractall('.')
    os.remove(ZIP_NAME)
    print('Done.')

def build_parameters_csv(
    training_path=TRAINING_PATH, csv_name=CSV_NAME, im_suffix=IM_SUFFIX
):
    csv_path = os.path.join(training_path, csv_name)
    if os.path.exists(csv_path):
        print('{:} already exists, skipping ellipse fitting.'.format(
            csv_path
        ))
        return

    subjects = sorted([
        f for f in os.listdir(training_path)
        if f.endswith(im_suffix)
    ])

    params = []
    for im_name in subjects:
        subject = im_name.split('.')[0]
        label_name = '{:}_Annotation.png'.format(subject)
        label_path = os.path.join(training_path, label_name)
        labels = imread(label_path)
        a, b, x0, y0, theta = robust_fit_ellipse(labels > 0)
        print('Fit', subject, a, b, x0, y0, theta)
        params.append([subject, a, b, x0, y0, theta])

    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in params:
            writer.writerow(row)

def get_dataloaders(
    training_path=TRAINING_PATH, csv_name=CSV_NAME, im_suffix=IM_SUFFIX,
    val_split=0.1, train_batch=8, test_batch=8, seed=42
):
    subjects = sorted([
        f for f in os.listdir(training_path)
        if f.endswith(im_suffix)
    ])
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_split))
    val_list = shuffled[:n_val]
    train_list = shuffled[n_val:]

    print('Training / validation samples = {:d} / {:d}'.format(
        len(train_list), len(val_list)
    ))

    train_set = SkullUSDataset(
        training_path, csv_name, sub_list=train_list
    )

    val_set = SkullUSDataset(
        training_path, csv_name, sub_list=val_list
    )

    train_loader = DataLoader(
        train_set, train_batch, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, test_batch, shuffle=False, num_workers=4
    )

    return train_loader, val_loader

def train_unet(epochs=50, patience=10, train_batch=8, test_batch=8):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_dataloaders(
        train_batch=train_batch, test_batch=test_batch
    )

    net = SimpleUNet(device=device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    net.fit(train_loader, val_loader, epochs=epochs, patience=patience)
    net.save_model(model_path)
    print('Model saved to {:}'.format(model_path))

    return net

def save_principal_axis_image(h, w, x0, y0, a, theta, output_path):
    # init
    black_image = np.zeros((h, w, 3), dtype=np.uint8)
    dx = a * np.cos(theta)
    dy = a * np.sin(theta)

    cv2.line(
        black_image,
        (int(x0 - dx), int(y0 - dy)),
        (int(x0 + dx), int(y0 + dy)),
        (0, 0, 255),
        3
    )

    cv2.imwrite(output_path, black_image)

def save_ellipse_image(h, w, a, b, x0, y0, theta, output_path):
    # init
    black_image = np.zeros((h, w, 3), dtype=np.uint8)
    dx = a * np.cos(theta)
    dy = a * np.sin(theta)
    center = (int(x0), int(y0))
    axes = (int(a), int(b))
    angle = np.degrees(theta)

    cv2.line(
        black_image,
        (int(x0 - dx), int(y0 - dy)),
        (int(x0 + dx), int(y0 + dy)),
        (0, 0, 255),
        3
    )

    cv2.ellipse(
        black_image,
        center,
        axes,
        angle,
        0,
        360,
        (0, 0, 255),
        3
    )

    cv2.circle(
        black_image,
        center,
        5,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(output_path, black_image)

def setup(xcf_path, epochs=50, patience=10):
    """
    1) Trains SimpleUNet on HC18 if models/unet.pt doesn't exist yet
       (downloading the dataset first if needed).
    2) Runs the (loaded or freshly trained) net over every .xcf file in
       xcf_path and computes its skull mask.

    Returns: (net, masks) where masks is {xcf_filename: binary_mask}.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    # create output folders
    os.makedirs(OUTPUT_SKULL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SKULL_IMAGE_DIR, exist_ok=True)

    net = SimpleUNet(device=device)
    if os.path.exists(model_path):
        net.load_model(model_path)
        print('Loaded existing UNet from {:}'.format(model_path))
    else:
        download_dataset()
        build_parameters_csv()
        net = train_unet(epochs=epochs, patience=patience)

    net.eval()

    subjects = sorted(
        f for f in os.listdir(xcf_path) if f.endswith('.xcf')
    )

    masks = {}
    for subject in subjects:
        image, data, _, _ = load_xcf_points(os.path.join(xcf_path, subject))
        plt.title(f"input data {subject}")
        # plt.imshow(data.transpose(1, 2, 0)), plt.show()
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).unsqueeze(0).to(net.device)
            seg = torch.sigmoid(
                net(data_tensor)
            ).detach().cpu().squeeze().numpy()

        # result mask convert
        mask = (seg > 0.5) * 255
        mask = mask.astype(np.uint8)

        masks[subject] = mask
        print('Segmented skull : ', subject)

        # create skull color version
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image[mask == 255] = [255, 0, 0]

        # save skull image (with US data)
        output_skull_image_path = os.path.join(OUTPUT_SKULL_IMAGE_DIR, f"result_{subject[:-4]}_skull_image.png")
        # cv2.imwrite(output_skull_image_path, image)

        # save only skull
        h, w = image.shape[:2]
        black_image = np.zeros((h, w, 3), dtype=np.uint8)
        black_image[mask == 255] = [255, 0, 0]
        output_skull_path = os.path.join(OUTPUT_SKULL_DIR, f"result_{subject[:-4]}_skull.png")
        # cv2.imwrite(output_skull_path, black_image)

    return net, masks

def principal_axis(xcf_path, subject, mask, save=True):
    """
    Given an .xcf file and its skull mask, fits the principal axis
    (robust_fit_ellipse) and plots it as a line over the original image.

    Returns: a, b, x0, y0, theta
    """
    image, _, _, _ = load_xcf_points(os.path.join(xcf_path, subject))
    a, b, x0, y0, theta = robust_fit_ellipse(mask)

    if save:
        # make folders
        os.makedirs(OUTPUT_PRINCIPAL_AXIS, exist_ok = True)
        os.makedirs(OUTPUT_ELLIPSE_DIR, exist_ok = True)

        # save principal axis image
        h, w = image.shape[:2]
        output_principal_axis_path = os.path.join(OUTPUT_PRINCIPAL_AXIS, f"result_{subject[:-4]}_principal_axis.png")
        save_principal_axis_image(h, w, x0, y0, a, theta, output_principal_axis_path)

        # save proncipal axis with ellipse image
        output_ellipse_path = os.path.join(OUTPUT_ELLIPSE_DIR, f"result_{subject[:-4]}_ellipse.png")
        save_ellipse_image(h, w, a, b, x0, y0, theta, output_ellipse_path)

    return a, b, x0, y0, theta

def write_ellipse_param_to_csv(results):
    with open("OOO_ellipse_params.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def main(xcf_path):
    net, masks = setup(xcf_path)
    ellipse_params = []
    keys = ["a","b","x0","y0","theta"]
    for subject, mask in masks.items():
        results = {}
        results["name"] = subject
        results.update(zip(keys, principal_axis(xcf_path, subject, mask)))
        ellipse_params.append(results)

    write_ellipse_param_to_csv(ellipse_params)

if __name__ == '__main__':
    xcf_folder = sys.argv[1] if len(sys.argv) > 1 else 'girona_xcf'
    main(xcf_folder)
