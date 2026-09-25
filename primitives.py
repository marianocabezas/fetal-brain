import argparse
import os
import time
import gzip
import pickle
import numpy as np
import pydicom as dicom
from pathlib import Path
from copy import deepcopy
from time import strftime
from functools import partial
import os
import glob
from PIL import Image
from skimage.morphology import skeletonize
from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm
import pydicom as dicom
from matplotlib.patches import Ellipse # Added this import
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import find_contours
from matplotlib.patches import Ellipse # Added this import
from collections import OrderedDict

from datasets import EllipseDataset
from utils import process_annotation
from criteria import ellipse_loss, line_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms


def plot_ellipse(ax, img, x0, y0, vx, vy, a, b, color, num_steps=100000):
    try:
        w, h = img.size
    except TypeError:
        h, w, _ = img.shape
    ax.imshow(img)

    # Ellipse
    t = np.linspace(0, 2 * np.pi, num_steps)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_theta = vx  # - vy
    sin_theta = vy  # vx

    # Standard parametric form
    x = a * cos_t
    y = b * sin_t

    # Rotation and translation
    xe = x0 + x * cos_theta - y * sin_theta
    ye = y0 + x * sin_theta + y * cos_theta

    ax.scatter([xe], [ye], color=color, label='Ellipse', s=1)

    ax.scatter([x0], [y0], color='yellow', label='Midpoint')
    mjr_x1 = x0 - a * vx
    mjr_y1 = y0 - a * vy
    ax.quiver(
        mjr_x1, mjr_y1, 2 * a * vx, 2 * a * vy,
        scale_units='xy', scale=1,
        color='teal', angles='xy'
    )
    mnr_x1 = x0 + b * vy
    mnr_y1 = y0 - b * vx
    ax.quiver(
        mnr_x1, mnr_y1, - 2 * b * vy, 2 * b * vx,
        scale_units='xy', scale=1,
        color='cyan', angles='xy'
    )

    m = vy / vx
    x_ini = 0
    y_ini = y0 - m * x0
    x_end = w
    y_end = (w - x0) * m + y0

    ax.plot([x_ini, x_end], [y_ini, y_end], 'b--')
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.set_axis_off()


def output_to_params(image, data_tensor):
    h, w, _ = image.shape

    # > Prediction
    x0, y0, vx, vy, a, wb = data_tensor

    x0 = w * x0.item() / 2 + w / 2
    y0 = h * y0.item() / 2 + h / 2
    vx = vx.item()
    vy = vy.item()
    a = a.item() * 100
    b = a * wb.item()

    return x0, y0, vx, vy, a, b


def main():
    parser = argparse.ArgumentParser()
    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    n_points = 100000
    n_epochs = 1000
    input_path = ''
    output_path = ''

    image_files = sorted(glob.glob('training_set/training_set/*_HC.png'))

    for img_path in image_files[:20]:
        subj_code, _ = os.path.basename(img_path).split('_')
        ann_path = img_path.replace('.png', '_Annotation.png')
        if not os.path.exists(ann_path): continue

        img = Image.open(img_path).convert('RGB')
        data = process_annotation(ann_path)
        if data is None: continue

        x0, y0, a, b, theta = data['params']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].contour(data['binary'], colors='cyan', linewidths=0.5)
        axes[0].set_title(f"Image: {os.path.basename(img_path)}")
        axes[0].set_axis_off()

        plot_ellipse(
            axes[1],
            img, x0, y0, np.sin(theta), -np.cos(theta), a, b,
            'green'
        )

        axes[1].set_title(
            f"PyTorch Fit: a={a:.1f}, b={b:.1f}, θ={np.rad2deg(theta):.4f}"
        )
        plt.show()
        plt.savefig(os.path.join(output_path, f'{subj_code}_ellipse.png'))
        plt.close()

    # > Deep learning part
    full_dataset = EllipseDataset(image_files)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [0.64, 0.16, 0.2]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Segment first > Fine-tune ellipse
    seg_model = models.segmentation.fcn_resnet50(weights='DEFAULT')
    last_features = seg_model.classifier[-1].in_channels
    seg_model.classifier[-1] = nn.Conv2d(
        last_features, 1, kernel_size=1, stride=1
    )
    seg_model.to(device)

    ellipse_model = nn.Sequential(
        OrderedDict([
            ("flattening", nn.Sequential(
                seg_model.classifier[0],
                seg_model.classifier[1],
                seg_model.classifier[2],
                seg_model.classifier[3],
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten()
            )),
            ("mlp", nn.Sequential(
                nn.Linear(
                    in_features=last_features,
                    out_features=last_features * 8,
                    bias=True
                ),
                nn.GELU(approximate='none'),
                nn.LayerNorm(
                    (last_features * 8,),
                    eps=1e-06, elementwise_affine=True
                ),
                nn.Linear(
                    in_features=last_features * 8,
                    out_features=last_features // 2,
                    bias=True
                ),
                nn.GELU(approximate='none'),
                nn.LayerNorm(
                    (last_features // 2,),
                    eps=1e-06, elementwise_affine=True
                ),
                nn.Linear(last_features // 2, 6)
            ))
        ])
    )

    ellipse_model.to(device)

    # Ellipse model
    # model = models.resnet50(weights='DEFAULT')
    # model.fc = nn.Linear(model.fc.in_features, 6)
    # model = model.to(device)

    final_model = nn.ModuleList([
        seg_model,
        ellipse_model
    ])

    print(
        'Final model has {:,d} parameters'.format(sum(p.numel() for p in final_model.parameters() if p.requires_grad)))

    criterion = ellipse_loss
    seg_criterion = F.binary_cross_entropy_with_logits
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    seg_optimizer = optim.Adam(seg_model.parameters(), lr=1e-3)
    reg_optimizer = optim.AdamW(ellipse_model.parameters(), lr=1e-4)

    # Training Loop > SEGMENTATION

    train_loop = tqdm(
        range(20), leave=False,
        desc='Epoch loop (segmentation)',
    )
    best_loss = np.inf
    best_e = 0
    best_state = deepcopy(seg_model.state_dict())
    for epoch in train_loop:  # Increased epochs slightly
        running_loss = 0.0
        batch_loop = tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False,
            desc='Training loop (segmentation)',
        )
        seg_model.train()
        for b_i, (images, (ellipse, masks)) in batch_loop:
            images = images.to(device)
            ellipse = ellipse.to(device)
            masks = masks.to(device)
            seg_optimizer.zero_grad()
            outputs = seg_model(images)['out'].squeeze(1)
            loss = seg_criterion(outputs, masks)
            loss.backward()
            seg_optimizer.step()
            running_loss += loss.item()
            batch_loop.set_postfix(
                current_loss=f'{loss.item():.5f}',
                running_loss=f'{running_loss / (b_i + 1):.5f}',
            )
        train_loss = running_loss / len(train_loader)

        running_loss = 0.0
        val_loop = tqdm(
            enumerate(val_loader), total=len(val_loader), leave=False,
            desc='Validation loop (segmentation)',
        )
        seg_model.eval()
        with torch.no_grad():
            for b_i, (images, (ellipse, masks)) in val_loop:
                images = images.to(device)
                ellipse = ellipse.to(device)
                masks = masks.to(device)
                outputs = seg_model(images)['out'].squeeze(1)
                loss = seg_criterion(outputs, masks)
                running_loss += loss.item()
                val_loop.set_postfix(
                    current_loss=f'{loss.item():.5f}',
                    running_loss=f'{running_loss / (b_i + 1):.5f}',
                )
        val_loss = running_loss / len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_e = epoch
            best_state = deepcopy(seg_model.state_dict())
        train_loop.set_postfix(
            train_loss=f'{train_loss:.5f}',
            val_loss=f'{val_loss:.5f}',
            best_loss=f'{best_loss:.5f}',
            best_epoch=best_e
        )
        print(
            f"Epoch {epoch + 1:02d} (segmentation) | training Loss: {train_loss:.5f} | validation Loss: {val_loss:.5f} ")

    seg_model.load_state_dict(best_state)

    print(f"Best epoch {best_e + 1:02d} (segmentation) - validation Loss: {best_loss:.5f} ")

    # Training Loop > REGRESSION

    train_loop = tqdm(
        range(n_epochs), leave=False,
        desc='Epoch loop (regression)',
    )
    best_loss = np.inf
    best_epoch = 0
    best_s = deepcopy(seg_model.state_dict())
    best_e = deepcopy(ellipse_model.state_dict())
    for epoch in train_loop:  # Increased epochs slightly
        running_loss = 0.0
        batch_loop = tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False,
            desc='Training loop (regression)',
        )
        ellipse_model.train()
        for b_i, (images, (ellipse, masks)) in batch_loop:
            images = images.to(device)
            ellipse = ellipse.to(device)
            masks = masks.to(device)
            reg_optimizer.zero_grad()
            features = seg_model.backbone(images)['out']
            outputs = ellipse_model(features)
            loss = criterion(outputs, ellipse)
            loss.backward()
            reg_optimizer.step()
            running_loss += loss.item()
            batch_loop.set_postfix(
                current_loss=f'{loss.item():.5f}',
                running_loss=f'{running_loss / (b_i + 1):.5f}',
            )
        train_loss = running_loss / len(train_loader)

        running_loss = 0.0
        val_loop = tqdm(
            enumerate(val_loader), total=len(val_loader), leave=False,
            desc='Validation loop (regression)',
        )
        ellipse_model.eval()
        with ((torch.no_grad())):
            for b_i, (images, (ellipse, masks)) in val_loop:
                images = images.to(device)
                ellipse = ellipse.to(device)
                masks = masks.to(device)
                features = seg_model.backbone(images)['out']
                outputs = ellipse_model(features)
                loss = criterion(outputs, ellipse)
                running_loss += loss.item()
                val_loop.set_postfix(
                    current_loss=f'{loss.item():.5f}',
                    running_loss=f'{running_loss / (b_i + 1):.5f}',
                )
        val_loss = running_loss / len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_s = deepcopy(seg_model.state_dict())
            best_e = deepcopy(ellipse_model.state_dict())
        train_loop.set_postfix(
            last_train=f'{train_loss:.5f}',
            last_val=f'{val_loss:.5f}',
            best_loss=f'{best_loss:.5f}',
            best_epoch=best_epoch
        )
        print(
            f"Epoch {epoch + 1:02d} (regression) | training Loss: {train_loss:.5f} | validation Loss: {val_loss:.5f} ")

    # seg_model.load_state_dict(best_s)
    ellipse_model.load_state_dict(best_e)

    print(f"Best epoch {best_epoch + 1:02d} (regression) - validation Loss: {best_loss:.5f} ")


    # Evaluation
    ellipse_model.eval()
    test_loss = 0.0
    plot_i = 0
    max_plots = 50

    with torch.no_grad():
        for images, (ellipse, masks) in test_loader:
            images, ellipses, masks = images.to(device), ellipse.to(device), masks.to(device)
            features = seg_model.backbone(images)['out']
            outputs = ellipse_model(features)
            loss = criterion(outputs, ellipses)
            test_loss += loss.item()
            for image_tensor, output, target in zip(images, outputs, ellipses):
                print('Prediction (normalised)', output)
                print('Ground truth (normalised)', target)
                _, h, w = image_tensor.shape
                print('800 x 540', w, h)

                if plot_i < max_plots:
                    np_image = image_tensor.detach().cpu().numpy()
                    plot_image = np.moveaxis(np_image, 0, -1)

                    # > Prediction
                    p_x0, p_y0, p_vx, p_vy, p_a, p_b = output_to_params(plot_image, output)
                    print('Prediction', p_x0, p_y0, p_vx, p_vy, p_a, p_b)

                    # > Ground truth
                    t_x0, t_y0, t_vx, t_vy, t_a, t_b = output_to_params(plot_image, target)
                    print('Ground truth', t_x0, t_y0, t_vx, t_vy, t_a, t_b)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    plot_ellipse(axes[0], plot_image, p_x0, p_y0, p_vx, p_vy, p_a, p_b, 'red')
                    plot_ellipse(axes[1], plot_image, t_x0, t_y0, t_vx, t_vy, t_a, t_b, 'green')

                    plot_i += 1

        print(f"Testing ellipse error: {test_loss / len(test_loader):.5f}")

if __name__ == '__main__':
    main()
