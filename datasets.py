import os
import csv
import numpy as np
from PIL import Image
from matplotlib.image import imread
from torch.utils.data.dataset import Dataset

from utils import process_annotation

import torch


def ellipse_to_mask(a, b, theta, x0, y0, height, width):
    # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    A = a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2
    B = 2 * (b ** 2 - a ** 2) * np.sin(theta) * np.cos(theta) ** 2
    C = a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2
    D = -2 * A * x0 - B * y0
    E = -2 * C * y0 - B * x0
    F = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 - a ** 2 * b ** 2

    x, y = np.meshgrid(range(width), range(height))
    values = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    return values


class SkullUSDataset(Dataset):
    def __init__(
        self, data_path, csv_file, label_suffix='_Annotation.png',
        sub_list=None, im_size=None
    ):
        if im_size is None:
            self.im_size = (540, 800)
        else:
            self.im_size = im_size
        half_x = self.im_size[1] / 2
        half_y = self.im_size[0] / 2
        csv_path = os.path.join(data_path, csv_file)
        self.data = []
        self.labels = []
        self.brains = []
        self.params = []
        with open(csv_path, 'r') as csvfile:
            evalreader = csv.reader(csvfile)
            for subject, a, b, x0, y0, theta in evalreader:
                im_name = '{:}.png'.format(subject)
                if sub_list is None or im_name in sub_list:
                    im_path = os.path.join(
                        data_path, im_name
                    )
                    im = imread(im_path)
                    if im.shape == self.im_size:
                        self.data.append(imread(im_path))
                        label_name = subject + label_suffix
                        label_path = os.path.join(data_path, label_name)
                        self.labels.append(imread(label_path))
                        norm_a = (float(a) - half_x) / half_x
                        norm_b = (float(b) - half_y) / half_y
                        norm_x0 = (float(x0) - half_x) / half_x
                        norm_y0 = (float(y0) - half_y) / half_y
                        norm_theta = float(theta) / np.pi
                        self.params.append(np.array([
                            norm_a, norm_b, norm_x0, norm_y0, norm_theta
                        ]))
                        self.brains.append(
                            ellipse_to_mask(
                                float(a), float(b), float(theta),
                                float(x0), float(y0),
                                self.im_size[0], self.im_size[1]
                            ) < 0
                        )

    def __getitem__(self, index):
        image = self.data[index]
        image_norm = (image - image.mean()) / image.std()
        data = np.repeat(
            np.expand_dims(image_norm, 0), 3, axis=0
        ).astype(np.float32)
        boundary = np.expand_dims(
            self.labels[index].astype(np.float32), axis=0
        )
        brain = np.expand_dims(
            self.brains[index].astype(np.float32), axis=0
        )
        params = self.params[index].astype(np.float32)

        return data, (boundary, brain, params)

    def __len__(self):
        return len(self.data)


class FetalDataset(Dataset):
    """
    Dataset that walks a folder,
    creates all label images if
    they did not exist and store
    images and labels,
    needs a dictionary indicating the
    layers of interest as well as those that are areas
    the US image should be the layer in the final position
    """
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __getitem__(self, index):
        x = np.moveaxis(self.images[index], -1, 0).astype(np.float32)
        target = self.masks[index].astype(np.int64)

        return x, target

    def __len__(self):
        return len(self.images)


class FetalMixedDataset(Dataset):
    """
    Dataset that walks a folder,
    creates all label images if
    they did not exist and store
    images and labels,
    needs a dictionary indicating the
    layers of interest as well as those that are areas
    the US image should be the layer in the final position
    """
    def __init__(self, images, masks, points, coefficients):
        self.images = images
        self.masks = masks
        self.points = points
        self.coefficients = coefficients

    def __getitem__(self, index):
        x = np.moveaxis(self.images[index], -1, 0).astype(np.float32)
        target = (
            self.masks[index].astype(np.int64),
            (
                self.points[index].astype(np.int64),
                self.coefficients[index].astype(np.float32)
             )
        )

        return x, target

    def __len__(self):
        return len(self.images)


class EllipseDataset(Dataset):
    def __init__(self, img_paths):

        self.images = []
        self.masks = []
        self.params = []

        for f in img_paths:
            image_rgb = Image.open(f).convert('RGB')
            if image_rgb.size == (800, 540):
                ann_path = f.replace('.png', '_Annotation.png')

                # Load image
                w, h = image_rgb.size  # 800, 540 based on previous logs

                # Extract parameters using the function defined in the previous cell
                np_image = np.array(image_rgb)
                min_int = np.min(np_image, axis=(0, 1), keepdims=True)
                max_int = np.max(np_image, axis=(0, 1), keepdims=True)
                np_image = (np_image - min_int) / (max_int - min_int)
                self.images.append(np.moveaxis(np_image, -1, 0).astype(np.float32))

                data = process_annotation(ann_path)
                if data is None:
                    x0, y0, a, b, theta = torch.zeros(5, dtype=torch.float32)
                    mask = None
                else:
                    x0, y0, a, b, theta = data['params']
                    mask = data['binary']

                self.masks.append(mask)

                # Normalized parameters
                x0 = 2 * (x0 - w / 2) / w
                y0 = 2 * (y0 - h / 2) / h
                vx = np.sin(theta)
                vy = - np.cos(theta)
                wb = b / a

                d = np.sqrt(w ** 2 + h ** 2)
                # diagonal norm
                # a = a / d
                # log norm
                a = np.log(1 + a) / np.log(1 + d)

                self.params.append(torch.tensor([x0, y0, vx, vy, a, wb], dtype=torch.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        np_image = self.images[idx]
        target = (self.params[idx], self.masks[idx].astype(np.float32))
        return np_image, target
