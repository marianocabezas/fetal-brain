import os
import csv
import numpy as np
from matplotlib.image import imread
from torch.utils.data.dataset import Dataset


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
                        norm_theta = float(theta)
                        self.params.append(np.array([
                            norm_a, norm_b, norm_x0, norm_y0, norm_theta
                        ]))

    def __getitem__(self, index):
        image = self.data[index]
        image_norm = (image - image.mean()) / image.std()
        data = np.repeat(
            np.expand_dims(image_norm, 0), 3, axis=0
        ).astype(np.float32)
        labels = np.expand_dims(self.labels[index].astype(np.float32), axis=0)
        params = self.params[index].astype(np.float32)

        return data, (labels, params)

    def __len__(self):
        return len(self.data)
