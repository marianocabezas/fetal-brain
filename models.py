import time
import itertools
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from base import BaseModel, Autoencoder2D, ResConv2dBlock
from criteria import tp_binary_loss, tn_binary_loss, dsc_binary_loss


def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


class ConvNeXtTiny(BaseModel):
    def __init__(
        self, n_outputs=5, pretrained=True, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__()
        # Init
        self.n_classes = n_outputs
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                self.cnext = models.convnext_tiny(weights='IMAGENET1K_V1')
            except TypeError:
                self.cnext = models.convnext_tiny(pretrained)
        else:
            self.cnext = models.convnext_tiny()
        last_features = self.cnext.classifier[-1].in_features
        self.cnext.classifier[-1] = nn.Linear(last_features, self.n_classes)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(p, t[2])
            },
            {
                'name': 'seg',
                'weight': 1,
                'f': self.segmentation_loss
            }
        ]

        self.val_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(p, t[2])
            },
            {
                'name': 'seg',
                'weight': 1,
                'f': self.segmentation_loss
            }
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def segmentation_loss(self, pred, target, x_size=540, y_size=800):
        target_seg = 1 - target[0].float().squeeze(1)
        half_x = x_size / 2
        half_y = y_size / 2
        norm_a = (pred[:, 0]).view(-1, 1, 1)
        norm_b = (pred[:, 1]).view(-1, 1, 1)
        norm_x0 = (pred[:, 2]).view(-1, 1, 1)
        norm_y0 = (pred[:, 3]).view(-1, 1, 1)
        theta = (pred[:, 4]).view(-1, 1, 1)

        a = norm_a * half_x + half_x
        b = norm_b * half_y + half_y
        x0 = norm_x0 * half_x + half_x
        y0 = norm_y0 * half_y + half_y

        Ap = a ** 2 * torch.sin(theta) ** 2 + b ** 2 * torch.cos(theta) ** 2
        Bp = 2 * (b ** 2 - a ** 2) * torch.sin(theta) * torch.cos(theta) ** 2
        Cp = a ** 2 * torch.cos(theta) ** 2 + b ** 2 * torch.sin(theta) ** 2
        Dp = -2 * Ap * x0 - Bp * y0
        Ep = -2 * Cp * y0 - Bp * x0
        Fp = Ap * x0 ** 2 + Bp * x0 * y0 + Cp * y0 ** 2 - a ** 2 * b ** 2

        x, y = torch.meshgrid(
            torch.arange(end=x_size), torch.arange(end=y_size),
            indexing='ij'
        )
        x = torch.repeat_interleave(
            x.unsqueeze(dim=0), len(pred), dim=0
        ).to(self.device)
        y = torch.repeat_interleave(
            y.unsqueeze(dim=0), len(pred), dim=0
        ).to(self.device)

        values = Ap * x * x + Bp * x * y + Cp * y * y + Dp * x + Ep * y + Fp
        abs_values = torch.abs(values) * 10
        # max_values, _ = torch.max(abs_values, dim=0, keepdim=True)
        # norm_values = abs_values / max_values
        norm_values = abs_values / torch.sqrt(1 + abs_values * abs_values)

        return F.binary_cross_entropy(norm_values, target_seg)

    def forward(self, data):
        self.cnext.to(self.device)
        params = self.cnext(data)
        coords = torch.clamp(params[:, :-1], -1, 1)
        theta = torch.clamp(params[:, -1:], - np.pi / 2, np.pi / 2)
        return torch.cat([coords, theta], dim=-1)


class SimpleUNet(BaseModel):
    def __init__(
        self,
        conv_filters=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_images=3,
        verbose=True,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 32, 128, 256, 256, 1024, 1024]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.segmenter = nn.Sequential(
            Autoencoder2D(
                self.conv_filters, device, n_images, block=ResConv2dBlock,
                norm=norm_f
            ),
            nn.Conv2d(self.conv_filters[0], 1, 1)
        )
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t[0])
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(p, t[0])
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t[0])
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t[0])
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t[0])
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        return self.segmenter(data)


class Segmenter(BaseModel):
    def __init__(self, n_outputs):
        super().__init__()
        self.n_classes = n_outputs

        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': self._cross_entropy
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': self._cross_entropy
            },

            {
                'name': 'dsc',
                'weight': 1,
                'f': self._dsc_loss
            },
            {
                'name': 'mIoU',
                'weight': 0,
                'f': self._mean_iou
            },
        ]

    def _cross_entropy(self, predicted, target):
        try:
            target, roi = target
            predicted = torch.stack([
                p_i.squeeze(1)[roi]
                for p_i in torch.split(predicted, 1, dim=1)
            ], dim=1)
            target = target[roi]
        except ValueError:
            pass

        return F.cross_entropy(predicted, target)

    def _intersection(self, predicted, target):
        try:
            target, roi = target
            p = torch.argmax(predicted, dim=1)
            intersection = torch.stack([
                torch.stack([
                    torch.sum(
                        torch.logical_and(
                            p_i[r_i] == label, t_i[r_i] == label
                        ).type_as(p)
                    )
                    for label in range(self.n_classes)
                ])
                for p_i, t_i, r_i in zip(p, target, roi)
            ])
        except ValueError:
            p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
            t = torch.flatten(target, start_dim=1).to(predicted.device)
            intersection = torch.stack([
                torch.sum(
                    torch.logical_and(p == label, t == label).type_as(p),
                    dim=1
                )
                for label in range(self.n_classes)
            ])

        return intersection

    def _union(self, predicted, target):
        try:
            target, roi = target
            p = torch.argmax(predicted, dim=1)
            union = torch.stack([
                torch.stack([
                    torch.sum(
                        torch.logical_or(
                            p_i[r_i] == label, t_i[r_i] == label
                        ).type_as(p)
                    )
                    for label in range(self.n_classes)
                ])
                for p_i, t_i, r_i in zip(p, target, roi)
            ])
        except ValueError:
            p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
            t = torch.flatten(target, start_dim=1).to(predicted.device)
            union = torch.stack([
                torch.sum(
                    torch.logical_or(p == label, t == label).type_as(p),
                    dim=1
                )
                for label in range(self.n_classes)
            ])

        return union

    def _sum_masks(self, predicted, target):
        try:
            target, roi = target
            p = torch.argmax(predicted, dim=1)
            sum_pred = torch.stack([
                torch.stack([
                    torch.sum((p_i[r_i] == label).type_as(p))
                    for label in range(self.n_classes)
                ])
                for p_i, r_i in zip(p, roi)
            ])
            sum_target = torch.stack([
                torch.stack([
                    torch.sum((t_i[r_i] == label).type_as(p))
                    for label in range(self.n_classes)
                ])
                for t_i, r_i in zip(target, roi)
            ])
        except ValueError:
            p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
            t = torch.flatten(target, start_dim=1)
            sum_pred = torch.stack([
                torch.sum((p == label).type_as(p), dim=1)
                for label in range(self.n_classes)
            ])
            sum_target = torch.stack([
                torch.sum((t == label).type_as(p), dim=1)
                for label in range(self.n_classes)
            ])
        return sum_pred, sum_target

    def _dsc_loss(self, predicted, target):
        intersection = self._intersection(predicted, target)
        sum_pred, sum_target = self._sum_masks(predicted, target)
        dsc_k = torch.nanmean(
            2 * intersection / (sum_pred + sum_target), dim=0
        )
        if len(dsc_k) > 0:
            dsc = 1 - torch.mean(dsc_k)
        else:
            dsc = torch.mean(0. * predicted)

        return torch.clamp(dsc, 0., 1.)

    def _mean_iou(self, predicted, target):
        intersection = self._intersection(predicted, target)
        union = self._union(predicted, target)
        miou_k = torch.nanmean(intersection / union, dim=0)
        if len(miou_k) > 0:
            miou = 1 - torch.mean(miou_k)
        else:
            miou = torch.mean(0. * predicted)

        return torch.clamp(miou, 0., 1.)

    def forward(self, *inputs):
        return None

    def patch_inference(
        self, data, patch_size, batch_size, case=0, n_cases=1, t_start=None
    ):
        # Init
        self.eval()

        # Init
        t_in = time.time()
        if t_start is None:
            t_start = t_in

        # This branch is only used when images are too big. In this case
        # they are split in patches and each patch is trained separately.
        # Currently, the image is partitioned in blocks with no overlap,
        # however, it might be a good idea to sample all possible patches,
        # test them, and average the results. I know both approaches
        # produce unwanted artifacts, so I don't know.
        # Initial results. Filled to 0.
        if isinstance(data, tuple):
            data_shape = data[0].shape[1:]
        else:
            data_shape = data.shape[1:]
        seg = np.zeros((self.n_classes,) + data_shape)
        counts = np.zeros(data_shape)

        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        steps = [
            list(
                range(0, lim - patch_size, patch_size // 4)
            ) + [lim - patch_size]
            for lim in data_shape
        ]

        steps_product = list(itertools.product(*steps))
        batches = range(0, len(steps_product), batch_size)
        n_batches = len(batches)

        # The following code is just a normal test loop with all the
        # previously computed patches.
        for bi, batch in enumerate(batches):
            # Here we just take the current patch defined by its slice
            # in the x and y axes. Then we convert it into a torch
            # tensor for testing.
            slices = [
                tuple([slice(idx, idx + patch_size) for idx in indices])
                for indices in steps_product[batch:(batch + batch_size)]
            ]

            # Testing itself.
            with torch.no_grad():
                if isinstance(data, list) or isinstance(data, tuple):
                    batch_cuda = tuple(
                        torch.stack([
                            torch.from_numpy(
                                x_i[slice(None), xslice, yslice]
                            ).type(torch.float32).to(self.device)
                            for xslice, yslice in slices
                        ])
                        for x_i in data
                    )
                    seg_out = self(*batch_cuda)
                else:
                    batch_cuda = torch.stack([
                        torch.from_numpy(
                            data[slice(None), xslice, yslice]
                        ).type(torch.float32).to(self.device)
                        for xslice, yslice in slices
                    ])
                    seg_out = self(batch_cuda)

            # Then we just fill the results image.
            for si, (xslice, yslice) in enumerate(slices):
                counts[xslice, yslice] += 1
                seg_bi = seg_out[si].cpu().numpy()
                seg[:, xslice, yslice] += seg_bi

            # Printing
            self.print_batch(bi, n_batches, case, n_cases, t_start, t_in)

        seg /= counts

        return seg


class FCN_ResNet50(Segmenter):
    def __init__(
        self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__(n_outputs)
        # Init
        self.channels = n_inputs
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT
                self.fcn = models.segmentation.fcn_resnet50(weights=weights)
            except TypeError:
                self.fcn = models.segmentation.fcn_resnet50(pretrained)
        else:
            self.fcn = models.segmentation.fcn_resnet50()
        if n_inputs > 3:
            conv_input = nn.Conv2d(
                n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # We assume that RGB channels will be the first 3
            conv_input.weight.data[:, :3, ...].copy_(
                self.fcn.backbone.conv1.weight.data
            )
            self.fcn.backbone.conv1 = conv_input
        elif n_inputs < 3:
            self.fcn.backbone.conv1 = nn.Conv2d(
                n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.last_features = self.fcn.classifier[-1].in_channels
        self.fcn.classifier[-1] = nn.Conv2d(
            self.last_features, n_outputs, kernel_size=1, stride=1
        )
        self.aux_last_features = self.fcn.aux_classifier[-1].in_channels
        self.fcn.aux_classifier[-1] = nn.Conv2d(
            self.aux_last_features, n_outputs, kernel_size=1, stride=1
        )

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.fcn.to(self.device)
        return self.fcn(data)['out']

    def target_layer(self):
        return self.fcn.backbone


class FCN_ResNet101(Segmenter):
    def __init__(
        self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__(n_outputs)
        # Init
        self.channels = n_inputs
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                weights = models.segmentation.FCN_ResNet101_Weights.DEFAULT
                self.fcn = models.segmentation.fcn_resnet101(weights=weights)
            except TypeError:
                self.fcn = models.segmentation.fcn_resnet101(pretrained)
        else:
            self.fcn = models.segmentation.fcn_resnet101()
        if n_inputs > 3:
            conv_input = nn.Conv2d(
                n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # We assume that RGB channels will be the first 3
            conv_input.weight.data[:, :3, ...].copy_(
                self.fcn.backbone.conv1.weight.data
            )
            self.fcn.backbone.conv1 = conv_input
        elif n_inputs < 3:
            self.fcn.backbone.conv1 = nn.Conv2d(
                n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.last_features = self.fcn.classifier[-1].in_channels
        self.fcn.classifier[-1] = nn.Conv2d(
            self.last_features, n_outputs, kernel_size=1, stride=1
        )
        self.aux_last_features = self.fcn.aux_classifier[-1].in_channels
        self.fcn.aux_classifier[-1] = nn.Conv2d(
            self.aux_last_features, n_outputs, kernel_size=1, stride=1
        )

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.fcn.to(self.device)
        return self.fcn(data)['out']

    def target_layer(self):
        return self.fcn.backbone