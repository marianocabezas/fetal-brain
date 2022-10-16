from torchvision import models
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
