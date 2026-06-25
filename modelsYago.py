import time
import math
import itertools
import numpy as np
from torchvision import models
from torchvision.models import segmentation as seg_models
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel, Autoencoder2D
from transformers import SegformerForSemanticSegmentation



class Classifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.train_functions = [{'name': 'xentropy', 'weight': 1, 'f': F.cross_entropy}]
        self.val_functions = [
            {'name': 'xent', 'weight': 0, 'f': F.cross_entropy},
            {'name': 'acc',  'weight': 0, 'f': self._accuracy_loss},
            {'name': 'tpr',  'weight': 1, 'f': self._tpr_loss},
            {'name': 'tnr',  'weight': 1, 'f': self._tnr_loss},
        ]

    def _accuracy_loss(self, predicted, target):
        pred_y = torch.argmax(predicted, dim=1)
        return 1 - torch.mean((pred_y == target).to(dtype=torch.float32))

    def _tpr_loss(self, predicted, target):
        mask = target == 1
        pred_y = torch.argmax(predicted, dim=1)[mask]
        return 1 - torch.mean(pred_y.to(dtype=torch.float32))

    def _tnr_loss(self, predicted, target):
        mask = target == 0
        pred_y = torch.argmax(predicted, dim=1)[mask] == 0
        return 1 - torch.mean(pred_y.to(dtype=torch.float32))

    def forward(self, *inputs):
        return None


class Segmenter(BaseModel):
    def __init__(self, n_outputs):
        super().__init__()
        self.n_classes = n_outputs
        self.train_functions = [{'name': 'xentropy', 'weight': 1, 'f': self._cross_entropy}]
        self.val_functions = [
            {'name': 'xent', 'weight': 1, 'f': self._cross_entropy},
            {'name': 'dsc',  'weight': 1, 'f': self._dsc_loss},
            {'name': 'mIoU', 'weight': 0, 'f': self._mean_iou},
        ]

    def _cross_entropy(self, predicted, target):
        return F.cross_entropy(predicted, target)

    def _intersection(self, predicted, target):
        try:
            target, roi = target
            p = torch.argmax(predicted, dim=1)
            intersection = torch.stack([
                torch.stack([
                    torch.sum(torch.logical_and(p_i[r_i] == label, t_i[r_i] == label).type_as(p))
                    for label in range(self.n_classes)
                ])
                for p_i, t_i, r_i in zip(p, target, roi)
            ])
        except ValueError:
            p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
            t = torch.flatten(target, start_dim=1).to(predicted.device)
            intersection = torch.stack([
                torch.sum(torch.logical_and(p == label, t == label).type_as(p), dim=1)
                for label in range(self.n_classes)
            ])
        return intersection

    def _union(self, predicted, target):
        try:
            target, roi = target
            p = torch.argmax(predicted, dim=1)
            union = torch.stack([
                torch.stack([
                    torch.sum(torch.logical_or(p_i[r_i] == label, t_i[r_i] == label).type_as(p))
                    for label in range(self.n_classes)
                ])
                for p_i, t_i, r_i in zip(p, target, roi)
            ])
        except ValueError:
            p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
            t = torch.flatten(target, start_dim=1).to(predicted.device)
            union = torch.stack([
                torch.sum(torch.logical_or(p == label, t == label).type_as(p), dim=1)
                for label in range(self.n_classes)
            ])
        return union

    def _sum_masks(self, predicted, target):
        try:
            target, roi = target
            p = torch.argmax(predicted, dim=1)
            sum_pred = torch.stack([
                torch.stack([torch.sum((p_i[r_i] == label).type_as(p)) for label in range(self.n_classes)])
                for p_i, r_i in zip(p, roi)
            ])
            sum_target = torch.stack([
                torch.stack([torch.sum((t_i[r_i] == label).type_as(p)) for label in range(self.n_classes)])
                for t_i, r_i in zip(target, roi)
            ])
        except ValueError:
            p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
            t = torch.flatten(target, start_dim=1)
            sum_pred   = torch.stack([torch.sum((p == label).type_as(p), dim=1) for label in range(self.n_classes)])
            sum_target = torch.stack([torch.sum((t == label).type_as(p), dim=1) for label in range(self.n_classes)])
        return sum_pred, sum_target

    def _dsc_loss(self, predicted, target):
        intersection = self._intersection(predicted, target)
        sum_pred, sum_target = self._sum_masks(predicted, target)
        dsc_k = torch.nanmean(2 * intersection / (sum_pred + sum_target), dim=0)
        dsc = 1 - torch.mean(dsc_k) if len(dsc_k) > 0 else torch.mean(0. * predicted)
        return torch.clamp(dsc, 0., 1.)

    def _mean_iou(self, predicted, target):
        intersection = self._intersection(predicted, target)
        union = self._union(predicted, target)
        miou_k = torch.nanmean(intersection / union, dim=0)
        miou = 1 - torch.mean(miou_k) if len(miou_k) > 0 else torch.mean(0. * predicted)
        return torch.clamp(miou, 0., 1.)

    def forward(self, *inputs):
        return None

    def _bn_to_eval(self, module):
        """Switch all BatchNorm layers to eval mode (avoids single-sample batch norm crash)."""
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def patch_inference(self, data, patch_size, batch_size, case=0, n_cases=1, t_start=None):
        self.eval()
        t_in = time.time()
        if t_start is None:
            t_start = t_in
        data_shape = data[0].shape[1:] if isinstance(data, tuple) else data.shape[1:]
        seg    = np.zeros((self.n_classes,) + data_shape)
        counts = np.zeros(data_shape)
        steps  = [list(range(0, lim - patch_size, patch_size // 4)) + [lim - patch_size] for lim in data_shape]
        steps_product = list(itertools.product(*steps))
        batches   = range(0, len(steps_product), batch_size)
        n_batches = len(batches)
        for bi, batch in enumerate(batches):
            slices = [
                tuple([slice(idx, idx + patch_size) for idx in indices])
                for indices in steps_product[batch:(batch + batch_size)]
            ]
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    batch_cuda = tuple(
                        torch.stack([
                            torch.from_numpy(x_i[slice(None), xs, ys]).type(torch.float32).to(self.device)
                            for xs, ys in slices
                        ]) for x_i in data
                    )
                    seg_out = self(*batch_cuda)
                else:
                    batch_cuda = torch.stack([
                        torch.from_numpy(data[slice(None), xs, ys]).type(torch.float32).to(self.device)
                        for xs, ys in slices
                    ])
                    seg_out = self(batch_cuda)
                torch.cuda.empty_cache()
            for si, (xs, ys) in enumerate(slices):
                counts[xs, ys] += 1
                seg[:, xs, ys] += seg_out[si].cpu().numpy()
            self.print_batch(bi, n_batches, case, n_cases, t_start, t_in)
        seg /= counts
        return seg


# ── helper ─────────────────────────────────────────────────────────────────────

def _make_verbose_print(network, train_f, val_f, device, verbose):
    if verbose > 1:
        print('Network created on device {:} with training losses [{:}] and validation losses [{:}]'.format(
            device,
            ', '.join(f['name'] for f in train_f),
            ', '.join(f['name'] for f in val_f),
        ))


# ── segmentation networks ──────────────────────────────────────────────────────

class FCN_ResNet50(Segmenter):
    def __init__(self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
        self.fcn = models.segmentation.fcn_resnet50(weights=weights)
        self._adapt_input(self.fcn.backbone, 'conv1', n_inputs, 64, 7, 2, 3)
        self._adapt_output(self.fcn.classifier, n_outputs)
        self._adapt_output(self.fcn.aux_classifier, n_outputs)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def _adapt_input(self, backbone, attr, n_inputs, out_ch, k, s, p):
        if n_inputs != 3:
            new_conv = nn.Conv2d(n_inputs, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
            if n_inputs > 3:
                new_conv.weight.data[:, :3, ...].copy_(getattr(backbone, attr).weight.data)
            setattr(backbone, attr, new_conv)

    def _adapt_output(self, head, n_outputs):
        in_ch = head[-1].in_channels
        head[-1] = nn.Conv2d(in_ch, n_outputs, kernel_size=1, stride=1)

    def forward(self, data):
        self.fcn.to(self.device)
        return self.fcn(data)['out']

    def target_layer(self):
        return self.fcn.backbone


class FCN_ResNet101(Segmenter):
    def __init__(self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        weights = models.segmentation.FCN_ResNet101_Weights.DEFAULT if pretrained else None
        self.fcn = models.segmentation.fcn_resnet101(weights=weights)
        self._adapt_input(n_inputs)
        self._adapt_output(n_outputs)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def _adapt_input(self, n_inputs):
        if n_inputs != 3:
            new_conv = nn.Conv2d(n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if n_inputs > 3:
                new_conv.weight.data[:, :3, ...].copy_(self.fcn.backbone.conv1.weight.data)
            self.fcn.backbone.conv1 = new_conv

    def _adapt_output(self, n_outputs):
        for head in [self.fcn.classifier, self.fcn.aux_classifier]:
            in_ch = head[-1].in_channels
            head[-1] = nn.Conv2d(in_ch, n_outputs, kernel_size=1, stride=1)

    def forward(self, data):
        self.fcn.to(self.device)
        return self.fcn(data)['out']

    def target_layer(self):
        return self.fcn.backbone


class DeeplabV3_ResNet50(Segmenter):
    def __init__(self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.dl3 = seg_models.deeplabv3_resnet50(weights=weights)
        self._adapt_input(n_inputs)
        self._adapt_output(n_outputs)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def _adapt_input(self, n_inputs):
        if n_inputs != 3:
            new_conv = nn.Conv2d(n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if n_inputs > 3:
                new_conv.weight.data[:, :3, ...].copy_(self.dl3.backbone.conv1.weight.data)
            self.dl3.backbone.conv1 = new_conv

    def _adapt_output(self, n_outputs):
        in_ch = self.dl3.classifier[-1].in_channels
        self.dl3.classifier[-1] = nn.Conv2d(in_ch, n_outputs, kernel_size=1, stride=1)

    def forward(self, data):
        self.dl3.to(self.device)
        if self.training:
            self._bn_to_eval(self.dl3)
        return self.dl3(data)['out']

    def target_layer(self):
        return self.dl3.backbone


class DeeplabV3_ResNet101(Segmenter):
    def __init__(self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        weights = seg_models.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        self.dl3 = seg_models.deeplabv3_resnet101(weights=weights)
        self._adapt_input(n_inputs)
        self._adapt_output(n_outputs)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def _adapt_input(self, n_inputs):
        if n_inputs != 3:
            new_conv = nn.Conv2d(n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if n_inputs > 3:
                new_conv.weight.data[:, :3, ...].copy_(self.dl3.backbone.conv1.weight.data)
            self.dl3.backbone.conv1 = new_conv

    def _adapt_output(self, n_outputs):
        in_ch = self.dl3.classifier[-1].in_channels
        self.dl3.classifier[-1] = nn.Conv2d(in_ch, n_outputs, kernel_size=1, stride=1)

    def forward(self, data):
        self.dl3.to(self.device)
        if self.training:
            self._bn_to_eval(self.dl3)
        return self.dl3(data)['out']

    def target_layer(self):
        return self.dl3.backbone


class DeeplabV3_MobileNet(Segmenter):
    def __init__(self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        weights = seg_models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.dl3 = seg_models.deeplabv3_mobilenet_v3_large(weights=weights)
        self._adapt_input(n_inputs)
        self._adapt_output(n_outputs)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def _adapt_input(self, n_inputs):
        if n_inputs != 3:
            new_conv = nn.Conv2d(n_inputs, 16, kernel_size=3, stride=2, padding=1, bias=False)
            if n_inputs > 3:
                new_conv.weight.data[:, :3, ...].copy_(getattr(self.dl3.backbone, '0')[0].weight.data)
            getattr(self.dl3.backbone, '0')[0] = new_conv

    def _adapt_output(self, n_outputs):
        in_ch = self.dl3.classifier[-1].in_channels
        self.dl3.classifier[-1] = nn.Conv2d(in_ch, n_outputs, kernel_size=1, stride=1)

    def forward(self, data):
        self.dl3.to(self.device)
        if self.training:
            self._bn_to_eval(self.dl3)
        return self.dl3(data)['out']

    def target_layer(self):
        return self.dl3.backbone


class LRASPP_MobileNet(Segmenter):
    def __init__(self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        weights = seg_models.LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.lraspp = seg_models.lraspp_mobilenet_v3_large(weights=weights)
        self._adapt_input(n_inputs)
        self._adapt_output(n_outputs)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def _adapt_input(self, n_inputs):
        if n_inputs != 3:
            new_conv = nn.Conv2d(n_inputs, 16, kernel_size=3, stride=2, padding=1, bias=False)
            if n_inputs > 3:
                new_conv.weight.data[:, :3, ...].copy_(getattr(self.lraspp.backbone, '0')[0].weight.data)
            getattr(self.lraspp.backbone, '0')[0] = new_conv

    def _adapt_output(self, n_outputs):
        for attr in ['high_classifier', 'low_classifier']:
            in_ch = getattr(self.lraspp.classifier, attr).in_channels
            setattr(self.lraspp.classifier, attr, nn.Conv2d(in_ch, n_outputs, kernel_size=1, stride=1))

    def forward(self, data):
        self.lraspp.to(self.device)
        if self.training:
            self._bn_to_eval(self.lraspp)
        return self.lraspp(data)['out']

    def target_layer(self):
        return self.lraspp.backbone


class Unet2D(Segmenter):
    def __init__(self, n_inputs, n_outputs, conv_filters=None, lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)
        self.channels, self.lr, self.device = n_inputs, lr, device
        self.conv_filters = conv_filters if conv_filters is not None else [32, 64, 128, 256, 512]
        self.segmenter = nn.Sequential(
            Autoencoder2D(self.conv_filters, device, n_inputs),
            nn.Conv2d(self.conv_filters[0], n_outputs, 1)
        )
        self.segmenter.to(device)
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def forward(self, data):
        self.segmenter.to(self.device)
        return self.segmenter(data)

    def target_layer(self):
        return self.segmenter[0]


class SegFormer(Segmenter):
    """
    SegFormer wrapping the HuggingFace implementation.
    Uses nvidia/mit-b2 backbone by default — b0 is faster, b4/b5 are stronger.
    Requires: pip install transformers
    """
    def __init__(self, n_inputs, n_outputs, backbone='nvidia/mit-b2', lr=1e-3,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True):
        super().__init__(n_outputs)

        self.channels, self.lr, self.device = n_inputs, lr, device

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=n_outputs,
            ignore_mismatched_sizes=True,
        )

        if n_inputs != 3:
            old_proj = self.model.segformer.encoder.patch_embeddings[0].proj
            self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
                n_inputs, old_proj.out_channels,
                kernel_size=old_proj.kernel_size, stride=old_proj.stride, padding=old_proj.padding
            )

        self.model.to(device)
        self.model.gradient_checkpointing_enable()
        self.optimizer_alg = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        _make_verbose_print(self, self.train_functions, self.val_functions, device, verbose)

    def forward(self, data):
        self.model.to(self.device)
        out = self.model(pixel_values=data).logits
        return F.interpolate(out, size=data.shape[-2:], mode='bilinear', align_corners=False)

    def target_layer(self):
        # HF refactors moved/renamed the encoder attribute across versions.
        # Hook the encoder if it's exposed, otherwise the SegformerModel itself
        # — both emit a ModelOutput with hidden_states, which _to_feature_map
        # resolves to the last-stage (B, C, H, W) feature map.
        seg = self.model.segformer
        return getattr(seg, 'encoder', seg)