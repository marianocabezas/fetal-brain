"""
Two-head wrapper: segmentation (unchanged) + midline regression.

Wraps any existing Segmenter so it keeps producing the same segmentation logits
AND additionally predicts the midline target vector [a, b, xmin, xmax]
(degree+3 values in general). The regression head reads the encoder features via
a forward hook on the base network's target_layer(), so this works uniformly
across FCN / DeepLab / LR-ASPP / UNet / SegFormer without touching any of their
forward() methods.

By subclassing BaseModel it INHERITS the full training machinery
(fit / mini_batch_loop / observe / save_model / load_model), which all run with
self = this wrapper. Only forward, the loss lists and inference are overridden,
so training genuinely exercises the midline head.

forward(x) -> (seg_logits, midline_vec)
"""
import torch
import torch.nn as nn

from base import BaseModel
from midline import midline_loss_torch


def _to_feature_map(out):
    """Normalise whatever target_layer() emits into a (B, C, ...) tensor."""
    if isinstance(out, dict):                 # torchvision backbones -> OrderedDict
        out = list(out.values())[-1]
    if hasattr(out, 'last_hidden_state'):     # HF encoders
        out = out.last_hidden_state
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out


class MidlineHead(nn.Module):
    """Global-pool encoder features -> MLP -> midline target vector."""
    def __init__(self, n_out, hidden=256):
        super().__init__()
        self.n_out  = n_out
        self.hidden = hidden
        self.net    = None  # lazily built once channel count is known

    def _build(self, in_ch, device):
        self.net = nn.Sequential(
            nn.Linear(in_ch, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.n_out),
        ).to(device)

    def forward(self, feat):
        feat = _to_feature_map(feat)
        if feat.dim() == 3:           # (B, seq, C) transformer tokens
            pooled = feat.mean(dim=1)
        else:                         # (B, C, H, W)
            pooled = feat.mean(dim=(2, 3))
        if self.net is None:
            self._build(pooled.shape[1], pooled.device)
        return self.net(pooled)


class MidlineSegmenter(BaseModel):
    def __init__(self, base_net, degree=1, lambda_int=1.0, lambda_midline=1.0):
        super().__init__()
        self.base           = base_net
        self.degree         = degree
        self.n_mid          = degree + 3          # coeffs + xmin + xmax
        self.lambda_int     = lambda_int
        self.lambda_midline = lambda_midline
        self.device         = base_net.device
        self.channels       = base_net.channels
        self.head           = MidlineHead(self.n_mid)
        self.init           = False               # matches experiment's net.init = False

        # capture encoder features through a forward hook on the base encoder
        self._feat = None
        base_net.target_layer().register_forward_hook(self._hook)

        # eagerly build the head so its params exist BEFORE the optimizer is
        # created (a lazy build on first forward would leave it untrained).
        self._build_head(base_net)

        seg = base_net
        self.train_functions = [
            {'name': 'xe',  'weight': 1.0,
             'f': lambda p, t: seg._cross_entropy(p[0], t[0])},
            {'name': 'mid', 'weight': self.lambda_midline,
             'f': lambda p, t: midline_loss_torch(
                 p[1], t[1], self.degree, self.lambda_int)[0]},
        ]
        self.val_functions = [
            {'name': 'xe',  'weight': 1.0,
             'f': lambda p, t: seg._cross_entropy(p[0], t[0])},
            {'name': 'dsc', 'weight': 1.0,
             'f': lambda p, t: seg._dsc_loss(p[0], t[0])},
            {'name': 'mid', 'weight': self.lambda_midline,
             'f': lambda p, t: midline_loss_torch(
                 p[1], t[1], self.degree, self.lambda_int)[0]},
        ]
        self.acc_functions = {}

        # single optimiser over base + head
        self.optimizer_alg = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=base_net.optimizer_alg.param_groups[0]['lr']
        )

    def _hook(self, module, inp, out):
        self._feat = out

    def _build_head(self, base_net, probe_size=64):
        was_training = base_net.training
        base_net.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, base_net.channels, probe_size, probe_size,
                                device=self.device)
            base_net(dummy)            # fills self._feat via the hook
            self.head(self._feat)      # builds head.net with the correct in_ch
        if was_training:
            base_net.train()

    def forward(self, data):
        seg = self.base(data)          # runs encoder -> fills self._feat
        midline = self.head(self._feat)
        return seg, midline

    def inference(self, data):
        """Return ONLY the segmentation map (keeps _compute_dsc unchanged)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(data).to(self.device)
            seg, _ = self(x)
            return seg.cpu().numpy()

    def inference_midline(self, data):
        """Return the predicted midline vector(s) [a, b, xmin, xmax]."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(data).to(self.device)
            _, midline = self(x)
            return midline.cpu().numpy()
