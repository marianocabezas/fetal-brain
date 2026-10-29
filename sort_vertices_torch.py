"""
Pure-PyTorch replacement for the compiled `sort_vertices` CUDA extension used
by EllipseNet's rotated-IoU loss (models/Rotated_IoU/cuda_op).

Same inputs and output as sort_vertices.sort_vertices_forward:
  vertices  (B, N, M, 2) float, already centred on the polygon mean
  mask      (B, N, M)    bool, valid vertices (M = 24: 8 corners + 16 crossings)
  num_valid (B, N)       int
returns idx (B, N, 9) long: valid vertices sorted anticlockwise starting from
the +x axis, the first repeated after the last, then padded with the index
of an invalid crossing point (a zero vertex). Fewer than 3 valid vertices
gives all padding, i.e. zero area.

The result is used only through the shoelace formula, so the starting vertex
and repeated points do not change the area.
"""
import math
import torch

MAX_NUM_VERT_IDX = 9
INTERSECTION_OFFSET = 8


def sort_vertices_forward(vertices, mask, num_valid):
    B, N, M, _ = vertices.shape
    ang = torch.atan2(vertices[..., 1], vertices[..., 0])
    ang = torch.remainder(ang, 2 * math.pi)
    ang = torch.where(mask, ang, torch.full_like(ang, 10.0))   # invalid last
    order = torch.argsort(ang, dim=-1)[..., :MAX_NUM_VERT_IDX - 1]   # (B, N, 8)

    # pad = first invalid crossing point (index >= 8), as in the CUDA kernel
    inv = ~mask[..., INTERSECTION_OFFSET:]
    pad = torch.argmax(inv.int(), dim=-1) + INTERSECTION_OFFSET        # (B, N)

    nv = num_valid.long()
    pos = torch.arange(MAX_NUM_VERT_IDX, device=vertices.device).view(1, 1, -1)
    idx = torch.cat([order, order[..., :1]], dim=-1)                  # (B, N, 9)
    first = order[..., :1].expand(-1, -1, MAX_NUM_VERT_IDX)
    padv = pad.unsqueeze(-1).expand(-1, -1, MAX_NUM_VERT_IDX)
    idx = torch.where(pos == nv.unsqueeze(-1), first, idx)
    idx = torch.where(pos > nv.unsqueeze(-1), padv, idx)
    idx = torch.where((nv < 3).unsqueeze(-1), padv, idx)
    return idx.int()
