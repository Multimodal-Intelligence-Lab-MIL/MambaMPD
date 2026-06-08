"""Edge-Guided Attention (EGA) module.

Faithful implementation of Section 3.4 (Eq. 5-7) and the Table 1 pseudo-code.

EGA refines an encoder feature ``f_e`` using an auxiliary decoder prediction
``f_d`` (a single-channel deep-supervision output bilinearly aligned to the
encoder resolution).  The pipeline is:

* **Phase I - multi-scale edge prior.**  A Laplacian operator ``L`` is applied to
  ``f_d`` at scales {1, 2, 4} (via average pooling + upsampling) and the
  responses are aggregated and normalised into a single-channel edge map
  ``f_edge`` (Table 1, steps 1-3).
* **Global / local extractors.**  Complementary global (broad contextual
  saliency) and local (fine structural detail) responses are derived from the
  encoder feature, combined and modulated by the edge prior:
  ``f_m = (f_g + f_loc) * f_edge``  (Eq. 6).
* **Phase III - fusion + refinement.**  ``f_m`` is concatenated with the encoder
  feature, refined by a 3x3 convolution and added back via a residual
  connection (Eq. 7), then recalibrated by an Efficient Channel Attention (ECA)
  block whose descriptor fuses a global term with an edge-aware term
  (``z = lambda * z_glob + (1 - lambda) * z_edge``; Table 1, Phase II).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _laplacian_kernel(device, dtype):
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=device, dtype=dtype)
    return k.view(1, 1, 3, 3)


def laplacian(x):
    """Apply the fixed Laplacian edge operator L to a single-channel map."""
    kernel = _laplacian_kernel(x.device, x.dtype)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel)


class MultiScaleEdgePrior(nn.Module):
    """Phase I: build a normalised multi-scale Laplacian edge prior from f_d."""

    def __init__(self, scales=(1, 2, 4), eps=1e-5):
        super().__init__()
        self.scales = scales
        self.eps = eps

    def forward(self, pred):
        # pred: (B, 1, H, W), an auxiliary decoder prediction (sigmoid space).
        H, W = pred.shape[-2:]
        responses = []
        for k in self.scales:
            if k == 1:
                e = laplacian(pred)
            else:
                pooled = F.avg_pool2d(pred, kernel_size=k)
                e = F.interpolate(laplacian(pooled), size=(H, W), mode="bilinear", align_corners=False)
            responses.append(e)
        edge = torch.stack(responses, dim=0).mean(dim=0)
        # Normalise to [0, 1] per-sample for stable modulation.
        flat = edge.flatten(1)
        mn = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        mx = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        edge = (edge - mn) / (mx - mn + self.eps)
        return edge


class GlobalFeatureExtractor(nn.Module):
    """Captures broad contextual saliency via global-context channel gating."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.bn(self.conv(x))
        gate = torch.sigmoid(self.gap(y))
        return y * gate


class LocalFeatureExtractor(nn.Module):
    """Preserves local structural detail via a depthwise 3x3 convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EdgeAwareECA(nn.Module):
    """Efficient Channel Attention with an edge-aware channel descriptor.

    Computes a global average descriptor ``z_glob`` and an edge-weighted
    descriptor ``z_edge``, fuses them with a learnable balance ``lambda`` and
    obtains channel weights via a lightweight 1D convolution (Table 1, Phase II).
    """

    def __init__(self, channels, gamma=2, b=1, eps=1e-5):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        self.eps = eps

    def forward(self, x, edge):
        B, C, _, _ = x.shape
        z_glob = x.mean(dim=(2, 3))  # (B, C)
        w = edge  # (B, 1, H, W)
        z_edge = (x * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + self.eps)  # (B, C)
        lamb = torch.sigmoid(self.lamb)
        z = lamb * z_glob + (1.0 - lamb) * z_edge  # (B, C)
        attn = self.conv(z.unsqueeze(1)).squeeze(1)  # (B, C)
        attn = torch.sigmoid(attn).view(B, C, 1, 1)
        return x * attn


class EGA(nn.Module):
    """Edge-Guided Attention at a single decoder/encoder stage.

    Args:
        channels: channel count of the encoder feature ``f_e``; the refined
            output keeps the same channel count so it can replace the skip
            connection feeding the decoder.
    """

    def __init__(self, channels):
        super().__init__()
        self.edge_prior = MultiScaleEdgePrior()
        self.global_ext = GlobalFeatureExtractor(channels)
        self.local_ext = LocalFeatureExtractor(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.eca = EdgeAwareECA(channels)

    def forward(self, f_e, f_d):
        """Refine encoder feature ``f_e`` (B, C, H, W) with decoder prediction ``f_d`` (B, 1, H, W)."""
        if f_d.shape[-2:] != f_e.shape[-2:]:
            f_d = F.interpolate(f_d, size=f_e.shape[-2:], mode="bilinear", align_corners=False)

        f_edge = self.edge_prior(f_d)                       # Phase I  (B, 1, H, W)
        f_g = self.global_ext(f_e)                          # global contextual saliency
        f_loc = self.local_ext(f_e)                         # local structural detail
        f_m = (f_g + f_loc) * f_edge                        # Eq. 6: edge-modulated response
        f_a = self.fuse(torch.cat([f_e, f_m], dim=1)) + f_e  # Eq. 7: concat + 3x3 conv + residual
        return self.eca(f_a, f_edge)                        # ECA channel recalibration
