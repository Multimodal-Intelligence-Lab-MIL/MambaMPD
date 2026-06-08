"""Frequency-Aware Augmentation (FAA) module.

Implements Section 3.2 / Fig. 3 of the paper.  FAA is placed *before* the patch
embedding and operates on the raw input ``X`` so that high-frequency detail is
injected before any spatial downsampling.  It uses a dual-branch design:

* a **residual branch** (a depthwise convolution), and
* a **wavelet branch** that performs a hierarchical (J-level) Haar wavelet
  decomposition, refines every sub-band with a shared small convolution, and
  reconstructs the feature coarse-to-fine via the inverse wavelet transform.

The DWT/IWT use fixed Haar filters implemented as stride-2 (transposed)
convolutions; their weights are frozen and receive no gradient updates, so FAA
adds a negligible number of trainable parameters (Table 10).  The reconstructed
feature is recalibrated by a Squeeze-and-Excitation block and finally passed
through a 7x7 depthwise convolution followed by instance normalization.
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# --------------------------------------------------------------------------- #
# Wavelet primitives (fixed Haar filters, frozen).
# --------------------------------------------------------------------------- #
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


def wavelet_transform_init(filters):
    class _WaveletTransform(Function):
        @staticmethod
        def forward(ctx, inp):
            with torch.no_grad():
                return wavelet_transform(inp, filters)

        @staticmethod
        def backward(ctx, grad_output):
            return inverse_wavelet_transform(grad_output, filters), None

    return _WaveletTransform().apply


def inverse_wavelet_transform_init(filters):
    class _InverseWaveletTransform(Function):
        @staticmethod
        def forward(ctx, inp):
            with torch.no_grad():
                return inverse_wavelet_transform(inp, filters)

        @staticmethod
        def backward(ctx, grad_output):
            return wavelet_transform(grad_output, filters), None

    return _InverseWaveletTransform().apply


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WTConv2d(nn.Module):
    """Hierarchical wavelet convolution (dual-branch: depthwise residual + wavelet).

    ``wt_levels`` controls the decomposition depth J; only the low-frequency
    approximation is decomposed further at each level (paper Eq. 3).  ``wt_type``
    ``'db1'`` is the Haar basis.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, wt_levels=2, wt_type="db1"):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d expects in_channels == out_channels"

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        # Residual branch: a depthwise convolution applied to the input.
        self.base_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding="same", stride=1, dilation=1, groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # Wavelet branch: shared refinement conv phi(.) per level over the 4 sub-bands.
        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4, in_channels * 4, kernel_size, padding="same", stride=1, dilation=1,
                    groups=in_channels * 4, bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x)) + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


class FAA(nn.Module):
    """Frequency-Aware Augmentation module (paper Section 3.2).

    Args:
        channels: number of input/output channels (operates on the raw input,
            so this equals the number of image channels).
        wt_levels: wavelet decomposition depth J (paper's best is J = 2).
        wt_type: wavelet basis; ``'db1'`` is the Haar wavelet.
    """

    def __init__(self, channels=3, wt_levels=2, wt_type="db1", se_reduction=16):
        super().__init__()
        self.wt = WTConv2d(channels, channels, kernel_size=3, wt_levels=wt_levels, wt_type=wt_type)
        self.se = SELayer(channels, reduction=se_reduction)
        self.refine = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.InstanceNorm2d(channels, eps=1e-5, affine=True)

    def forward(self, x):
        identity = x
        # Dual-branch wavelet + residual fusion.
        z = self.wt(x)
        # Channel recalibration to emphasize informative frequency responses.
        z = self.se(z)
        # Local-global interaction + normalization, with an overall residual.
        z = self.norm(self.refine(z))
        return identity + z
