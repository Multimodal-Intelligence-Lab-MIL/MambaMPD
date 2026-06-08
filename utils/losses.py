"""Training objective for MambaMPD.

Following OSDMamba, the paper uses a hybrid objective combining Focal Loss and
Jaccard Loss:

    L = -alpha * (1 - p_t)^gamma * log(p_t) + (1 - IoU)

Deep supervision is applied across decoder stages, so the final objective sums
the hybrid loss over the main prediction and all auxiliary predictions using
predefined layer-wise weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import losses


class HybridLoss(nn.Module):
    """Focal + Jaccard hybrid loss for multi-class segmentation."""

    def __init__(self, mode="multiclass", alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma)
        self.jaccard = losses.JaccardLoss(mode=mode)

    def forward(self, logits, target):
        return self.focal(logits, target) + self.jaccard(logits, target)


class DeepSupervisionLoss(nn.Module):
    """Aggregate the hybrid loss over deep-supervision outputs.

    Each auxiliary prediction is bilinearly upsampled to the ground-truth
    resolution before the loss is computed.  Layer-wise weights decay
    geometrically from the finest (main) prediction and are normalised to sum
    to one.
    """

    def __init__(self, base_loss=None, weights=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else HybridLoss()
        self.weights = weights

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target)

        weights = self.weights[: len(outputs)]
        norm = sum(weights)
        total = 0.0
        for w, out in zip(weights, outputs):
            if out.shape[-2:] != target.shape[-2:]:
                out = F.interpolate(out, size=target.shape[-2:], mode="bilinear", align_corners=False)
            total = total + w * self.base_loss(out, target)
        return total / norm
