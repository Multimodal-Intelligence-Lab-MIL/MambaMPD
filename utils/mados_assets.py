"""MADOS class labels, colour mapping and small training helpers (TTA, schedules)."""

import argparse

import numpy as np
import torch
from torchvision.transforms.functional import hflip

mados_cat_mapping = {
    "Marine Debris": 1, "Dense Sargassum": 2, "Sparse Floating Algae": 3,
    "Natural Organic Material": 4, "Ship": 5, "Oil Spill": 6, "Marine Water": 7,
    "Sediment-Laden Water": 8, "Foam": 9, "Turbid Water": 10, "Shallow Water": 11,
    "Waves & Wakes": 12, "Oil Platform": 13, "Jellyfish": 14, "Sea snot": 15,
}

mados_color_mapping = {
    "Marine Debris": "red", "Dense Sargassum": "green", "Sparse Floating Algae": "limegreen",
    "Natural Organic Material": "brown", "Ship": "orange", "Oil Spill": "thistle",
    "Marine Water": "navy", "Sediment-Laden Water": "gold", "Foam": "purple",
    "Turbid Water": "darkkhaki", "Shallow Water": "darkturquoise", "Waves & Wakes": "bisque",
    "Oil Platform": "dimgrey", "Jellyfish": "hotpink", "Sea snot": "yellow",
}

labels = list(mados_cat_mapping.keys())


def bool_flag(s):
    """Parse a boolean command-line argument."""
    if s.lower() in {"off", "false", "0"}:
        return False
    if s.lower() in {"on", "true", "1"}:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def TTA(img, reverse_aggregation=False):
    """Test-time augmentation: 4 rotations x horizontal flip (8 views)."""
    im_list = []
    if not reverse_aggregation:
        for k in [0, 1, 2, 3]:
            im = torch.rot90(img, k=k, dims=[-2, -1])
            im_list.append(im)
            im_list.append(hflip(im))
        return torch.cat(im_list)

    for k in [3, 2, 1, 0]:
        im = hflip(img[k * 2 + 1, :, :])
        im_list.append(torch.rot90(im, k=-k, dims=[-2, -1]))
        im_list.append(torch.rot90(img[k * 2, :, :], k=-k, dims=[-2, -1]))
    img = torch.stack(im_list)
    return torch.mode(img, dim=0, keepdim=True)[0]
