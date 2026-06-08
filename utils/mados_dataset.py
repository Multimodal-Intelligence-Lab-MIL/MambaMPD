"""MADOS (Marine Debris and Oil Spill) dataset pipeline.

Adapted from the official MADOS / MariNeXt framework (Kikaki et al., 2024) so it
can be trained with the MambaMPD model. MADOS provides 11-band Sentinel-2
patches (bands resampled to 10 m by nearest-neighbour) with 15 thematic classes
and *sparse* annotations: unannotated pixels carry the label ``-1`` and are
ignored by the loss and the metrics.

Requires ``rasterio`` and GDAL (``osgeo``) to read the multispectral ``.tif``
patches.
"""

import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

# Pixel-level class distribution (sums to 1.0), used to weight the loss.
class_distr = torch.Tensor(
    [0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 0.34775, 0.20638,
     0.00062, 0.1169, 0.09188, 0.01309, 0.00917, 0.00176, 0.00963]
)

# Per-band normalisation statistics (11 Sentinel-2 bands).
bands_mean = np.array(
    [0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
     0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944], dtype="float32"
)
bands_std = np.array(
    [0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
     0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], dtype="float32"
)


def get_band(path):
    return int(path.split("_")[-2])


class MADOS(Dataset):
    """MADOS semantic-segmentation dataset (11-band, 15 classes, sparse labels)."""

    def __init__(self, path, splits, mode="train"):
        if mode == "train":
            self.ROIs_split = np.genfromtxt(os.path.join(splits, "train_X.txt"), dtype="str")
        elif mode == "test":
            self.ROIs_split = np.genfromtxt(os.path.join(splits, "test_X.txt"), dtype="str")
        elif mode == "val":
            self.ROIs_split = np.genfromtxt(os.path.join(splits, "val_X.txt"), dtype="str")
        else:
            raise ValueError(f"unknown mode: {mode}")

        import rasterio
        from rasterio.enums import Resampling
        from osgeo import gdal
        from tqdm import tqdm

        self.X = []
        self.y = []
        self.tiles = glob(os.path.join(path, "*"))

        for tile in tqdm(self.tiles, desc="Load " + mode + " set to memory"):
            crop_splits = [f.split("_cl_")[-1] for f in glob(os.path.join(tile, "10", "*_cl_*"))]
            for crop in crop_splits:
                crop_name = os.path.basename(tile) + "_" + crop.split(".tif")[0]
                if crop_name not in self.ROIs_split:
                    continue

                all_bands = sorted(glob(os.path.join(tile, "*", "*L2R_rhorc*_" + crop)), key=get_band)
                current_image = []
                for band in all_bands:
                    upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10
                    with rasterio.open(band, mode="r") as src:
                        current_image.append(
                            src.read(
                                1,
                                out_shape=(int(src.height * upscale_factor), int(src.width * upscale_factor)),
                                resampling=Resampling.nearest,
                            ).copy()
                        )
                self.X.append(np.stack(current_image))

                cl_path = os.path.join(tile, "10", os.path.basename(tile) + "_L2R_cl_" + crop)
                ds = gdal.Open(cl_path)
                self.y.append(np.copy(ds.ReadAsArray().astype(np.int64)))
                ds = None

        self.X = np.stack(self.X)
        self.y = np.stack(self.y) - 1  # categories from 1..15 -> 0..14 (unannotated -> -1)

        self.impute_nan = np.tile(bands_mean, (self.X.shape[-1], self.X.shape[-2], 1))
        self.mode = mode
        self.length = len(self.y)

    def __len__(self):
        return self.length

    def getnames(self):
        return self.ROIs_split

    def __getitem__(self, index):
        image = self.X[index]
        target = self.y[index]

        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype("float32")  # CxWxH -> WxHxC
        nan_mask = np.isnan(image)
        image[nan_mask] = self.impute_nan[nan_mask]

        target = target[:, :, np.newaxis]
        if self.mode == "train":
            image, target = self.join_transform(image, target)

        image = (image.astype(np.float32).transpose(2, 0, 1).copy() - bands_mean.reshape(-1, 1, 1)) / bands_std.reshape(-1, 1, 1)
        target = target.squeeze()
        return image.copy(), target.copy()

    def join_transform(self, image, target):
        f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]
        if f != 2:
            image = self.flip_array(image, f)
            target = self.flip_array(target, f)
        if np.random.random() < 0.8:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, (1, 0))
            target = np.rot90(target, k, (1, 0))
        return image, target

    @staticmethod
    def flip_array(array, flip_code):
        if flip_code != -1:
            return np.flip(array, flip_code)
        return np.fliplr(np.flipud(array))


def gen_weights(class_distribution, c=1.02):
    """Inverse-log class weights for the (weighted) cross-entropy loss."""
    return 1 / torch.log(c + class_distribution)


def vscp(image, target):
    """Very Simple Copy-Paste augmentation (overlays annotated pixels)."""
    n_augmented = image.shape[0] // 2
    image_temp = image[: n_augmented * 2].copy()
    target_temp = target[: n_augmented * 2].copy()

    image_augmented, target_augmented = [], []
    for i in range(n_augmented):
        image_temp[i, :, target_temp[i + n_augmented] != -1] = image_temp[i + n_augmented, :, target_temp[i + n_augmented] != -1]
        image_augmented.append(image_temp[i].copy())
        target_temp[i, target_temp[i + n_augmented] != -1] = target_temp[i + n_augmented, target_temp[i + n_augmented] != -1]
        target_augmented.append(target_temp[i].copy())

    return np.stack(image_augmented), np.stack(target_augmented)
