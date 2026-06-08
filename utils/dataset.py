"""M4D SAR oil-spill dataset and dataloaders.

The M4D dataset (Krestenitis et al., 2019) provides 1002 training and 110 test
SAR images with 5 semantic classes: sea surface, oil spill, look-alike, ship and
land.  We hold out 5% of the training set for validation, following the paper.

Note: the paper additionally evaluates on the MADOS multispectral (11-band)
benchmark.  MADOS support is not included here (see the project README "Datasets"
section for how it would be added: an 11-channel patch-embed and per-band
normalisation).
"""

import os

import torch
import torchvision.transforms as transforms
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .image_preprocessing import ImagePadder
from .logger import load_dict_from_json


class M4DSAROilSpillDataset(Dataset):
    def __init__(self, dir_data, list_images, which_set="train", file_stats_json="image_stats.json"):
        self.dir_data = dir_data
        self.which_set = which_set
        self.file_stats_json = file_stats_json
        try:
            self.dict_stats = load_dict_from_json(self.file_stats_json)
        except OSError:
            dir_json = os.path.dirname(os.path.realpath(__file__))
            self.dict_stats = load_dict_from_json(os.path.join(dir_json, self.file_stats_json))

        self._dir_images = os.path.join(self.dir_data, "images")
        self._dir_labels = os.path.join(self.dir_data, "labels_1D")

        self._list_images = sorted(list_images)
        self._list_labels = [f.replace(".jpg", ".png") for f in self._list_images]

        dir_pad_image = os.path.dirname(self._dir_images)
        self._image_padder = ImagePadder(
            os.path.join("/".join(os.path.normpath(dir_pad_image).split(os.sep)[:-1]), "train", "images")
        )

        self._image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[self.dict_stats["mean"]] * 3,
                    std=[self.dict_stats["std"]] * 3,
                ),
            ]
        )

        self._affine_transform = None
        if self.which_set == "train":
            self._affine_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
            )

    def __len__(self):
        return len(self._list_images)

    def __getitem__(self, idx):
        file_image = os.path.join(self._dir_images, self._list_images[idx])
        file_label = os.path.join(self._dir_labels, self._list_labels[idx])

        image = imread(file_image)
        label = imread(file_label)

        image = self._image_padder.pad_image(image)
        label = self._image_padder.pad_label(label)

        if self.which_set == "train":
            image_tensor = torch.from_numpy(image)
            label_tensor = torch.unsqueeze(torch.from_numpy(label), dim=-1)
            stacked = torch.cat([image_tensor, label_tensor], dim=-1)
            stacked = torch.permute(stacked, [2, 0, 1])
            stacked = self._affine_transform(stacked)
            stacked = torch.permute(stacked, [1, 2, 0]).numpy()
            image = stacked[:, :, :-1]
            label = stacked[:, :, -1]

        image = self._image_transform(image)
        return image, label


def get_dataloaders_for_training(dir_dataset, batch_size, random_state=None, num_workers=4, val_size=0.05):
    list_images = sorted(
        f for f in os.listdir(os.path.join(dir_dataset, "train", "images")) if f.endswith(".jpg")
    )
    list_train_images, list_valid_images = train_test_split(
        list_images, test_size=val_size, shuffle=True, random_state=random_state
    )
    print("dataset information")
    print(f"number of train samples: {len(list_train_images)}")
    print(f"number of validation samples: {len(list_valid_images)}")

    train_dataset = M4DSAROilSpillDataset(os.path.join(dir_dataset, "train"), list_train_images, which_set="train")
    valid_dataset = M4DSAROilSpillDataset(os.path.join(dir_dataset, "train"), list_valid_images, which_set="valid")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


def get_dataloader_for_inference(dir_dataset, batch_size=1, num_workers=4):
    list_inference_images = sorted(
        f for f in os.listdir(os.path.join(dir_dataset, "test", "images")) if f.endswith(".jpg")
    )
    inference_dataset = M4DSAROilSpillDataset(
        os.path.join(dir_dataset, "test"), list_inference_images, which_set="test"
    )
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return inference_loader, list_inference_images
