"""Evaluation / inference for MambaMPD on the M4D test set.

Computes per-class IoU, mean IoU and pixel accuracy (Table 4 of the paper) and
optionally saves colourised prediction masks.

Example
-------
    python eval.py \
        --dir_dataset "/path/to/M4D Oil Spill Detection Dataset" \
        --file_model_weights mambampd/mambampd_best.pt \
        --dir_save_preds preds/
"""

import argparse
import os

import numpy as np
import torch
from skimage.io import imsave

from models import build_mambampd
from utils.dataset import get_dataloader_for_inference
from utils.metrics import compute_class_IOU, compute_mean_pixel_acc

# M4D label -> RGB colour mapping (sea, oil spill, look-alike, ship, land).
LABEL_TO_COLOR = {
    0: np.array([0, 0, 0]),
    1: np.array([0, 255, 255]),
    2: np.array([255, 0, 0]),
    3: np.array([153, 76, 0]),
    4: np.array([0, 153, 0]),
}
CLASS_NAMES = ["sea_surface", "oil_spill", "look_alike", "ship", "land"]


def _main_output(outputs):
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def colorize(pred_label_arr, num_classes):
    one_hot = np.eye(num_classes)[pred_label_arr]
    mask = np.zeros((pred_label_arr.shape[0], pred_label_arr.shape[1], 3))
    for sem_class in range(num_classes):
        layer = one_hot[:, :, sem_class].reshape(*pred_label_arr.shape, 1)
        mask += layer * LABEL_TO_COLOR[sem_class].reshape(1, 3)
    return mask.astype(np.uint8)


@torch.no_grad()
def evaluate(loader, list_images, model, num_classes, device, dir_masks=None):
    model.eval()
    infer_acc = 0.0
    class_iou = np.array([])

    for i, (image, label) in enumerate(loader):
        image = image.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        pred_label = torch.argmax(torch.softmax(_main_output(model(image)), dim=1), dim=1)

        infer_acc += compute_mean_pixel_acc(label, pred_label)
        sample_iou = compute_class_IOU(label, pred_label, num_classes=num_classes)
        class_iou = sample_iou if class_iou.size == 0 else np.vstack((class_iou, sample_iou))

        if dir_masks is not None:
            arr = np.squeeze(pred_label.detach().cpu().numpy()).astype(np.uint8)
            h, w = arr.shape
            mask = colorize(arr, num_classes)[11 : h - 11, 15 : w - 15]
            imsave(os.path.join(dir_masks, list_images[i].replace(".jpg", ".png")), mask)

    infer_acc /= len(loader)
    per_class_iou = np.nanmean(class_iou, axis=0)
    return infer_acc, per_class_iou


def run(flags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, list_images = get_dataloader_for_inference(flags.dir_dataset)
    print(f"number of test samples: {len(list_images)}")

    model = build_mambampd(
        num_classes=flags.num_classes,
        in_chans=flags.in_chans,
        deep_supervision=bool(flags.deep_supervision),
        use_faa=bool(flags.use_faa),
        use_ega=bool(flags.use_ega),
    )
    model.to(device)
    model.load_state_dict(torch.load(flags.file_model_weights, map_location=device))

    dir_masks = None
    if flags.dir_save_preds:
        dir_masks = os.path.join(flags.dir_save_preds, "masks")
        os.makedirs(dir_masks, exist_ok=True)

    infer_acc, per_class_iou = evaluate(loader, list_images, model, flags.num_classes, device, dir_masks)
    per_class_iou = per_class_iou * 100

    print("\nM4D test-set metrics")
    print(f"pixel accuracy: {infer_acc * 100:.3f} %")
    print(f"mean IoU: {np.mean(per_class_iou):.3f} %")
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        print(f"  {name:<14s}: {iou:.2f} %")


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir_dataset", type=str, required=True)
    parser.add_argument("--file_model_weights", type=str, required=True)
    parser.add_argument("--dir_save_preds", type=str, default="", help="directory to save colourised masks (optional)")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--in_chans", type=int, default=3)
    parser.add_argument("--deep_supervision", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_faa", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_ega", type=int, default=1, choices=[0, 1])
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    run(get_args())
