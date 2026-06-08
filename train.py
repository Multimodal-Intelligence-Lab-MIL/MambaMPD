"""Training entry point for MambaMPD on the M4D SAR oil-spill dataset.

Reproduces the training protocol described in Section 4.2 of the paper:
SGD (momentum 0.9, weight decay 1e-4), initial learning rate 0.001 with a
"Poly" decay schedule, batch size 8, 100 epochs, a hybrid Focal + Jaccard loss,
deep supervision, and an ImageNet-pretrained VMamba-Tiny encoder.

Example
-------
    python train.py \
        --dir_dataset "/path/to/M4D Oil Spill Detection Dataset" \
        --pretrained_ckpt pretrained/vssmtiny_dp01_ckpt_epoch_292.pth \
        --batch_size 8 --num_epochs 100 --learning_rate 0.001
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler

from models import build_mambampd, load_pretrained_ckpt
from utils.dataset import get_dataloaders_for_training
from utils.losses import DeepSupervisionLoss
from utils.logger import CSVWriter, write_dict_to_json
from utils.metrics import compute_mean_IOU, compute_mean_pixel_acc


class PolynomialLR(LRScheduler):
    """Polynomial ("Poly") learning-rate decay."""

    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power, self.min_lr)
            for base_lr in self.base_lrs
        ]


def _main_output(outputs):
    """Return the full-resolution prediction from a (possibly deep-supervision) output."""
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


@torch.no_grad()
def validation_loop(loader, model, criterion, device, num_classes):
    model.eval()
    num_batches = len(loader)
    valid_loss, valid_acc, valid_iou = 0.0, 0.0, 0.0
    for image, label in loader:
        image = image.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        outputs = model(image)
        valid_loss += criterion(outputs, label).item()

        pred_label = torch.argmax(torch.softmax(_main_output(outputs), dim=1), dim=1)
        valid_acc += compute_mean_pixel_acc(label, pred_label)
        valid_iou += compute_mean_IOU(label, pred_label, num_classes=num_classes)

    return valid_loss / num_batches, valid_acc / num_batches, valid_iou / num_batches


def train_loop(loader, model, criterion, optimizer, device):
    model.train()
    num_batches = len(loader)
    train_loss = 0.0
    for image, label in loader:
        image = image.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / num_batches


def batch_train(flags):
    dir_path = os.path.join(flags.dir_model, flags.which_model)
    os.makedirs(dir_path, exist_ok=True)
    csv_writer = CSVWriter(
        file_name=os.path.join(dir_path, "train_metrics.csv"),
        column_names=["epoch", "train_loss", "valid_loss", "valid_acc", "valid_IOU"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = get_dataloaders_for_training(
        flags.dir_dataset, flags.batch_size, random_state=flags.random_state
    )

    model = build_mambampd(
        num_classes=flags.num_classes,
        in_chans=flags.in_chans,
        deep_supervision=bool(flags.deep_supervision),
        use_faa=bool(flags.use_faa),
        use_ega=bool(flags.use_ega),
    )
    if flags.pretrained_ckpt and os.path.isfile(flags.pretrained_ckpt):
        model = load_pretrained_ckpt(model, flags.pretrained_ckpt)
    else:
        print("No pretrained checkpoint loaded (training encoder from scratch).")
    model.to(device)

    criterion = DeepSupervisionLoss()

    if flags.which_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=flags.learning_rate, momentum=0.9, weight_decay=flags.weight_decay
        )
        lr_scheduler = PolynomialLR(optimizer, flags.num_epochs + 1, power=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=flags.learning_rate, weight_decay=flags.weight_decay)
        lr_scheduler = None

    print(f"\ntraining MambaMPD ({flags.which_model}) on {device}\n")
    write_dict_to_json(os.path.join(dir_path, "params.json"), vars(flags))

    best_iou = 0.0
    for epoch in range(1, flags.num_epochs + 1):
        t0 = time.time()
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_acc, valid_iou = validation_loop(
            valid_loader, model, criterion, device, flags.num_classes
        )
        print("-" * 100)
        print(f"Epoch {epoch}/{flags.num_epochs}  time {time.time() - t0:.1f}s  train_loss {train_loss:.5f}")
        print(f"  valid_loss {valid_loss:.5f}  valid_acc {valid_acc:.5f}  valid_IOU {valid_iou:.5f}")

        csv_writer.write_row(
            [epoch, round(train_loss, 5), round(valid_loss, 5), round(valid_acc, 5), round(valid_iou, 5)]
        )
        torch.save(model.state_dict(), os.path.join(dir_path, f"mambampd_{flags.which_model}_{epoch}.pt"))
        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(model.state_dict(), os.path.join(dir_path, "mambampd_best.pt"))
        if lr_scheduler is not None:
            lr_scheduler.step()

    print(f"Training complete. Best validation mIoU: {best_iou:.5f}")
    csv_writer.close()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir_dataset", type=str, required=True, help="path to the M4D dataset root")
    parser.add_argument("--dir_model", type=str, default=os.getcwd(), help="directory to save checkpoints")
    parser.add_argument("--which_model", type=str, default="mambampd", help="run/checkpoint name")
    parser.add_argument("--pretrained_ckpt", type=str, default="pretrained/vssmtiny_dp01_ckpt_epoch_292.pth",
                        help="ImageNet-pretrained VMamba-Tiny checkpoint")
    parser.add_argument("--which_optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--in_chans", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=3)
    parser.add_argument("--deep_supervision", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_faa", type=int, default=1, choices=[0, 1], help="enable Frequency-Aware Augmentation")
    parser.add_argument("--use_ega", type=int, default=1, choices=[0, 1], help="enable Edge-Guided Attention")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    batch_train(get_args())
