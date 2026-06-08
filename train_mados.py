"""Train MambaMPD on the MADOS multispectral marine-pollution dataset.

This integrates the official MADOS / MariNeXt training framework (sparse-label
dataset, VSCP augmentation, weighted cross-entropy, EMA, multi-step schedule,
pixel-level evaluation) but replaces the MariNeXt backbone with the MambaMPD
model. Following the paper, the patch-embed / stem accepts 11 Sentinel-2 bands
(``Cin = 11``) while the rest of the VSS encoder keeps its ImageNet-pretrained
weights; deep supervision is applied across decoder stages.

MADOS patches are 240x240. MambaMPD downsamples by 32x, so each input is resized
to ``--img_size`` (256, divisible by 32) before the model and the logits are
upsampled back to the annotation resolution for the loss and metrics.

Example
-------
    python train_mados.py --path ./data/MADOS \
        --pretrained_ckpt pretrained/vssmtiny_dp01_ckpt_epoch_292.pth \
        --batch 4 --epochs 100
"""

import argparse
import ast
import json
import logging
import os
import random
from os.path import dirname as up
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_mambampd, load_pretrained_ckpt
from utils.losses import DeepSupervisionLoss
from utils.mados_assets import bool_flag, cosine_scheduler
from utils.mados_dataset import MADOS, class_distr, gen_weights, vscp
from utils.mados_metrics import Evaluation

root_path = os.path.abspath(up(__file__))


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _main_output(outputs):
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def main(options):
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    os.makedirs(os.path.join(root_path, "logs"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(root_path, "logs", "log_mambampd_mados_" + str(time()) + ".log"),
        filemode="a", level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s",
    )
    writer = SummaryWriter(os.path.join(root_path, "logs", options["tensorboard"] + "_" + str(time())))

    splits_path = os.path.join(options["path"], "splits")
    dataset_train = MADOS(options["path"], splits_path, "train")
    dataset_val = MADOS(options["path"], splits_path, "val")

    train_loader = DataLoader(
        dataset_train, batch_size=options["batch"], shuffle=True, num_workers=options["num_workers"],
        pin_memory=options["pin_memory"], worker_init_fn=seed_worker, generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        dataset_val, batch_size=options["batch"], shuffle=False, num_workers=options["num_workers"],
        pin_memory=options["pin_memory"], worker_init_fn=seed_worker, generator=g,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = options["img_size"]

    model = build_mambampd(
        num_classes=options["output_channels"], in_chans=options["input_channels"], deep_supervision=True
    )
    if options["pretrained_ckpt"] and os.path.isfile(options["pretrained_ckpt"]):
        model = load_pretrained_ckpt(model, options["pretrained_ckpt"])
    model.to(device)

    model_ema = None
    if options["model_ema"]:
        from timm.utils import ModelEma

        model_ema = ModelEma(model, decay=options["model_ema_decay"], device=device, resume="")
        ema_decay_schedule = cosine_scheduler(
            options["model_ema_decay"], 0.999, options["epochs"], len(train_loader)
        )

    # Weighted cross-entropy over annotated pixels, aggregated across deep-supervision stages.
    weight = gen_weights(class_distr, c=options["weight_param"]).to(device)
    base_loss = nn.CrossEntropyLoss(
        ignore_index=-1, reduction="mean", weight=weight, label_smoothing=options["label_smoothing"]
    )
    criterion = DeepSupervisionLoss(base_loss=base_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["decay"])
    if options["reduce_lr_on_plateau"] == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, options["lr_steps"], gamma=0.1)

    start, epochs, eval_every = options["resume_from_epoch"] + 1, options["epochs"], options["eval_every"]
    best_iou = 0.0
    model.train()

    for epoch in range(start, epochs + 1):
        training_loss, training_batches, i_board = [], 0, 0
        for it, (image, target) in enumerate(tqdm(train_loader, desc="training")):
            it = len(train_loader) * (epoch - 1) + it

            if options["vscp"]:
                image_aug, target_aug = vscp(image.cpu().numpy(), target.cpu().numpy())
                image = torch.cat([image, torch.tensor(image_aug)])
                target = torch.cat([target, torch.tensor(target_aug)])

            image = image.to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            model_in = F.interpolate(image, size=(img_size, img_size), mode="bilinear", align_corners=False)
            outputs = model(model_in)
            loss = criterion(outputs, target)
            loss.backward()

            if options["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_grad"])
            optimizer.step()

            if model_ema is not None:
                model_ema.decay = ema_decay_schedule[it]
                model_ema.update(model)

            training_batches += target.shape[0]
            training_loss.append((loss.data * target.shape[0]).tolist())
            writer.add_scalar("training loss", loss, (epoch - 1) * len(train_loader) + i_board)
            i_board += 1

        logging.info("Training loss: " + str(sum(training_loss) / training_batches))

        model_dir = os.path.join(options["checkpoint_path"], str(epoch))
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

        if epoch % eval_every == 0 or epoch == 1:
            val_loss, acc_val = validate(model, val_loader, criterion, device, img_size, options["output_channels"])
            logging.info(f"Epoch {epoch}  val_loss {val_loss:.5f}  Evaluation {acc_val}")
            writer.add_scalar("F1/val macroF1", acc_val["macroF1"], epoch)
            writer.add_scalar("IoU/val MacroIoU", acc_val["IoU"], epoch)
            print(f"[epoch {epoch}] val_loss {val_loss:.5f}  macroF1 {acc_val['macroF1']*100:.2f}  mIoU {acc_val['IoU']*100:.2f}")

            if acc_val["IoU"] > best_iou:
                best_iou = acc_val["IoU"]
                torch.save(model.state_dict(), os.path.join(options["checkpoint_path"], "model_best.pth"))

            if options["reduce_lr_on_plateau"] == 1:
                scheduler.step(val_loss)
            else:
                scheduler.step()
            model.train()

            if model_ema is not None and options["model_ema_eval"]:
                _, acc_ema = validate(model_ema.ema, val_loader, criterion, device, img_size, options["output_channels"])
                torch.save(model_ema.ema.state_dict(), os.path.join(model_dir, "model_ema.pth"))
                writer.add_scalar("F1/val macroF1 (EMA)", acc_ema["macroF1"], epoch)
                writer.add_scalar("IoU/val MacroIoU (EMA)", acc_ema["IoU"], epoch)

    print(f"Training complete. Best validation mIoU: {best_iou*100:.2f}")


@torch.no_grad()
def validate(model, val_loader, criterion, device, img_size, num_classes):
    model.eval()
    seed_all(0)
    val_loss, val_batches = [], 0
    y_true, y_pred = [], []
    for image, target in tqdm(val_loader, desc="validating"):
        image = image.to(device)
        target = target.to(device)

        model_in = F.interpolate(image, size=(img_size, img_size), mode="bilinear", align_corners=False)
        outputs = model(model_in)
        val_loss.append((criterion(outputs, target).data * target.shape[0]).tolist())

        logits = F.interpolate(_main_output(outputs), size=target.shape[-2:], mode="bilinear", align_corners=False)
        logits = logits.movedim(1, 3).reshape((-1, num_classes))
        target_flat = target.reshape(-1)
        mask = target_flat != -1
        probs = torch.softmax(logits[mask], dim=1).cpu().numpy()

        val_batches += int(mask.sum().item())
        y_pred += probs.argmax(1).tolist()
        y_true += target_flat[mask].cpu().numpy().tolist()

    acc_val = Evaluation(np.asarray(y_pred), np.asarray(y_true))
    return (sum(val_loss) / max(val_batches, 1)), acc_val


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to the MADOS dataset root")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--img_size", default=256, type=int, help="model input size (divisible by 32)")
    parser.add_argument("--resume_from_epoch", default=0, type=int)
    parser.add_argument("--input_channels", default=11, type=int)
    parser.add_argument("--output_channels", default=15, type=int)
    parser.add_argument("--weight_param", default=1.03, type=float)
    parser.add_argument("--pretrained_ckpt", default="pretrained/vssmtiny_dp01_ckpt_epoch_292.pth", type=str)

    parser.add_argument("--vscp", type=bool_flag, default=True)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--reduce_lr_on_plateau", default=0, type=int)
    parser.add_argument("--lr_steps", default="[45,65]", type=str)

    parser.add_argument("--checkpoint_path", default=os.path.join(root_path, "trained_models_mados"))
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--model_ema", type=bool_flag, default=True)
    parser.add_argument("--model_ema_decay", type=float, default=0.999)
    parser.add_argument("--model_ema_eval", type=bool_flag, default=True)

    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_memory", default=False, type=bool_flag)
    parser.add_argument("--tensorboard", default="tsboard_mados", type=str)

    options = vars(parser.parse_args())
    lr_steps = ast.literal_eval(options["lr_steps"])
    options["lr_steps"] = lr_steps if isinstance(lr_steps, list) else [lr_steps]
    return options


if __name__ == "__main__":
    opts = get_args()
    os.makedirs(opts["checkpoint_path"], exist_ok=True)
    print("parsed input parameters:")
    print(json.dumps(opts, indent=2))
    main(opts)
