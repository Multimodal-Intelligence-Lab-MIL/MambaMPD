"""Evaluate MambaMPD on the MADOS test (or val) split.

Loads one or more trained checkpoints (a folder of ``*.pth`` is treated as an
ensemble via majority vote), runs inference over the chosen split, and reports
the pixel-level Evaluation metrics and a labelled confusion matrix on annotated
pixels. Optionally applies 8-view test-time augmentation (with ``--batch 1``).

Example
-------
    python eval_mados.py --path ./data/MADOS \
        --model_path trained_models_mados/model_best.pth --split test
"""

import argparse
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_mambampd
from utils.mados_assets import TTA, bool_flag, labels
from utils.mados_dataset import MADOS
from utils.mados_metrics import Evaluation, confusion_matrix


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _main_output(outputs):
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def load_models(model_path, in_chans, num_classes, device):
    model_files = glob(os.path.join(model_path, "*.pth")) if os.path.isdir(model_path) else [model_path]
    models_list = []
    for model_file in model_files:
        model = build_mambampd(num_classes=num_classes, in_chans=in_chans, deep_supervision=True)
        model.to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        models_list.append(model)
    return models_list


def main(options):
    seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = options["img_size"]

    splits_path = os.path.join(options["path"], "splits")
    dataset_test = MADOS(options["path"], splits_path, options["split"])
    test_loader = DataLoader(dataset_test, batch_size=options["batch"], shuffle=False)

    models_list = load_models(options["model_path"], options["input_channels"], options["output_channels"], device)

    y_true, y_predicted = [], []
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="testing"):
            use_tta = options["test_time_augmentations"] and options["batch"] == 1
            if use_tta:
                image = TTA(image)

            image = image.to(device)
            target = target.to(device)
            model_in = F.interpolate(image, size=(img_size, img_size), mode="bilinear", align_corners=False)

            all_predictions = []
            for model in models_list:
                logits = F.interpolate(_main_output(model(model_in)), size=target.shape[-2:], mode="bilinear", align_corners=False)
                predictions = torch.softmax(logits, dim=1).argmax(1)
                if use_tta:
                    predictions = TTA(predictions, reverse_aggregation=True)
                all_predictions.append(predictions)

            predictions = torch.mode(torch.cat(all_predictions), dim=0, keepdim=True)[0]

            predictions = predictions.reshape(-1)
            target = target.reshape(-1)
            mask = target != -1
            y_predicted += predictions[mask].cpu().numpy().tolist()
            y_true += target[mask].cpu().numpy().tolist()

    acc = Evaluation(y_predicted, y_true)
    print("Evaluation:", acc)
    print("Confusion Matrix:\n" + confusion_matrix(y_true, y_predicted, labels, options["results_percentage"]).to_string())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to the MADOS dataset root")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--model_path", required=True, help="a .pth checkpoint or a folder of checkpoints (ensemble)")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--input_channels", default=11, type=int)
    parser.add_argument("--output_channels", default=15, type=int)
    parser.add_argument("--test_time_augmentations", default=True, type=bool_flag, help="8-view TTA (use --batch 1)")
    parser.add_argument("--results_percentage", default=True, type=bool_flag)
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(get_args())
