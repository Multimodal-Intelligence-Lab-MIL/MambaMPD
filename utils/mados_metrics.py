"""Pixel-level segmentation metrics for MADOS (sparse annotations).

Adapted from the official MADOS framework. ``Evaluation`` returns macro / micro /
weighted Precision, Recall and F1 together with macro IoU; ``confusion_matrix``
produces a labelled confusion matrix with per-class IoU / Precision / Recall / F1
and overall accuracy. Metrics are computed only on annotated pixels.
"""

import numpy as np
import pandas as pd
import sklearn.metrics as metr
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


def Evaluation(y_predicted, y_true):
    info = {
        "macroPrec": precision_score(y_true, y_predicted, average="macro"),
        "microPrec": precision_score(y_true, y_predicted, average="micro"),
        "weightPrec": precision_score(y_true, y_predicted, average="weighted"),
        "macroRec": recall_score(y_true, y_predicted, average="macro"),
        "microRec": recall_score(y_true, y_predicted, average="micro"),
        "weightRec": recall_score(y_true, y_predicted, average="weighted"),
        "macroF1": f1_score(y_true, y_predicted, average="macro"),
        "microF1": f1_score(y_true, y_predicted, average="micro"),
        "weightF1": f1_score(y_true, y_predicted, average="weighted"),
        "subsetAcc": accuracy_score(y_true, y_predicted),
        "IoU": jaccard_score(y_true, y_predicted, average="macro"),
    }
    return info


def confusion_matrix(y_gt, y_pred, labels, percentage=False):
    cm = metr.confusion_matrix(y_gt, y_pred)
    f1_macro = metr.f1_score(y_gt, y_pred, average="macro")
    mRec = metr.recall_score(y_gt, y_pred, average="macro")
    OA = metr.accuracy_score(y_gt, y_pred)
    UA = metr.precision_score(y_gt, y_pred, average=None)
    Rec = metr.recall_score(y_gt, y_pred, average=None)
    f1 = metr.f1_score(y_gt, y_pred, average=None)
    IoC = metr.jaccard_score(y_gt, y_pred, average=None)
    mIoC = metr.jaccard_score(y_gt, y_pred, average="macro")

    sz1, sz2 = cm.shape
    cm_with_stats = np.zeros((sz1 + 4, sz2 + 2))
    cm_with_stats[0:-4, 0:-2] = cm
    cm_with_stats[-3, 0:-2] = np.round(100 * IoC, 1)
    cm_with_stats[-2, 0:-2] = np.round(100 * UA, 1)
    cm_with_stats[-1, 0:-2] = np.round(100 * f1, 1)
    cm_with_stats[0:-4, -1] = np.round(100 * Rec, 1)
    cm_with_stats[-4, 0:-2] = np.sum(cm, axis=0)
    cm_with_stats[0:-4, -2] = np.sum(cm, axis=1)

    cm_list = cm_with_stats.tolist()

    first_row = list(labels) + ["Sum", "Recall"]
    first_col = list(labels) + ["Sum", "IoU", "Precision", "F1-score"]

    for idx, sublist in enumerate(cm_list):
        if idx == sz1:
            sublist[-2], sublist[-1] = "mRec:", round(100 * mRec, 1)
        elif idx == sz1 + 1:
            sublist[-2], sublist[-1] = "mIoU:", round(100 * mIoC, 1)
        elif idx == sz1 + 2:
            sublist[-2], sublist[-1] = "OA:", round(100 * OA, 1)
        elif idx == sz1 + 3:
            sublist[-2], sublist[-1] = "F1-macro:", round(100 * f1_macro, 1)
        cm_list[idx] = sublist

    conf_array = np.array(cm_list)
    if percentage:
        temp = conf_array[:-4, :].astype(float)
        conf_array[:-4, :-2] = (100 * temp[:, :-2] / temp[:, -2].reshape(-1, 1)).round(1).astype(str)

    df = pd.DataFrame(conf_array)
    df.columns = first_row
    df.index = first_col
    return df
