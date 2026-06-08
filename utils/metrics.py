"""Segmentation metrics: mean pixel accuracy, mean IoU and per-class IoU."""

import numpy as np


def compute_mean_pixel_acc(true_label, pred_label):
    """Mean pixel accuracy over a batch of (B, H, W) label tensors."""
    if true_label.shape != pred_label.shape:
        print("shape mismatch:", true_label.shape, pred_label.shape)
        return
    if true_label.dim() != 3:
        print("true_label must be 3-D, got dim", true_label.dim())
        return

    acc_sum = 0
    for i in range(true_label.shape[0]):
        true_arr = true_label[i].clone().detach().cpu().numpy().astype(np.int32)
        pred_arr = pred_label[i].clone().detach().cpu().numpy().astype(np.int32)
        same = (true_arr == pred_arr).sum()
        a, b = true_arr.shape
        acc_sum += same / (a * b)
    return acc_sum / true_label.shape[0]


def compute_mean_IOU(true_label, pred_label, num_classes=5):
    """Mean IoU over classes present in the ground truth."""
    present_iou_list = []
    pred_label = pred_label.view(-1)
    true_label = true_label.view(-1)
    for sem_class in range(num_classes):
        pred_inds = pred_label == sem_class
        target_inds = true_label == sem_class
        if target_inds.long().sum().item() == 0:
            continue
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        present_iou_list.append(float(intersection) / float(union))
    return np.mean(np.array(present_iou_list))


def compute_class_IOU(true_label, pred_label, num_classes=5):
    """Per-class IoU (NaN where the class is absent from the ground truth)."""
    pred_label = pred_label.view(-1)
    true_label = true_label.view(-1)
    per_class_iou = np.zeros(num_classes)
    for sem_class in range(num_classes):
        pred_inds = pred_label == sem_class
        target_inds = true_label == sem_class
        if target_inds.long().sum().item() == 0:
            per_class_iou[sem_class] = float("nan")
            continue
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        per_class_iou[sem_class] = float(intersection) / float(union)
    return per_class_iou
