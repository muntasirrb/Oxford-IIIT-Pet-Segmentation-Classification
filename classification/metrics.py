import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def seg_metrics(seg_logits, masks, num_classes=2):
    preds   = seg_logits.argmax(dim=1)
    iou_list, dice_list = [], []

    for cls in range(num_classes):
        pred_cls   = (preds == cls)
        target_cls = (masks == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union        = (pred_cls | target_cls).sum().item()
        denom        = pred_cls.sum().item() + target_cls.sum().item()

        if union  > 0: iou_list .append(intersection / union)
        if denom  > 0: dice_list.append((2 * intersection) / denom)

    pixel_acc = (preds == masks).float().mean().item()
    mean_iou  = sum(iou_list)  / len(iou_list)  if iou_list  else 0.0
    mean_dice = sum(dice_list) / len(dice_list) if dice_list else 0.0

    return {"pixel_acc": pixel_acc, "mean_iou": mean_iou, "dice": mean_dice}


def cls_metrics(cls_logits, breeds):
    with torch.no_grad():
        preds      = cls_logits.argmax(dim=1).cpu().numpy()
        targets_np = breeds.cpu().numpy()
        batch_size = breeds.size(0)

        _, pred_topk = cls_logits.topk(min(5, cls_logits.size(1)),
                                       dim=1, largest=True, sorted=True)
        top1 = pred_topk[:, :1].eq(breeds.view(-1, 1)).any(dim=1).float().mean().item()
        top5 = pred_topk[:, :5].eq(breeds.view(-1, 1)).any(dim=1).float().mean().item()

        precision = precision_score(targets_np, preds, average="macro", zero_division=0)
        recall    = recall_score   (targets_np, preds, average="macro", zero_division=0)
        f1        = f1_score       (targets_np, preds, average="macro", zero_division=0)

        return {
            "top1"     : top1,
            "top5"     : top5,
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
            "preds"    : preds,
            "targets"  : targets_np,
        }