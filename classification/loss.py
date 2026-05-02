import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)
        targets_oh  = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class JointLoss(nn.Module):
    """
    Combined loss for joint training:
        Total = seg_weight * (CE + Dice) + cls_weight * CrossEntropy
    """
    def __init__(self, seg_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ce_seg     = nn.CrossEntropyLoss()
        self.dice       = DiceLoss()
        self.ce_cls     = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def forward(self, seg_logits, cls_logits, masks, breeds):
        cls_loss = self.ce_cls(cls_logits, breeds)

        if self.seg_weight == 0.0 or seg_logits is None:
            return cls_loss, torch.tensor(0.0), cls_loss

        seg_loss = self.ce_seg(seg_logits, masks) + self.dice(seg_logits, masks)
        total    = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return total, seg_loss, cls_loss