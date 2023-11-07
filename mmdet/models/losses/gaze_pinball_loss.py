# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weighted_loss



@LOSSES.register_module()
class GazePinballLoss(nn.Module):
    """Gaze pinball loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(GazePinballLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.q1 = 0.1
        self.q9 = 1-self.q1


    def forward(self,
                pred,
                var,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        q_10 = target - (pred - var)
        q_90 = target - (pred + var)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)


        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return self.loss_weight * (loss_10 + loss_90)

