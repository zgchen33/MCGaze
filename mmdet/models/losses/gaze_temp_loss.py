# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@LOSSES.register_module()
class GazeTempLoss(nn.Module):
    """Gaze Temp loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0,clip_len =None,reduction='mean', loss_weight=1.0):
        super(GazeTempLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.clip_len = clip_len
        assert self.clip_len != None

    def forward(self,
                pred,
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
        # 下面由target得到gaze_dim
        gaze_dim = target.size(1)


        pred = pred.view(-1, self.clip_len, gaze_dim)  # 现在是 b，t，3
        b_s = pred.shape[0]
        loss = torch.zeros(b_s, self.clip_len).cuda()
        loss[:, 0] = torch.sum(torch.abs(2 * pred[:, 0, :] - 2 * pred[:, 1, :]), dim=-1)
        loss[:, -1] = torch.sum(torch.abs(2 * pred[:, -1, :] - 2 * pred[:, -2, :]), dim=-1)
        loss[:,1:-1]= torch.sum(torch.abs(2 * pred[:, 1:-1, :] - pred[:, 2:, :] - pred[:, 0:-2, :]),dim=-1)
        loss = loss.view(-1)


        return self.loss_weight*loss.mean()