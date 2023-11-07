# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weighted_loss



@LOSSES.register_module()
class GazeArccosLoss(nn.Module):
    """Gaze Arccos loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(GazeArccosLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

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
        gaze_dim = target.size(1)
        if gaze_dim != 3:
            pred = self.yaw_pitch_to_vector(pred)
            target = self.yaw_pitch_to_vector(target)
        sim = F.cosine_similarity(pred, target, dim=-1, eps=1e-6)
        sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)

        # dot_product = torch.sum(pred * target, dim=-1)
        # norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=-1))
        # norm_target = torch.sqrt(torch.sum(target ** 2, dim=-1))
        # cos_sim = dot_product / (norm_pred * norm_target)

        loss_angle = torch.acos(sim)
        
        return self.loss_weight*loss_angle.mean()


    def yaw_pitch_to_vector(self, x):
        x = torch.reshape(x, (-1, 2))
        output = torch.zeros((x.shape[0], 3))
        output[:,2] = - torch.cos(x[:,1]) * torch.cos(x[:,0])
        output[:,0] = torch.cos(x[:,1]) * torch.sin(x[:,0])
        output[:,1] = torch.sin(x[:,1])
        return output

    def vector_to_yaw_pitch(self, x):
        x = torch.reshape(x, (-1, 3))
        x = x / torch.norm(x, dim=1).reshape(-1, 1)
        output = torch.zeros((x.shape[0], 2))
        output[:,0] = torch.atan2(x[:,0], - x[:,2])
        output[:,1] = torch.asin(x[:,1])
        return output