# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)


@HEADS.register_module()
class GazeHead(BaseModule):
    r"""Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_convs (int): Number of convolution layer.
            Defaults to 4.
        roi_feat_size (int): The output size of RoI extractor,
            Defaults to 14.
        in_channels (int): Input feature channels.
            Defaults to 256.
        conv_kernel_size (int): Kernel size of convolution layers.
            Defaults to 3.
        conv_out_channels (int): Output channels of convolution layers.
            Defaults to 256.
        num_classes (int): Number of classes.
            Defaults to 80
        class_agnostic (int): Whether generate class agnostic prediction.
            Defaults to False.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        upsample_cfg (dict): The config for upsample layer.
        conv_cfg (dict): The convolution layer config.
        norm_cfg (dict): The norm layer config.
        dynamic_conv_cfg (dict): The dynamic convolution layer config.
        loss_gaze (dict): The config for mask loss.
    """

    def __init__(self,
                 in_channels=256,
                 loss_gaze=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                 loss_cos = None,
                 loss_temp = None,
                 **kwargs):
        init_cfg = None
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(GazeHead, self).__init__(init_cfg)

        self.fp16_enabled = False
        self.in_channels = in_channels
        self.loss_gaze = build_loss(loss_gaze)
        self.loss_cos = build_loss(loss_cos)
        self.loss_temp = build_loss(loss_temp)
        self.gaze_fcs = nn.ModuleList()
        for _ in range(0,2):
            self.gaze_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.gaze_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.gaze_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))
        self.fc_gaze = nn.Linear(in_channels, 3)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)
        bias_init = bias_init_with_prob(0.01)  # 这两行是根据diihead的初始化抄过来的
        nn.init.constant_(self.fc_gaze.bias, bias_init)

    @auto_fp16()
    def forward(self, proposal_feat):
        """Forward function of gazeHead.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            mask_pred (Tensor): Predicted foreground masks with shape
                (batch_size*num_proposals, num_classes,
                                        pooling_h*2, pooling_w*2).
        """

        # # roi_feat是roi_align来的特征，proposal_feat是atten_feat,也就是query

        # proposal_feat = proposal_feat.reshape(-1, self.in_channels)  # 好像没变化 注释于2023.3.10

        # proposal_feat_iic = self.instance_interactive_conv(
        #     proposal_feat, roi_feat)        # 注意这里可没有残差连接
        # # 消融实验，可以直接把roi_feat flatten（不不，直接加入一个avg pool最简单），作为后续fc的输入
        # proposal_feat = proposal_feat + proposal_feat_iic
        # proposal_feat = self.instance_interactive_conv_norm(proposal_feat)
        for gaze_layer in self.gaze_fcs:# (B*t, d=256)
            gaze_feat = gaze_layer(proposal_feat)     # 不做消融实验：gaze_feat = gaze_layer(proposal_feat_iic)
        gaze_score = self.fc_gaze(gaze_feat) #(B*t, 3)
        return gaze_score

        # x = proposal_feat_iic.permute(0, 2, 1).reshape(roi_feat.size()) # [b*t里的正样本数,w*h,256] --> [b*t里的正样本数,256,w,h]
        #
        # for conv in self.convs:
        #     x = conv(x)
        # if self.upsample is not None:
        #     x = self.upsample(x)
        #     if self.upsample_method == 'deconv':
        #         x = self.relu(x)
        # mask_pred = self.conv_logits(x) # x本身没有预测类别，这里根据一个1*1卷积将通道数变为类别数
        # return mask_pred

    @force_fp32(apply_to=('gaze_pred', ))
    def loss(self, gaze_pred, gaze_targets, reduction_override=None):
        num_pos = torch.tensor(gaze_pred.size()[0],dtype=float).to(gaze_pred.device)
        avg_factor = reduce_mean(num_pos)
        loss = dict()
        
        weights = torch.ones(gaze_targets.shape).to(gaze_pred.device)
        gaze_targets = gaze_targets 
        loss_gaze = self.loss_gaze(
            gaze_pred,
            gaze_targets,
            weights,
            avg_factor=avg_factor)      
        loss['loss_gaze'] = loss_gaze

        loss_cos = self.loss_cos(
            gaze_pred,
            gaze_targets,
            weights,
            avg_factor=avg_factor) 
        loss['loss_cos'] = loss_cos

        loss_temp = self.loss_temp(
            gaze_pred,
            gaze_targets,
            weights,
            avg_factor=avg_factor) 
        loss['loss_temp'] = loss_temp
        return loss

    def get_targets(self, sampling_results, gt_gazes, rcnn_train_cfg):

        # pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        # print(1)
        # mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
        #                            gt_masks, rcnn_train_cfg)
        gaze_targets = torch.cat([gt_gaze[pos_assigned_gt_ind] for (gt_gaze, pos_assigned_gt_ind) in zip(gt_gazes, pos_assigned_gt_inds)])
        return gaze_targets
