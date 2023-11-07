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
class BlinkHead(BaseModule):
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
        loss_blink (dict): The config for mask loss.
    """

    def __init__(self,
                 in_channels=256,
                 loss_blink=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                 **kwargs):
        init_cfg = None
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(BlinkHead, self).__init__(init_cfg)

        self.fp16_enabled = False
        self.in_channels = in_channels
        self.loss_blink = build_loss(loss_blink)
        self.blink_fcs = nn.ModuleList()
        for _ in range(0,2):
            self.blink_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.blink_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.blink_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))
        self.fc_blink = nn.Linear(in_channels, 1)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)
        bias_init = bias_init_with_prob(0.01)  # 这两行是根据diihead的初始化抄过来的
        nn.init.constant_(self.fc_blink.bias, bias_init)

    @auto_fp16()
    def forward(self, proposal_feat):
        """Forward function of BlinkHead.

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
        for blink_layer in self.blink_fcs:
            blink_feat = blink_layer(proposal_feat)     # 不做消融实验：blink_feat = blink_layer(proposal_feat_iic)
        blink_score = self.fc_blink(blink_feat)
        return blink_score

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

    @force_fp32(apply_to=('blink_pred', ))
    def loss(self, blink_pred, blink_targets, reduction_override=None):
        num_pos = torch.tensor(blink_pred.size()[0],dtype=float).to(blink_pred.device)
        avg_factor = reduce_mean(num_pos)
        loss = dict()
        # print(f'{sum(blink_targets)} blinks, {len(blink_targets)} in totoal')
        # if sum(blink_targets) > 0 :
        #     print(f'{sum(blink_targets)} blinks, {len(blink_targets)} in totoal')
        blink_targets = 1 - blink_targets # 我们输入的是1为眨眼，0为非眨眼，我们的目标是希望眨眼时输出概率大，所以算focal loss时要把0作为眨眼，1作为非眨眼（none-object类)
        loss_blink = self.loss_blink(
            blink_pred,
            blink_targets,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
        loss['loss_blink'] = loss_blink
        return loss

    def get_targets(self, sampling_results, gt_blinks, rcnn_train_cfg):

        # pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        # print(1)
        # mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
        #                            gt_masks, rcnn_train_cfg)
        blink_targets = torch.cat([gt_blink[pos_assigned_gt_ind] for (gt_blink, pos_assigned_gt_ind) in zip(gt_blinks, pos_assigned_gt_inds)])
        return blink_targets
