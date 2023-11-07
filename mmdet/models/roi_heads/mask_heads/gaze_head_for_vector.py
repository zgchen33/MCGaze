# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
import copy
from mmdet.core import mask_target
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)


@HEADS.register_module()
class GazeHead(BaseModule):
    """Dynamic Mask Head for
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
                 gaze_dim = 3,
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
        self.gaze_dim = gaze_dim
        self.loss_gaze = build_loss(loss_gaze)
        #self.loss_cos = build_loss(loss_cos)
        self.loss_temp = build_loss(loss_temp)
        #self.gaze_fcs = nn.ModuleList()
        self.gaze_face_fcs = nn.ModuleList()
        self.gaze_eyes_fcs = nn.ModuleList()
        self.gaze_head_fcs = nn.ModuleList()
        for _ in range(0,2):
            self.gaze_face_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.gaze_face_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.gaze_face_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))

            self.gaze_eyes_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.gaze_eyes_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.gaze_eyes_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))

            self.gaze_head_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.gaze_head_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.gaze_head_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))
        # 若gaze为vector，则为3维；若为gaze需为yaw，pitch，则为[yaw, pitch, var](3维，对应pinball_loss)
        # 综上，无论gaze_dim=2 or 3，out_channel都是3
        self.fc_face = nn.Linear(in_channels, 3)
        self.fc_eyes = nn.Linear(in_channels, 3)
        self.fc_head = nn.Linear(in_channels, 3)
        self.fc_gaze = nn.Linear(3 * 3, 3)

        self.fusion_mode = 0

        if self.fusion_mode == 1:
            self.atten_fc_gaze = nn.Linear(in_channels * 3, 3)
        
        elif self.fusion_mode == 3 or self.fusion_mode == 4:
            self.fc_face_confidence = nn.Linear(in_channels, gaze_dim)
            self.fc_eyes_confidence = nn.Linear(in_channels, gaze_dim)
            self.fc_head_confidence = nn.Linear(in_channels, gaze_dim)
        elif self.fusion_mode == 2 or self.fusion_mode == 5: 
            self.fc_face_confidence = nn.Linear(in_channels, 1)
            self.fc_eyes_confidence = nn.Linear(in_channels, 1)
            self.fc_head_confidence = nn.Linear(in_channels, 1)


    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)
        bias_init = bias_init_with_prob(0.01)  # 这两行是根据diihead的初始化抄过来的
        nn.init.constant_(self.fc_face.bias, bias_init)
        nn.init.constant_(self.fc_eyes.bias, bias_init)
        nn.init.constant_(self.fc_head.bias, bias_init)
        nn.init.constant_(self.fc_gaze.bias, bias_init)
        if self.fusion_mode == 1:
            nn.init.constant_(self.atten_fc_gaze.bias, bias_init)
        elif self.fusion_mode in [2, 3, 4, 5] :
            nn.init.constant_(self.fc_face_confidence.bias, bias_init)
            nn.init.constant_(self.fc_eyes_confidence.bias, bias_init)
            nn.init.constant_(self.fc_head_confidence.bias, bias_init)
        


    @auto_fp16()
    def forward(self, attn_feats, cls_score):
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
        '''
        for gaze_layer in self.gaze_fcs:
            gaze_feat = gaze_layer(proposal_feat)     # 不做消融实验：gaze_feat = gaze_layer(proposal_feat_iic)
        gaze_score = self.fc_gaze(gaze_feat)
        '''
        
        gaze_face_feat = attn_feats[:, 0, :]
        gaze_eyes_feat = attn_feats[:, 1, :]
        gaze_head_feat = attn_feats[:, 2, :]

        for gaze_layer in self.gaze_face_fcs:
            gaze_face_feat = gaze_layer(gaze_face_feat)
        for gaze_layer in self.gaze_eyes_fcs:
            gaze_eyes_feat = gaze_layer(gaze_eyes_feat)
        for gaze_layer in self.gaze_head_fcs:
            gaze_head_feat = gaze_layer(gaze_head_feat)
            
        #gaze_feat = torch.cat((gaze_face_feat, gaze_head_feat), dim=1)
        face_gaze_score = self.fc_face(gaze_face_feat)
        eyes_gaze_score = self.fc_eyes(gaze_eyes_feat)
        head_gaze_score = self.fc_head(gaze_head_feat)

        ##############################下面为3个分支头得到各自gaze后再乘以各自score融合得到fusion_gaze################################
        if self.fusion_mode == 0:
            face_cls_score = cls_score[:,0,:].expand(face_gaze_score.shape)# [B*t, num_proposal(1), cls_score(1)]
            eyes_cls_score = cls_score[:,1,:].expand(eyes_gaze_score.shape)
            head_cls_score = cls_score[:,2,:].expand(head_gaze_score.shape)
            gaze_feat = torch.cat((face_cls_score * face_gaze_score, eyes_cls_score * eyes_gaze_score, head_cls_score * head_gaze_score), dim=1)
            gaze_score = self.fc_gaze(gaze_feat)
        ##############################上面为3个分支头得到各自gaze后再乘以各自score融合得到fusion_gaze################################

        elif self.fusion_mode == 1:
            # feat * score 
            face_cls_score = cls_score[:,0,:].expand(gaze_face_feat.shape)# [B*t, num_proposal(1), cls_score(1)]
            eyes_cls_score = cls_score[:,1,:].expand(gaze_eyes_feat.shape)
            head_cls_score = cls_score[:,2,:].expand(gaze_head_feat.shape)
            gaze_feat = torch.cat((face_cls_score * gaze_face_feat, eyes_cls_score * gaze_eyes_feat, head_cls_score * gaze_head_feat), dim=1)
            gaze_score = self.atten_fc_gaze(gaze_feat)

        elif self.fusion_mode == 2:
        # # 每个头1个1维可学习confidence
            face_confidence = self.fc_face_confidence(gaze_face_feat).expand(face_gaze_score.shape)
            eyes_confidence = self.fc_eyes_confidence(gaze_eyes_feat).expand(eyes_gaze_score.shape)
            head_confidence = self.fc_face_confidence(gaze_head_feat).expand(head_gaze_score.shape)
            gaze_score = face_gaze_score * face_confidence + eyes_gaze_score * eyes_confidence + head_gaze_score * head_confidence

        elif self.fusion_mode == 3:
        # # 每个头1个3维可学习confidence
            face_confidence = self.fc_face_confidence(gaze_face_feat)
            eyes_confidence = self.fc_eyes_confidence(gaze_eyes_feat)
            head_confidence = self.fc_face_confidence(gaze_head_feat)

            gaze_score = face_gaze_score * face_confidence + eyes_gaze_score * eyes_confidence + head_gaze_score * head_confidence


        elif self.fusion_mode == 4:
            # 每个头1个3维可学习confidence + 利用bbox的score
            face_confidence = self.fc_face_confidence(gaze_face_feat)
            eyes_confidence = self.fc_eyes_confidence(gaze_eyes_feat)
            head_confidence = self.fc_face_confidence(gaze_head_feat)

            face_cls_score = cls_score[:,0,:].expand(face_gaze_score.shape)# [B*t, num_proposal(1), cls_score(1)]
            eyes_cls_score = cls_score[:,1,:].expand(eyes_gaze_score.shape)
            head_cls_score = cls_score[:,2,:].expand(head_gaze_score.shape)
            gaze_feat = torch.cat((face_confidence * face_cls_score * face_gaze_score, eyes_confidence * eyes_cls_score * eyes_gaze_score, head_confidence * head_cls_score * head_gaze_score), dim=1)
            gaze_score = self.fc_gaze(gaze_feat)
            
        elif self.fusion_mode == 5:
            # # 每个头1个1维可学习confidence + 利用bbox的score
            face_confidence = self.fc_face_confidence(gaze_face_feat).expand(face_gaze_score.shape)
            eyes_confidence = self.fc_eyes_confidence(gaze_eyes_feat).expand(eyes_gaze_score.shape)
            head_confidence = self.fc_face_confidence(gaze_head_feat).expand(head_gaze_score.shape)

            face_cls_score = cls_score[:,0,:].expand(face_gaze_score.shape)# [B*t, num_proposal(1), cls_score(1)]
            eyes_cls_score = cls_score[:,1,:].expand(eyes_gaze_score.shape)
            head_cls_score = cls_score[:,2,:].expand(head_gaze_score.shape)
            gaze_feat = torch.cat((face_confidence * face_cls_score * face_gaze_score, eyes_confidence * eyes_cls_score * eyes_gaze_score, head_confidence * head_cls_score * head_gaze_score), dim=1)
            gaze_score = self.fc_gaze(gaze_feat)

        return gaze_score, face_gaze_score, eyes_gaze_score, head_gaze_score

    
    def loss(self, gaze_results, gaze_targets, gaze_weights, reduction_override=None):
        loss = dict()

        face_gaze_targets = torch.cat([torch.stack([t[0]]) for t in gaze_targets])
        eyes_gaze_targets = torch.cat([torch.stack([t[1]]) for t in gaze_targets])
        head_gaze_targets = torch.cat([torch.stack([t[2]]) for t in gaze_targets])

        face_gaze_weights = torch.cat([torch.stack([t[0]]) for t in gaze_weights])
        eyes_gaze_weights = torch.cat([torch.stack([t[1]]) for t in gaze_weights])
        head_gaze_weights = torch.cat([torch.stack([t[2]]) for t in gaze_weights])
        loss['face_gaze'] = {}
        loss['eyes_gaze'] = {}
        loss['head_gaze'] = {}
        loss['final_gaze'] = self.gaze_loss(gaze_results['gaze_score'], head_gaze_targets, head_gaze_weights, usingtemp=True)
        loss['face_gaze'] = self.gaze_loss(gaze_results['face_gaze_score'], face_gaze_targets, face_gaze_weights)
        loss['eyes_gaze'] = self.gaze_loss(gaze_results['eyes_gaze_score'], eyes_gaze_targets, eyes_gaze_weights)
        loss['head_gaze'] = self.gaze_loss(gaze_results['head_gaze_score'], head_gaze_targets, head_gaze_weights)

        return loss


    @force_fp32(apply_to=('gaze_pred', ))
    def gaze_loss(self, gaze_pred, gaze_targets, gaze_weights, usingtemp=False, reduction_override=None):
        #num_pos = torch.tensor(gaze_pred.size()[0],dtype=float).to(gaze_pred.device)

        pos_inds = torch.stack([t[0] != 0 for t in gaze_weights])
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        loss = dict()
        
        #weights = torch.ones(gaze_targets.shape).to(gaze_pred.device)
        gaze_targets = gaze_targets 
        loss_gaze = self.loss_gaze(
            gaze_pred[pos_inds.type(torch.bool)],
            gaze_targets[pos_inds.type(torch.bool)],
            gaze_weights[pos_inds.type(torch.bool)],
            avg_factor=avg_factor)      
        loss['loss_gaze'] = loss_gaze
        # loss_cos不好用，不如loss_arccos
        # loss_cos = self.loss_cos(
        #     gaze_pred[pos_inds.type(torch.bool)],
        #     gaze_targets[pos_inds.type(torch.bool)],
        #     gaze_weights[pos_inds.type(torch.bool)],
        #     avg_factor=avg_factor) 
        # loss['loss_cos'] = loss_cos
        if usingtemp:
            loss_temp = self.loss_temp(
                gaze_pred,
                gaze_targets,
                gaze_weights,
                avg_factor=avg_factor) 
            loss['loss_temp'] = loss_temp
        return loss

    
    def get_targets(self, sampling_results, gt_gazes, rcnn_train_cfg):

        # pos_proposals = [res.pos_bboxes for res in sampling_results]
        #pos_assigned_gt_inds = [
        #    res.pos_assigned_gt_inds for res in sampling_results
        #]
        # print(1)
        # mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
        #                            gt_masks, rcnn_train_cfg)
        ###############gaze_targets = torch.cat(gt_gazes)
        #gaze_targets = torch.cat([gt_gaze[pos_assigned_gt_ind] for (gt_gaze, pos_assigned_gt_ind) in zip(gt_gazes, pos_assigned_gt_inds)])
        
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        
        gaze_targets, gaze_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            gt_gazes,
            cfg=rcnn_train_cfg)
        
        return gaze_targets, gaze_weights
    

    def _get_target_single(self, pos_inds, neg_inds, gt_gazes, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_inds.size(0)
        num_neg = neg_inds.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        gaze_dim = gt_gazes.size(1)
        gaze_targets = gt_gazes.new_zeros(num_samples, gaze_dim)
        gaze_weights = gt_gazes.new_zeros(num_samples, gaze_dim)
        if num_pos > 0:
            if num_pos == 1:
                gaze_targets[pos_inds, :] = gt_gazes[0, :]
                gaze_weights[pos_inds, :] = 1
            else:
                gaze_targets[pos_inds, :] = gt_gazes[pos_inds, :]
                gaze_weights[pos_inds, :] = 1

        return gaze_targets, gaze_weights
