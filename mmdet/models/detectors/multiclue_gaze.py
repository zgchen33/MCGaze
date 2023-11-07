# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .queryinst import QueryInst
import numpy as np
import torch
import torch.nn as nn

@DETECTORS.register_module()
class MultiClueGaze(QueryInst):
    r"""Implementation of
    `Temporally Efﬁcient Vision Transformer for Video Instance Segmentation
    <http://arxiv.org/abs/2105.01928>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MultiClueGaze, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self, B, T, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x) # FPN 返回的特征金字塔得到4层特征
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_gazes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_ids=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN and QueryInst in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. This is required to train QueryInst.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'Sparse R-CNN and MultiClueGaze ' \
            'do not support external proposals'
        B, T, C, H, W = img.shape
        img = img.reshape(B*T, C, H, W)
        img_metas = [_ for clip in img_metas for _ in clip]
        gt_bboxes = [_ for clip in gt_bboxes for _ in clip] # 有些帧有多个instance，那么就会有多个gt box
        gt_labels = [_ for clip in gt_labels for _ in clip] # 同理，一帧也可能有多个
        gt_gazes = [_ for clip in gt_gazes for _ in clip]
        gt_ids = [_ for clip in gt_ids for _ in clip] # id代表什么？和instance有些对应关系好像
        
        x = self.extract_feat(B, T, img) # 返回FPN后的特征
        # proposal_boxes: [B*T, num_proposals=2, 4]; proposal_features: [B*T, num_proposals=2, 256]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas) # 这里的proposal_boxes和proposal_feature本身只有3个，也就是不包含时间维度，函数里先单纯复制，因为反正是初始值，下一个函数开始迭代更新
        roi_losses = self.roi_head.forward_train(
            B,
            T,
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_gazes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            gt_ids=gt_ids,
            imgs_whwh=imgs_whwh)
        return roi_losses
    
    def simple_test(self, img, img_metas, rescale=False, format=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        B, T = 1, img.size(0)
        x = self.extract_feat(B, T, img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas) # 这里的proposal_boxes和proposal_feature本身只有100个，也就是不包含时间维度，函数里先单纯复制，因为反正是初始值，下一个函数开始迭代更新
        results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale,
            format=format)
        return results