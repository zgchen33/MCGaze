import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmcv.runner import auto_fp16
from mmdet.core import multi_apply
from .bbox_head import BBoxHead
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer

@HEADS.register_module()
class GazeSTQIHead(BBoxHead):
    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(GazeSTQIHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), in_channels)[1]
        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
            
        if self.loss_cls.use_sigmoid:# 这里各自分类头用各自的fc去分类
            self.face_fc_cls = nn.Linear(in_channels, 1)
            self.eyes_fc_cls = nn.Linear(in_channels, 1)
            self.head_fc_cls = nn.Linear(in_channels, 1)
        else:
            self.face_fc_cls = nn.Linear(in_channels, 1 + 1)
            self.eyes_fc_cls = nn.Linear(in_channels, 1 + 1)
            self.head_fc_cls = nn.Linear(in_channels, 1 + 1)
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.face_fc_reg = nn.Linear(in_channels, 4)     # 这个其实继承了父类里定义过，不过这里又重新定义一遍，值会更新的
        self.eyes_fc_reg = nn.Linear(in_channels, 4)
        self.head_fc_reg = nn.Linear(in_channels, 4)

        # TODO czg important: reg_class_agnostic???
        assert self.reg_class_agnostic, 'DIIHead only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'DIIHead only ' \
            'suppport `reg_decoded_bbox=True`'
    
    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(GazeSTQIHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            #nn.init.constant_(self.fc_cls.bias, bias_init)
            nn.init.constant_(self.face_fc_cls.bias, bias_init)
            nn.init.constant_(self.eyes_fc_cls.bias, bias_init)
            nn.init.constant_(self.head_fc_cls.bias, bias_init)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, clip_length):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals, d = proposal_feat.shape
        # proposal_feat是上一个stage更新后的proposal_feat,(是query-level的，类似于detr里的tgt），roi_feat是本阶段得到的roi特征
        # Self attention 先做各自帧内的spatial atten, 然后再做每个proposal内across t帧的atten,实现优雅
        proposal_feat = proposal_feat.permute(1, 0, 2) # [b*t,num_proposals,256] --> [num_proposals,b*t,256] 目的：self-atten batch那维放在中间
       
        ######### 把以下注释掉就是去掉了spatial-self-atten
        proposal_feat = self.attention_norm(self.attention(proposal_feat)) # self-atten batch那一维是b*t，所以就是每个t内部做spatial的self-atten,挺巧妙的，要是一般的atten,则t维度和特征维度混在一起应该
        ######### 把以上注释掉就是去掉了spatial-self-atten

        proposal_feat = proposal_feat.permute(1, 0, 2) # [num_proposals,b*t,256] --> [b*t,num_proposals,256]变回来
        #########把以下注释掉就是去掉了temporal-self-atten
        proposal_feat = proposal_feat.resize(N // clip_length, clip_length,
                                             num_proposals,
                                             d).permute(1, 0, 2, 3) # [b*t,num_proposals,256] --> [t,b,num_proposals,256]
        proposal_feat = proposal_feat.resize(clip_length,
                                             N * num_proposals // clip_length,
                                             d) # [t,b,num_proposals,256] --> [t,b*num_proposals,256],这是让每个proposal自己，在t维度内做self-atten，太妙了
        proposal_feat = self.attention_norm(self.attention(proposal_feat)) # 让每个proposal自己，在t维度内做self-atten
        proposal_feat = proposal_feat.resize(clip_length, N // clip_length,
                                             num_proposals,
                                             d).permute(1, 0, 2, 3) # [t,b*num_proposals,256] --> [b,t,num_proposals,256]
        proposal_feat = proposal_feat.resize(N, num_proposals, d) # [b,t,num_proposals,256] --> [b*t,num_proposals,256] 又回到了最初的shape
        ############把以上注释掉就是去掉了temporal-self-atten
        attn_feats = proposal_feat

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels) # [b*t,num_proposals,256] --> [b*t*num_proposals,256]
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat) # dynamic conv,应该是用proposal_feat作为滤波器，去滤波当前roi_feat
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic) # 就是和残差连接，和detr decoder很像
        obj_feat = self.instance_interactive_conv_norm(proposal_feat) # 一个layer norm

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))
        obj_feat = obj_feat.view(N, num_proposals, self.in_channels)

        cls_feat = obj_feat # [b*t*num_proposal, 256]
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:  # fc+layer_norm+relu resnet和vit backbone用的是一样的投影层
            cls_feat = cls_layer(cls_feat) #就是分类和回归再各自做一个小fc投影变换 resnet和vit backbone用的是一样的投影层
        for reg_layer in self.reg_fcs:  #  3* fc+layer_norm+relu
            reg_feat = reg_layer(reg_feat)

        #cls_score = obj_feat.new_full((N, num_proposals, 1), 0) # 最后的一个1为对应各自face，eyes，head的标签score
        face_cls_score = self.face_fc_cls(cls_feat[:, 0, :]).view(
            N, 1, 1 if self.loss_cls.use_sigmoid else 2) # [b*t,num_proposals,num_class]
        eyes_cls_score = self.eyes_fc_cls(cls_feat[:, 1, :]).view(
            N, 1, 1 if self.loss_cls.use_sigmoid else 2)
        head_cls_score= self.head_fc_cls(cls_feat[:, 2, :]).view(
            N, 1, 1 if self.loss_cls.use_sigmoid else 2)
        cls_score = torch.cat((face_cls_score, eyes_cls_score, head_cls_score), dim=1)
        face_bbox_delta= self.face_fc_reg(reg_feat[:, 0, :]).view(N, 1, 4) # [b*t,num_proposals,4]
        eyes_bbox_delta= self.eyes_fc_reg(reg_feat[:, 1, :]).view(N, 1, 4)
        head_bbox_delta= self.head_fc_reg(reg_feat[:, 2, :]).view(N, 1, 4)
        bbox_delta = torch.cat((face_bbox_delta, eyes_bbox_delta, head_bbox_delta),dim=1)
        return cls_score, bbox_delta, obj_feat, attn_feats
        # return cls_score, bbox_delta, obj_feat.view(
        #     N, num_proposals, self.in_channels), attn_feats # obj_feat其实就是更新后的tgt,那么还返回中间的atten_feats干甚？因为bbox和mask要共享这个atten_feat特征
        # atten_feats是spatio-temporal self-attention之后的tgt特征，和最终返回的obj_feat相比，它没有通过bbox的dynamic conv作用于roi特征+残差

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        """"Loss function of DIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        num_region = labels[0].shape[0]

        labels = torch.stack(labels, dim=0)
        labels = [labels[:, i] for i in range(num_region)]

        label_weights = torch.stack(label_weights, dim=0)
        label_weights = [label_weights[:, i] for i in range(num_region)]

        bbox_targets = torch.stack(bbox_targets,dim=0)
        bbox_targets = [bbox_targets[:, i, :] for i in range(num_region)]

        bbox_weights = torch.stack(bbox_weights,dim=0)
        bbox_weights = [bbox_weights[:, i, :] for i in range(num_region)]

        bbox_pred = torch.stack(bbox_pred,dim=0)
        bbox_pred = [bbox_pred[:, i, :] for i in range(num_region)]

        cls_score = [cls_score[:, i, :] for i in range(num_region)]
        
        imgs_whwh = [imgs_whwh[:, i, :] for i in range(num_region)]

        losses['face'] = self.head_loss(cls_score[0], bbox_pred[0], labels[0], label_weights[0], bbox_targets[0], bbox_weights[0], imgs_whwh[0])
        losses['eyes'] = self.head_loss(cls_score[1], bbox_pred[1], labels[1], label_weights[1], bbox_targets[1], bbox_weights[1], imgs_whwh[1])
        losses['head'] = self.head_loss(cls_score[2], bbox_pred[2], labels[2], label_weights[2], bbox_targets[2], bbox_weights[2], imgs_whwh[2])
        return losses
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def head_loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None):
        
        losses = dict()
        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        #temp_labels = labels.new_full(labels.shape, 0)

        neg_index = torch.nonzero((labels < 0) | (labels >= bg_class_ind))
        pos_index = torch.nonzero((labels >= 0) & (labels < bg_class_ind))
        labels[neg_index] = 1
        labels[pos_index] = 0

        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override) # 用的sigmoid focal loss，对于匹配到none-object的样本，希望模型预测人脸的概率越小越小，这部分loss也计算了。详情参考notion.
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)] # 就是输入图像的尺寸
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
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
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     -1,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            # pos_gt_labels为3种，但如果num_pos == 1时，只有head_bboxes一种而已
            if num_pos == 1:
                labels[pos_inds] = pos_gt_labels[-1]
            else:
                labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        # label: 得到了100个proposal各自的label,对于匹配到gt的,结果就是gt,对于一类来说就是0,none-object对应的label为0+1=1
        # label_weights: 100个proposal都是1
        # bbox_target: 100个proposal中匹配到gt的就是gt的bbox,匹配到none-object的就是[0,0,0,0]
        # bbox_weight: 100个proposal中匹配到gt的就是1,匹配到none-object的就是0
        # 总得来说，匹配到none-object的也要算分类loss,标签是原始分类(max_label+1)，代表none-object类，而bbox不算none-object的loss，只算匹配到gt的
        # 这个是每一帧都有，帧级算loss,比如一个query匹配到一个instance,这个instance在某些帧被遮挡而没出现，那么这个query在这些帧里的label也是none-object,这样算loss就非常合理了
        # 这个函数之前的操作是把输入转成list,每个元素是一帧的结果，调用多次self._get_target_single,每次的输入是一帧的，type是tensor
        # 其实这个函数的目的在于给计算loss的权重，对于分类，所有的都给weight=1,对于bbox,只有匹配到的正样本的loss weight=1,其余匹配到none-object的负样本的loss weight=0
        if concat:
            labels = torch.cat(labels, 0) # 把[b*t,num_proposal]展开成b*t*num_proposal
            label_weights = torch.cat(label_weights, 0) # 同理展开
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights