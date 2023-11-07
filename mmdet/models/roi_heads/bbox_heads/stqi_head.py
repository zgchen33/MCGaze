from mmdet.models.builder import HEADS
from .dii_head import DIIHead
from mmcv.runner import auto_fp16
from mmdet.core import multi_apply
import torch
import torch.nn as nn

@HEADS.register_module()
class STQIHead(DIIHead):
    def __init__(self, *args, **kwargs):
        super(STQIHead, self).__init__(*args, **kwargs)
    
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
        # blink_feat = obj_feat

        for cls_layer in self.cls_fcs:  # fc+layer_norm+relu resnet和vit backbone用的是一样的投影层
            cls_feat = cls_layer(cls_feat) #就是分类和回归再各自做一个小fc投影变换 resnet和vit backbone用的是一样的投影层
        for reg_layer in self.reg_fcs:  #  3* fc+layer_norm+relu
            reg_feat = reg_layer(reg_feat)
        # for blink_layer in self.blink_fcs:
        #     blink_feat = blink_layer(blink_feat)

        #cls_score = obj_feat.new_full((N, num_proposals, 1), 0) # 最后的一个1为对应各自face，eyes，head的标签score
        face_cls_score = self.face_fc_cls(cls_feat[:, 0, :]).view(
            N, 1, 1 if self.loss_cls.use_sigmoid else 2) # [b*t,num_proposals,num_class]
        eyes_cls_score = self.eyes_fc_cls(cls_feat[:, 1, :]).view(
            N, 1, 1 if self.loss_cls.use_sigmoid else 2)
        head_cls_score= self.head_fc_cls(cls_feat[:, 2, :]).view(
            N, 1, 1 if self.loss_cls.use_sigmoid else 2)
        cls_score = torch.cat((face_cls_score, eyes_cls_score, head_cls_score), dim=1)
        #bbox_delta = obj_feat.new_full((N, num_proposals, 4), 0)
        face_bbox_delta= self.face_fc_reg(reg_feat[:, 0, :]).view(N, 1, 4) # [b*t,num_proposals,4]
        eyes_bbox_delta= self.eyes_fc_reg(reg_feat[:, 1, :]).view(N, 1, 4)
        head_bbox_delta= self.head_fc_reg(reg_feat[:, 2, :]).view(N, 1, 4)
        bbox_delta = torch.cat((face_bbox_delta, eyes_bbox_delta, head_bbox_delta),dim=1)
        # blink_score = self.fc_blink(blink_feat).view(N, num_proposals, 1)
        return cls_score, bbox_delta, obj_feat, attn_feats
        # return cls_score, bbox_delta, obj_feat.view(
        #     N, num_proposals, self.in_channels), attn_feats # obj_feat其实就是更新后的tgt,那么还返回中间的atten_feats干甚？因为bbox和mask要共享这个atten_feat特征
        # atten_feats是spatio-temporal self-attention之后的tgt特征，和最终返回的obj_feat相比，它没有通过bbox的dynamic conv作用于roi特征+残差


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
        # pos_blinks_list = [res.pos_blinks for res in sampling_results]
        # neg_blinks_list = [res.neg_blinks for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        # pos_gt_blinks_list = [res.pos_gt_blinks for res in sampling_results]
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