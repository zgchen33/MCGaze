import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList
from .sparse_roi_head import SparseRoIHead
from ..builder import HEADS, build_head, build_roi_extractor
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh

@HEADS.register_module()
class MultiClueGazeROIHead(SparseRoIHead):

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=None,
                 mask_roi_extractor=None,
                 bbox_head=None,
                 mask_head=None,
                 gaze_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MultiClueGazeROIHead, self).__init__(num_stages,
            stage_loss_weights,
            proposal_feature_channel,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        if gaze_head is not None:
            self.init_gaze_head(mask_roi_extractor, gaze_head)

    @property
    def with_gaze(self):
        """bool: whether the RoI head contains a `gaze_head`"""
        return hasattr(self, 'gaze_head') and self.gaze_head is not None
    
    def init_gaze_head(self, mask_roi_extractor, gaze_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            gaze_head (dict): Config of mask in mask head.
        """
        self.gaze_head = nn.ModuleList()
        if not isinstance(gaze_head, list):
            gaze_head = [gaze_head for _ in range(self.num_stages)]
        assert len(gaze_head) == self.num_stages
        for head in gaze_head:
            self.gaze_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas, clip_length):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage] # 每个stage的roi_align_extractor,内部有4个roi_extractor,对应FPN输出的四层特征
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois) # [b*t*num_proposal,256,所提取roi特征的w,所提取roi特征的h] 这一步做的应该是roi align,输入的rois是上一阶段得到的绝对bbox值吧
        # cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
        #     bbox_feats, object_feats, clip_length)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats, clip_length) # 输入参数object_feat是The object feature extracted from the previous stage (tgt)
        # atten_feats是spatio-temporal self-attention之后的tgt特征，和最终返回的obj_feat相比，它没有通过bbox的dynamic conv作用于roi特征+残差
        # 上面返回的bbox_pred好像是一个delta
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas) # rois应该是上一个stage的qeury的bbox预测 [x1,y1,x2,y2]，bbox_pred是本阶段tgt预测的bbox,是个delta
        # 上面函数根据本阶段预测的delta得到更新的bbox t*[x1,y1,x2,y2]
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=proposal_list,
            object_feats=object_feats,
            attn_feats=attn_feats,
            detach_cls_score_list=[cls_score[i].detach() for i in range(num_imgs)],
            detach_proposal_list=[item.detach() for item in proposal_list]
        ) # 为何要detac? 可能是分配标签的过程不想有梯度

        return bbox_results
    
    def _gaze_forward(self, stage, attn_feats, cls_score):
        """Mask head forward function used in both training and testing."""
        gaze_head = self.gaze_head[stage]
        # do not support caffe_c4 model anymore
        gaze_score, face_gaze_score, eyes_gaze_score, head_gaze_score = gaze_head(attn_feats, cls_score)
        gaze_results = dict()
        # 若gaze为vector，所需要设置的标准化为 '模为1' 的操作，在gaze_head内已经完成了
        gaze_results['gaze_score'] = gaze_score
        gaze_results['face_gaze_score'] = face_gaze_score
        gaze_results['eyes_gaze_score'] = eyes_gaze_score
        gaze_results['head_gaze_score'] = head_gaze_score
        return gaze_results

    def _gaze_forward_train(self, stage, attn_feats, cls_score, sampling_results,
                            gt_gazes, rcnn_train_cfg):
        """Run forward function and calculate loss for mask head in
        training."""
        '''
        attn_feats = torch.cat([  # 对每个匹配到的目标的256维特征计算gaze
            feats[res.pos_inds]
            for (feats, res) in zip(attn_feats, sampling_results)
        ]) # (B*t,)
        '''
        gaze_results = self._gaze_forward(stage, attn_feats, cls_score)

        gaze_targets = self.gaze_head[stage].get_targets(
            sampling_results, gt_gazes, rcnn_train_cfg)

        # pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results]) # 这个是对的，本身就是和prediction对应好的，不需要根据pos_assigned_gt_inds来排序了

        loss_gaze = self.gaze_head[stage].loss(gaze_results,
                                               *gaze_targets)
        gaze_results.update(loss_gaze)
        return gaze_results

    def forward_train(self,
                      B,
                      T,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_gazes,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None,
                      gt_ids=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """
        num_imgs = len(img_metas)  # b*t 张图片
        # TODO:czg 将proposal_boxes和features等处理为分别的1个face和1个head
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1) # (b*t,1,4) --> (b*t,num_proposals(3),4)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]  # b*t,4
        object_feats = proposal_features  # list (b*t)个 每个元素是一个tensor（num_proposals(100)，4）
        all_stage_loss = {}

        for stage in range(self.num_stages):    # 开始迭代
            rois = bbox2roi(proposal_list)  # 形成tensor(b*t*num_proposals(3),5)
            bbox_results = self._bbox_forward(
                stage, x, rois, object_feats, img_metas, clip_length=T) # 这个函数和测试阶段调用的是同一函数
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']  # 每个输入图像的num_proposals个预测的分类值(B*t, num_proposals)
            proposal_list = bbox_results['detach_proposal_list']  # 每个输入图像的num_proposals个预测的bbox值(B*t, num_proposals, 4)
            for i in range(B):
                normolize_bbox_ccwh = []
                for j in range(T):  # 遍历一个输入clip的每一帧
                    normolize_bbox_ccwh.append(
                        bbox_xyxy_to_cxcywh(proposal_list[i * T + j] /
                                            imgs_whwh[i * T])) # proposal_list中的每一个元素绝对坐标的，这里和原图相除应该是得到0-1归一化值，这个i*t+j就是取出对应batch,对应图片对应帧的100个proposal出来
                # 在这个clip里面每个proposal与每个gt的对应关系
                assign_result = self.bbox_assigner[stage].assign(
                    normolize_bbox_ccwh,            # 传入的bbox_pred只是提供一些shape信息,然后计算匹配cost,这个函数只是求分配的对应id,并没有做好每个pred的target gt
                    cls_pred_list[i * T:i * T + T],     # 因此，该函数无需输入gaze
                    gt_bboxes[i * T:i * T + T],
                    gt_labels[i * T:i * T + T],
                    img_metas[i * T],
                    gt_ids=gt_ids[i * T:i * T + T]) # 这里要进行匈牙利匹配了，确实是输出的t帧和gt的t帧进行instance-level匹配,不过assign_result给出了帧级的匹配，就是如果这一帧少一个gt,那result这一帧匹配到none object
                sampling_result = []
                for j in range(T):  # 遍历每一帧
                    #print("B:{}; T:{}".format(i, j))
                    sampling_result.append(self.bbox_sampler[stage].sample(
                        assign_result[j], proposal_list[i * T + j],gt_bboxes[i * T + j]
                        ))  # 这个采样就是划分出了正负样本，其实没采样，比如100个proposal里有3个匹配到了gt,那么就返回三个正样本，97个负样本
                sampling_results.extend(sampling_result)    # 上面的sample函数会给出正样本的bbox的gt
            
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                False) # bbox_targets是一个4维的tuple, 分别是labels,label_weights,bbox_targets,bbox_weights，作为后续算loss传入的4个参数 关于label的维度就是[b*t*num_proposals],关于bbox的就是[b*t*num_proposals,4]
            cls_score = bbox_results['cls_score']   # [b*t,num_proposal,num_class]
            decode_bbox_pred = bbox_results['decode_bbox_pred'] # [b*t*num_proposal, 4]

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score, #cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred,
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            if self.with_gaze:
                gaze_results = self._gaze_forward_train(
                    stage, bbox_results['object_feats'], torch.stack(cls_pred_list), sampling_results,
                    gt_gazes, self.train_cfg[stage])
                single_stage_loss['final_gaze'] = gaze_results['final_gaze']
                single_stage_loss['face_gaze'] = gaze_results['face_gaze']
                single_stage_loss['eyes_gaze'] = gaze_results['eyes_gaze']
                single_stage_loss['head_gaze'] = gaze_results['head_gaze']

            for key, value in single_stage_loss.items():
                for key1, value1 in value.items():
                    all_stage_loss[f'stage{stage}_{key}_{key1}'] = value1 * \
                                    self.stage_loss_weights[stage]
            object_feats = bbox_results['object_feats']

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False,
                    format=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has a mask branch,
            it is a list[tuple] that contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]    # [t,num_proposal,4]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        object_feats = proposal_features # [t,num_proposal,256]
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs
            return bbox_results

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas, clip_length=len(img_metas)) # 和train调用的同一函数
            # 根据本阶段预测的delta得到更新的bbox [t,4] 4为[x1,y1,x2,y2]
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []
        attn_feats = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]
        # 测试阶段，只使用了最后一次的迭代结果，丢弃了前面的迭代结果
        cls_score_mean = cls_score.mean(dim=0) # 各帧的分类结果取平均，作为整体query的分类结果，某帧遮挡了怎么办？这是只是根据平局值top10选取query,后续实际上还是把其各帧的预测结果拿出来了

        for img_id in range(num_imgs):
            labels_per_img = [0, 1, 2] 
            bbox_pred_per_img = proposal_list[img_id]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor) # bbox直接除以scale_factor,格式是[x1,y1,x2,y2]
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, cls_score[img_id, :, :]], dim=1)) # scores_per_img[:, None]这句把[10] -->[10,1],然后和bbox_pred_per_img在dim=1维度进行concat,整体变为[10,5]
            det_labels.append(labels_per_img)
        attn_feats = bbox_results['object_feats']  

        if format:
            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i], num_classes)
                for i in range(num_imgs)
            ]
        else:
            bbox_results = (det_bboxes, det_labels)

        if self.with_gaze:
            gaze_results = self._gaze_forward(stage, attn_feats, cls_score)
            return bbox_results, gaze_results
        else:
            segm_results = []
            for i in range(0, len(bbox_results[0])):
                segm_results.append(torch.zeros([10,1]))
            return bbox_results, segm_results
