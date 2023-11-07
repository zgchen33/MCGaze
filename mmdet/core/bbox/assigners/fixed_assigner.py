import torch

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class TeViTHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self, **kwargs):
        pass

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               gt_ids=None,
               eps=1e-7):
        """Computes one-to-one fixed matching.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        clip_length = len(bbox_pred)
        total_gt_ids = torch.unique(torch.cat(gt_ids))

        num_gts = total_gt_ids.numel()

        num_bboxes = bbox_pred[0].size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred[0].new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)
        assigned_labels = bbox_pred[0].new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
        
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return [
                AssignResult(
                    num_gts, assigned_gt_inds, None, labels=assigned_labels)
                for _ in range(clip_length)
            ]

        assign_results = []
        for _ in range(clip_length):
            num_gt_bboxes = gt_bboxes[_].size(0)
            assigned_gt_inds = bbox_pred[0].new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)
            assigned_labels = bbox_pred[0].new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            if num_gt_bboxes == 1:# 只有head_bboxes有gt
                num_gts = 1
                # assign 前两维 to background
                assigned_gt_inds[:-1] = 0
                assigned_gt_inds[-1] = 1
                assigned_labels[-1] = 2

            else:
                num_gts = 3
                
                assigned_gt_inds[0] = 1
                assigned_gt_inds[1] = 2
                assigned_gt_inds[2] = 3


                assigned_labels[0] = 0
                assigned_labels[1] = 1
                assigned_labels[2] = 2

            assign_results.append(
                AssignResult(
                    num_gts, assigned_gt_inds, None, labels=assigned_labels))
                    
        return assign_results