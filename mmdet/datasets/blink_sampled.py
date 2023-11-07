import os.path as osp
import random
from collections import defaultdict

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from pycocotools.ytvos import YTVOS

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose


@DATASETS.register_module()
class YoutubeVISDataset_Sampled(CustomDataset):

    CLASSES = ('person_face')

    def __init__(self,
                 ann_file,
                 pipeline,
                 clip_length,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.clip_length = clip_length
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)  # tuple类型，就是上面的CLASSSES写的那些字符串类别

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file) # 返回的是(videoId,video frame)这样的list，数量为总帧数，用于getitem的index

        # filter data infos if classes are customized
        # if self.custom_classes:
        #     self.data_infos = self.get_subset_by_classes()

        if self.proposal_file is not None: # 没进这个
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs() # 在dataset中删掉一些帧，这些帧中，没有一个instance，只要这帧中有一个instance就不会被删
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            self.sampled_data_infos = self._sample_imgs()


        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        print(f'sample_num = {len(self.sampled_data_infos)}')
        print(f'origin__num = {len(self.data_infos)}')

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        self.youtube = YTVOS(ann_file)   # coco api来读其gt标注文件的
        self.cat_ids = self.youtube.getCatIds() # 就是类别1-40的数字组成的list
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)} # 变成一个字典{1:0,2:1,3:2 ..... 40:39}代表类别和标签的映射关系，实际上就是1-40变为0-39
        vid_ids = self.youtube.getVidIds() # 一个列表1-2238，是video的数量

        vid_infos = []
        for i in vid_ids:
            info = self.youtube.loadVids([i])[0] # 加载当前video id的video的基本信息，就是data['video']的len=9的dict
            info['filenames'] = info['file_names'] # 这为啥复制一个差不多的东西出来
            vid_infos.append(info)
        self.vid_infos = vid_infos # 各个video的一些基本信息

        img_ids = []
        vid2frame = defaultdict(list) # 现在好像是创建了一个空的东西，
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))
                vid2frame[idx].append(frame_id)
        # 上面循环嵌套的功能就是记录每个视频的帧的index（数据集视频原本是5帧一采样，现在变成012345这样连续的index）
        self.vid2frame = vid2frame
        return img_ids

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        # self.flag = np.zeros(len(self), dtype=np.uint8)
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        for i, (vid, frame_id) in enumerate(self.data_infos):
            video_info = self.vid_infos[vid]
            if video_info['width'] / video_info['height'] > 1:
                self.flag[i] = 1

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = []
        if self.filter_empty_gt:
            for i, (vid, frame_id) in enumerate(self.data_infos):
                vid_id = self.vid_infos[vid]['id']
                ann_ids = self.youtube.getAnnIds(vidIds=[vid_id]) # 返回Video id=1的annotation的id,这个video里有几个instance就会返回几个id
                ann_info = self.youtube.loadAnns(ann_ids) #根据返回的id取出它们的annotation信息
                # if frame_id>=len(ann['bboxes']):
                #     print(1)
                anns = [
                    ann['bboxes'][frame_id] for ann in ann_info
                    if ann['bboxes'][frame_id] is not None
                ]
                if anns:
                    ids_with_ann.append(1)    # 最终，ids_with_ann有完整帧数个元素(61845)如果某一帧没有instance,则该元素值为0，如果这一帧有的instance有，有的instance的bbox是none,则元素值也是1，最终，sum()=61341
                else:
                    ids_with_ann.append(0)
        for i, (vid, frame_id) in enumerate(self.data_infos): # 下面这个循环是进一步看有没有图像分辨率小于32的，实际没有
            if self.filter_empty_gt and not ids_with_ann[i]:
                continue
            if min(self.vid_infos[vid]['width'],
                   self.vid_infos[vid]['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds # 最终返回了61341个元素，对应了data_infos中的index，目前，同时存在bbox和bbox=none的帧，仍然在valid_inds中
    def _sample_imgs(self):
        """只采样包含眨眼的帧"""
        ids_with_blink = []
        if self.filter_empty_gt:
            for i, (vid, frame_id) in enumerate(self.data_infos):
                vid_id = self.vid_infos[vid]['id']
                ann_ids = self.youtube.getAnnIds(
                    vidIds=[vid_id])  # 返回Video id=1的annotation的id,这个video里有几个instance就会返回几个id
                ann_info = self.youtube.loadAnns(ann_ids)  # 根据返回的id取出它们的annotation信息
                anns = [
                    ann['blinks_binary'][frame_id] for ann in ann_info
                    if ann['blinks_binary'][frame_id] == 1
                ]
                if anns:
                    ids_with_blink.append(i)  # 最终，ids_with_ann有完整帧数个元素(61845)如果某一帧没有instance,则该元素值为0，如果这一帧有的instance有，有的instance的bbox是none,则元素值也是1，最终，sum()=61341
        return ids_with_blink

    def get_img_info(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_info = self.vid_infos[vid]
        img_info = dict(
            file_name=vid_info['file_names'][frame_id],
            filename=vid_info['filenames'][frame_id],
            width=vid_info['width'],
            height=vid_info['height'],
            frame_id=frame_id)
        return img_info

    def get_ann_info(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_id = self.vid_infos[vid]['id'] # 其实就是vid+1吧
        ann_ids = self.youtube.getAnnIds(vidIds=[vid_id])
        ann_info = self.youtube.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def get_cat_ids(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_id = self.vid_infos[vid]['id']
        ann_ids = self.youtube.getAnnIds(vidIds=[vid_id])
        ann_info = self.youtube.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _parse_ann_info(self, ann_info, frame_id):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_blinks = []

        for i, ann in enumerate(ann_info):  # 遍历每一个instance
            bbox = ann['bboxes'][frame_id]  # 提取这个instance的[frame_id]那一帧的bbox
            # area = ann['areas'][frame_id]
            if bbox is None: # 这里要注意，如果这一帧的某个instance gt是none,那么这个instance(这一帧)不会写在gt中
                continue
            x1, y1, w, h = bbox
            # if area <= 0 or w < 1 or h < 1:
            #     continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'] -
                              1)  # youtube instance id start from 1. 把从1开始变为从0开始算id
                gt_labels.append(self.cat2label[ann['category_id']]) # 这个好像也是类别从1开始变为从0开始，也就是json中的类别减1
                # gt_masks.append(self.youtube.annToMask(ann, frame_id)) # 应该是转化为一个0-1的mask 对于眨眼，先把mask去掉
                if frame_id >=len(ann['blinks_binary']):
                    print('weiduyouwenti!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                gt_blinks.append(ann['blinks_binary'][frame_id])
                # if ann['blink'][frame_id] == 1:
                #     print('blink=1 in dataloader')
                # else:
                #     print('blink = 0 in dataloader')
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32) # 就是从list转为numpy array
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_blinks = np.array(gt_blinks, dtype=np.int64)
        else:   # 这个循环我估计没进过吧，因为前面滤过一次valid_ins了
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)


        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # ann = dict(
        #     bboxes=gt_bboxes,
        #     labels=gt_labels,
        #     bboxes_ignore=gt_bboxes_ignore,
        #     masks=gt_masks,
        #     ids=gt_ids)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            blinks=gt_blinks,
            bboxes_ignore=gt_bboxes_ignore,
            ids=gt_ids)

        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        if self.test_mode:
            raise NotImplementedError
        while True:
            index = self.sampled_data_infos[idx]
            data = self.prepare_train_clip(index)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_clip(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_idxs = []
        for i in sample_range:
            valid_idx = (vid, i)
            if valid_idx in self.data_infos:
                valid_idxs.append(valid_idx)
        assert len(valid_idxs) > 0
        # valid_idxs.sort()
        # 上面几行是根据valid_idx来确定当前视频内有哪些帧可以被采样选取
        frame_interval = 2  # 现在是2帧一采样
        index_pre = [(vid, frame_id - frame_interval*i) for i in range(1, self.clip_length//2 + 1) if (frame_id - frame_interval*i) >= valid_idxs[0][1] and (vid,frame_id - frame_interval*i) in valid_idxs ]
        pre_res = [(vid, valid_idxs[0][1]) for i in range(0, self.clip_length//2 - len(index_pre))] # 补第一帧可用帧补剩下的
        index_pre = index_pre + pre_res
        index_post = [(vid, frame_id + frame_interval*i) for i in range(1, self.clip_length//2 +1) if (frame_id + frame_interval*i) <= valid_idxs[-1][1] and (vid,frame_id + frame_interval*i) in valid_idxs ]
        post_res = [(vid, valid_idxs[-1][1]) for i in range(0, self.clip_length//2 - len(index_post))] # 用最后一可用帧补
        index_post += post_res
        index_except_center = index_pre + index_post
        valid_idxs = [idx] + [self.data_infos.index(_) for _ in index_except_center]
        valid_idxs.sort()
        # try:
        #     valid_idxs = [idx] + [
        #         self.data_infos.index(_)
        #         for _ in random.sample(valid_idxs, self.clip_length - 1)
        #     ] # 这里好像是在一个视频里随机抽帧，未必连续
        #
        # except BaseException as e:
        #     print(e, vid_info)
        #     return None

        clip = []

        for _ in valid_idxs:
            clip.append(self.prepare_train_img(_))

            # if sum(clip[-1]['gt_blinks'].data) > 0 :
            #     print(f'there is 1 after self.prepare_train_img')

            if _ == valid_idxs[0]:
                for tsfm in self.pipeline.transforms:
                    if hasattr(tsfm, 'isfix'):
                        tsfm.isfix = True
            elif _ == valid_idxs[-1]:
                for tsfm in self.pipeline.transforms:
                    if hasattr(tsfm, 'isfix'):
                        tsfm.isfix = False

        data = {}
        for key in clip[0]:
            stacked = []
            stacked += [_[key].data for _ in clip]
            if isinstance(stacked[0], torch.Tensor) and clip[0][key].stack:
                stacked = torch.stack(stacked, dim=0)
            data[key] = DC(
                stacked,
                clip[0][key].stack,
                clip[0][key].padding_value,
                cpu_only=clip[0][key].cpu_only)  # 因为是多帧，所以stack一下，tensor第一维变成了t(至少img是)
        return data

    def prepare_test_clip(self, idx):
        raise NotImplementedError

    def prepare_train_img(self, idx):
        img_info = self.get_img_info(idx)   # 获取一些信息，比较重要的是图像的路径，其实该函数内部可以通过vid获取整个视频的信息list，只不过这里取出了frame_id那一帧的信息
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.sampled_data_infos)

    def evaluate(self,
                 results,
                 metric,
                 results_file='results.json',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False):
        self.result2json(results, results_file)

    def result2json(self, results, results_file):
        json_results = []
        vid_objs = {}
        for idx in range(len(self)):
            # assume results is ordered
            vid, frame_id = self.data_infos[idx]
            if idx == len(self) - 1:
                is_last = True
            else:
                _, frame_id_next = self.data_infos[idx + 1]
                is_last = frame_id_next == 0
            det, seg, id = results[idx]
            labels = []
            for i in range(len(det)):
                labels += [i for _ in range(len(det[i]))]
            det = np.vstack(det)
            segm = []
            for i in seg:
                segm += i
            ids = []
            for i in id:
                ids += i
            seg = segm
            id = ids

            for obj_index in range(len(id)):
                bbox = det[obj_index]
                segm = seg[obj_index]
                label = labels[obj_index]
                obj_id = id[obj_index]
                if obj_id not in vid_objs:
                    vid_objs[obj_id] = {'scores': [], 'cats': [], 'segms': {}}
                vid_objs[obj_id]['scores'].append(bbox[4])
                vid_objs[obj_id]['cats'].append(label)
                segm['counts'] = segm['counts'].decode()
                vid_objs[obj_id]['segms'][frame_id] = segm
            if is_last:
                # store results of  the current video
                for obj_id, obj in vid_objs.items():
                    data = dict()

                    data['video_id'] = vid + 1
                    data['score'] = np.array(obj['scores']).mean().item()
                    # majority voting for sequence category
                    data['category_id'] = np.bincount(np.array(
                        obj['cats'])).argmax().item() + 1
                    vid_seg = []
                    for fid in range(frame_id + 1):
                        if fid in obj['segms']:
                            vid_seg.append(obj['segms'][fid])
                        else:
                            vid_seg.append(None)
                    data['segmentations'] = vid_seg
                    json_results.append(data)
                vid_objs = {}
        mmcv.dump(json_results, results_file)
