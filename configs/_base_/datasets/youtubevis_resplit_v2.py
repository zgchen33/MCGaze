# dataset settings
dataset_type = 'YoutubeVISDataset'
# dataset_type = 'YoutubeVISDataset_Sampled'
# data_root = "/data/data1/zengwenzheng/code/dataset_building/BlinkTeViT/data/20220917_with_blink/"
data_root = "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/"
clip_length = 11    # 5
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='LoadAnnotations',
    #     with_bbox=True,
    #     with_mask=True,
    #     with_id=True,
    #     bitmask=True),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_id=True),
    dict(type='Resize', img_scale=[(640, 360)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(
    #     type='Collect',
    #     keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_ids']),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_blinks','gt_ids']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        clip_length=clip_length,
        img_prefix=data_root + 'train_rawframes/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '11111val.json',   # ann_file=data_root + 'annotations/valid.json'
        clip_length=clip_length,
        img_prefix=data_root + '21val/',     # img_prefix=data_root + 'valid/'
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '11111test.json',     # ann_file=data_root + 'annotations/test.json'
        clip_length=clip_length,
        img_prefix=data_root + '12test/', #  img_prefix=data_root + 'test/'
        pipeline=test_pipeline))
evaluation = dict(metric=['segm'])
