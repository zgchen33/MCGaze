_base_ = './multiclue_gaze_r50_gaze360.py'

num_stages = 4
clip_length = 7 

# because "l2cs" dataset is diffrent with gaze360, 
# here we need to define a new "data" dict.  
dataset_type = 'Gaze360Dataset'
data_root = "data/l2cs/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_gaze=True,
        with_id=True),
    dict(type='Resize', img_scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_gazes','gt_ids']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(_delete_=True,
        type=dataset_type,
        ann_file=data_root + 'train.json',
        clip_length=clip_length,
        img_prefix=data_root + 'train_rawframes/',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '11111val.json',   # ann_file=data_root + 'annotations/valid.json'
        clip_length=clip_length,
        img_prefix=data_root + '21val/',     # img_prefix=data_root + 'valid/'
        pipeline=test_pipeline
        ),
    test=dict(_delete_=True,
    type=dataset_type,
    ann_file=data_root + '11111test.json',     # ann_file=data_root + 'annotations/test.json'
    clip_length=clip_length,
    img_prefix=data_root + '12test/', #  img_prefix=data_root + 'test/'
    pipeline=test_pipeline))


lr_config = dict(policy='step', step=[12000], warmup_iters=1000)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=13000) 

work_dir = './work_dirs/multiclue_gaze_r50_l2cs'