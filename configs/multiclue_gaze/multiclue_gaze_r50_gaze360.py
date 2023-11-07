_base_ = [
    '../_base_/datasets/gaze360.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

num_stages = 4
clip_length = 7

model = dict(
    type='MultiClueGaze',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1, 
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='FixedEmbeddingRPNHead',
        proposal_feature_channel=256),
    roi_head=dict(
        type='MultiClueGazeROIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='GazeSTQIHead',
                num_classes=3,
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
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        gaze_head=[
            dict(
                type='GazeHead',
                in_channels=256,
                loss_gaze=dict(type='GazeArccosLoss', loss_weight=6.0),
                loss_temp=dict(type='GazeTempLoss', clip_len=clip_length, loss_weight=1.0)
            ) for _ in range(num_stages)
        ]
    ),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(type='FixedAssigner'),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1,
                mask_size=28,
            ) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None, rcnn=dict(max_per_img=2, mask_thr_binary=0.5))
    )

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))

lr_config = dict(policy='step', step=[6000], warmup_iters=1000)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=7000)
checkpoint_config = dict(interval=1000)    

work_dir = './work_dirs/multi-clue_gaze_r50_gaze360'