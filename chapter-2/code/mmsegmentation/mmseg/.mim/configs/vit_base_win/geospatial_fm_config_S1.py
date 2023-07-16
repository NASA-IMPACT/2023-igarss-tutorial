import os

### Configs
# data loader related
train_dir = "/dev/shm/train"
val_dir = "/dev/shm/val"
num_frames = int(os.getenv('NUM_FRAMES', 1))
img_size = int(os.getenv('IMG_SIZE', 256))
bands = ["B02", "B03", "B04", "B05"]
nodata_value = 0.0001
random_cropping = True
num_workers = int(os.getenv('DATA_LOADER_NUM_WORKERS', 2))

# model related
num_layers = int(os.getenv('NUM_LAYERS', 12))
patch_size = int(os.getenv('PATCH_SIZE', 16))
embed_dim = int(os.getenv('EMBED_DIM', 1024))
num_heads = int(os.getenv('NUM_HEADS', 8))
mask_ratio = float(os.getenv('MASK_RATIO', 0.75))
tubelet_size = 1

# training related
batch_size = int(os.getenv('BATCH_SIZE', 4))
lr = float(os.getenv('LR', 1e-4))
gamma = float(os.getenv('LR_DECAY', 0.85))
epochs = int(os.getenv('NUM_EPOCHS', 50))
distributed_mode = "DDP"

_base_ = [
    '../_base_/models/geospatial_fm_base_config.py',  # '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'Sen1Floods11'
data_root = '/u/jkbk/ai_for_climate_impact/semantic_segmentation_sustainability/senfloods/v1.1/data/flood_events/HandLabeled'
img_norm_cfg = dict(
    mean=[0.6851, 0.5235, 0.6851], std=[0.0820, 0.1102, 0.0820], to_rgb=True)

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=512),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='S1Hand',
        ann_dir='LabelHand',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='S1Hand',
        ann_dir='LabelHand',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='S1Hand',
        ann_dir='LabelHand',
        pipeline=test_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(
                     num_layers=12,
                     layer_decay_rate=0.9,
                 )
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True,  # Whether count by epoch or not.
    interval=190)  # The save interval.

optimizer_config = dict(grad_clip=None)

model = dict(
    pretrained='/dccstor/cimf/sen1floods11/fm_downstream/initial_fm_training/epoch-78-loss-0.0779.pt',
    backbone=dict(
        type='PretrainVisionTransformer',
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        encoder_in_chans=len(bands),
        encoder_embed_dim=embed_dim,
        encoder_depth=num_layers,
        encoder_num_heads=num_heads,
        encoder_num_classes=0,
        decoder_num_classes=tubelet_size * patch_size * patch_size * len(bands),
        decoder_embed_dim=int(embed_dim / 2),
        decoder_depth=int(num_layers / 4),
        decoder_num_heads=int(num_heads / 2),
        mlp_ratio=4,
        qkv_bias=True,
        tubelet_size=tubelet_size,
    ),
    decode_head=dict(
        num_classes=2,
        ignore_index=2
    ),
    auxiliary_head=dict(
        num_classes=2,
        ignore_index=2
    ))
