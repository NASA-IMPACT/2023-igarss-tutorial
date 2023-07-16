import os

### Configs
# data loader related
num_frames = int(os.getenv('NUM_FRAMES', 1))
img_size = int(os.getenv('IMG_SIZE', 224))
bands = ["B01", "B02", "B03", "B04"]
num_workers = int(os.getenv('DATA_LOADER_NUM_WORKERS', 2))

# model related
num_layers = int(os.getenv('NUM_LAYERS', 12))
patch_size = int(os.getenv('PATCH_SIZE', 16))
embed_dim = int(os.getenv('EMBED_DIM', 768))
num_heads = int(os.getenv('NUM_HEADS', 12))
tubelet_size = 1
checkpoint = os.getenv("CHECKPOINT_PATH", "")


save_path = '/home/workdir/burn_scar/hls-foundation/hls/downstream/finetune_ckpts/'

_base_ = [
    '../_base_/models/geospatial_fm_base_config.py',  # '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'Sen1Floods11'
#dataset_type = 'BurnScars'  #check for path iteration
#data_root = '/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled'

img_norm_cfg = dict(
    mean=[442.9805145263672, 699.5281867980957, 679.5248565673828, 2093.850761413574], std=[232.17283718767763, 236.6401508696276, 369.91184775358425, 850.2287590280677], to_rgb=True)  ## change the mean and std of all the bands

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
    samples_per_gpu=4,#changed from 4
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='S2Hand',
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

# This checkpoint config is later overwritten to allow for better logging in mmseg/apis/train.py l. 163
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True,  # Whether count by epoch or not.
    interval=5,
    out_dir=save_path)

optimizer_config = dict(grad_clip=None)

epoch_config = dict(
    epochs=100
)

workflow = [('train', 1),
            ('val', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.

model = dict(

    pretrained='/home/workdir/hls-model-weights/epoch-941-loss-0.0602.pt',
    backbone=dict(
        type='PretrainVisionTransformer',
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=num_layers,
        num_heads=num_heads,
        decoder_embed_dim=int(embed_dim / 2),
        decoder_depth=8,
        decoder_num_heads=num_heads,
        mlp_ratio=4.,
        norm_pix_loss=False
    ),
    decode_head=dict(
        num_classes=2,#changed from 3
        in_channels=embed_dim,
        # ignore_index=2
    ),
    auxiliary_head=dict(
        num_classes=2,#changed from 3
        in_channels=embed_dim,
        # ignore_index=2
    ))
