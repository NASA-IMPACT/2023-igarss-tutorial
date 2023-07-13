import os

gfm_model_name = '${gfm_ckpt}'

experiment = "flood"

work_dir = f'fine-tune-checkpoints/{experiment}'
save_path = f'fine-tune-checkpoints/{experiment}'

### Configs
# data loader related
num_frames = int(os.getenv('NUM_FRAMES', 1))
img_size = int(os.getenv('IMG_SIZE', 224))
num_workers = int(os.getenv('DATA_LOADER_NUM_WORKERS', 2))

# model related
num_layers = int(os.getenv('NUM_LAYERS', 6))
patch_size = int(os.getenv('PATCH_SIZE', 16))
embed_dim = int(os.getenv('EMBED_DIM', 512))
num_heads = int(os.getenv('NUM_HEADS', 8))
tubelet_size = 1
checkpoint = os.getenv("CHECKPOINT_PATH", "")

_base_ = ["../_base_/default_runtime.py", "../_base_/schedules/schedule_160k.py"]

dataset_type = 'GeospatialDataset'
data_root = f"/p/project/training2308/{experiment}/"  # changed data root folder

###----- Changed depending on data source
#--   Needs to be changed to use GeoDN data
#img_norm_cfg = dict(means=[0.21531178, 0.20978154, 0.18528642, 0.48253757], stds=[0.10392396, 0.10210076, 0.11696766, 0.19680527])
#bands = [1,2,3,8]

img_norm_cfg = dict(means=[0.09840321, 0.12590889, 0.11499668,  0.31537666], stds=[0.0617847 , 0.05957831, 0.06571424, 0.09343068])
## bands = [0,1,2,3]
bands = [2,1,0,3]

tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=False),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False, nodata=-1, nodata_replace=2),
    dict(type='BandsExtract', bands=bands),
    dict(type='ConstantMultiply', constant=0.0001),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='GeospatialRandomCrop', crop_size=crop_size),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, tile_size, tile_size)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(tile_size, tile_size)),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=False),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False, nodata=-1, nodata_replace=2),
    dict(type='BandsExtract', bands=bands),
    dict(type='ConstantMultiply', constant=0.0001),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='GeospatialRandomCrop', crop_size=crop_size),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, tile_size, tile_size)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(tile_size, tile_size)),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=['img_info', 'ann_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'gt_semantic_seg']),

]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=False),
    dict(type='BandsExtract', bands=bands),
    dict(type='ConstantMultiply', constant=0.0001),
    dict(type='ToTensor', keys=['img']),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='AddTimeDimension'),
    dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
    dict(type='CollectTestList', keys=['img'], meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="training/images",
        ann_dir="training/annotations",
        pipeline=train_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
        ignore_index=2,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="validation/images",
        ann_dir="validation/annotations",
        pipeline=val_pipeline,
        # pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
        ignore_index=2,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="validation/images",
        ann_dir="validation/annotations",
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
        ignore_index=2,
        gt_seg_map_loader_cfg=dict(nodata=-1, nodata_replace=2)))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=${learning_rate}, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(
                     num_layers=12,
                     layer_decay_rate=0.9,
                 )
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# loss_class_1 = 0.15
# loss_class_2 = 0.85
# loss_class_3 = 0.0

# This checkpoint config is later overwritten to allow for better logging in mmseg/apis/train.py l. 163
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=${num_iterations},
    out_dir=save_path)

evaluation = dict(interval=${iter_per_eval}, metric='mIoU', pre_eval=True, save_best='mIoU')

optimizer_config = dict(grad_clip=None)

# epoch_config = dict(
#     epochs=10
# )
runner=dict(max_iters=${num_iterations})
# workflow = [('train', 1),('val', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.
workflow = [('train', 1),('val', 1)] 

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    # type='EncoderDecoder',
    type='TemporalEncoderDecoder',
    pretrained="/p/project/training2308/hls-foundation/epoch-832-loss-0.0473.pt",
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
        num_classes=3,
        in_channels=embed_dim,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=${decode_head_conv},
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(${loss_function}))
    ,
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(128,128), crop_size=(224,224))
)

if ${aux_head}==True:
    model['auxiliary_head']=dict(
        num_classes=3,
        in_channels=embed_dim,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(${loss_function}))