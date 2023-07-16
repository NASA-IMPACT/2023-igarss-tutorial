_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

experiment = 'NASA_burn_scars_pspnet'

work_dir = '/dccstor/geofm-finetuning/fire-scars/finetune_ckpts' + '/' + experiment
save_path = work_dir


gpu_ids = [0]
dataset_type = 'GeospatialDataset'
# dataset_type = 'CustomDataset'
data_root = '/dccstor/geofm-finetuning/fire-scars/finetune-data'  # changed data root folder

img_norm_cfg = dict(
    means=[442.98051, 699.52818, 679.52485, 2093.85076],
    stds=[232.17283, 236.64015, 369.91184, 850.22875])  ## change the mean and std of all the bands

img_norm_cfg = dict(
    means=[0.044298051, 0.069952818, 0.067952485, 0.209385076],
    stds=[0.023217283, 0.023664015, 0.036991184, 0.085022875])  ## change the mean and std of all the bands


bands = [0, 1, 2, 3]
num_frames=1
tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False, nodata=-1, nodata_replace=2),
    # dict(type='BandsExtract', bands=bands),
    dict(type='ConstantMultiply', constant=0.0001),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='GeospatialRandomCrop', crop_size=crop_size),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), tile_size, tile_size)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, tile_size, tile_size)),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False, nodata=-1, nodata_replace=2),
    # dict(type='BandsExtract', bands=bands),
    dict(type='ConstantMultiply', constant=0.0001),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='GeospatialRandomCrop', crop_size=crop_size),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), tile_size, tile_size)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, tile_size, tile_size)),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys=['img_info', 'ann_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
                    'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'gt_semantic_seg']),

]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    # dict(type='BandsExtract', bands=bands),
    dict(type='ConstantMultiply', constant=0.0001),
    dict(type='ToTensor', keys=['img']),
    dict(type='TorchNormalize', **img_norm_cfg),
    # dict(type='AddTimeDimension'),
    dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
    dict(type='CollectTestList', keys=['img'],
         meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
                    'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='scenes',
        ann_dir='masks',
        pipeline=train_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        # ignore_index=2,
        split="/dccstor/geofm-finetuning/fire-scars/finetune-data/training_data.txt"),
    val=dict(
        type=dataset_type,
        data_root="/dccstor/geofm-finetuning/fire-scars/finetune-data/",
        img_dir='validation',
        ann_dir='validation',
        # pipeline=val_pipeline,
        pipeline=test_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        # ignore_index=2,
        split="/dccstor/geofm-finetuning/fire-scars/finetune-data/validation_data.txt"),
    test=dict(
        type=dataset_type,
        data_root="/dccstor/geofm-finetuning/fire-scars/finetune-data/",
        img_dir='validation',
        ann_dir='validation',
        pipeline=test_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        # ignore_index=2,
        split="/dccstor/geofm-finetuning/fire-scars/finetune-data/validation_data.txt",
        gt_seg_map_loader_cfg=dict(nodata=-1, nodata_replace=2)))

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# loss_class_1 = 0.15
# loss_class_2 = 0.85
# loss_class_3 = 0.0

# This checkpoint config is later overwritten to allow for better logging in mmseg/apis/train.py l. 163
checkpoint_config = dict(
    # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=6300,
    out_dir=save_path)

evaluation = dict(interval=100, metric='mIoU', pre_eval=True, save_best='mIoU')

# epoch_config = dict(
#     epochs=10
# )
runner = dict(max_iters=6300)
workflow = [('train',
             1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.
# workflow = [('train', 1)]

model = dict(
    type='TemporalEncoderDecoder',
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', stride=(128, 128), crop_size=(224, 224))
)