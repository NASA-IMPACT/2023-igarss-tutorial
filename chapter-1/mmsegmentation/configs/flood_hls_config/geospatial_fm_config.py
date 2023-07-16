import os

### Configs
# data loader related
num_frames = int(os.getenv("NUM_FRAMES", 1))  # timesteps to load
img_size = int(os.getenv("IMG_SIZE", 224))
num_workers = int(os.getenv("DATA_LOADER_NUM_WORKERS", 2))

# model related
num_layers = int(os.getenv("NUM_LAYERS", 6))
patch_size = int(os.getenv("PATCH_SIZE", 16))
embed_dim = int(os.getenv("EMBED_DIM", 512))
num_heads = int(os.getenv("NUM_HEADS", 8))
tubelet_size = 1
checkpoint = os.getenv("CHECKPOINT_PATH", "")

user_name = os.environ.get("USER")
experiment = "flood"

work_dir = f"/p/project/training2308/{user_name}/{experiment}/"

save_path = work_dir

_base_ = ["../_base_/default_runtime.py", "../_base_/schedules/schedule_160k.py"]

dataset_type = "GeospatialDataset"
# dataset_type = 'CustomDataset'

data_root = f"/p/project/training2308/data/{experiment}/"  # changed data root folder
# TODO: @Hamed, @Steve, this is just an example normalization and not adjusted to your data - pls update with your values

img_norm_cfg = dict(means=[0.09840321, 0.12590889, 0.11499668,  0.31537666], stds=[0.0617847 , 0.05957831, 0.06571424, 0.09343068])
## bands = [0,1,2,3]
bands = [2,1,0,3]

tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)

train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=-1,
        nodata_replace=2,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="GeospatialRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

val_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=-1,
        nodata_replace=2,
    ),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="GeospatialRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(
        type="Collect",
        keys=["img", "gt_semantic_seg"],
        meta_keys=[
            "img_info",
            "ann_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
            "gt_semantic_seg",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up={"2": 1, "3": 2},
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type="GeospatialDataset",
        data_root=data_root,
        img_dir="training/images",
        ann_dir="training/annotations",
        pipeline=train_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
        ignore_index=-1,
    ),
    val=dict(
        type="GeospatialDataset",
        data_root=data_root,
        img_dir="validation/images",
        ann_dir="validation/annotations",
        # pipeline=val_pipeline,
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
        ignore_index=-1,
    ),
    test=dict(
        type="GeospatialDataset",
        data_root=data_root,
        img_dir="validation/images",
        ann_dir="validation/annotations",
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
        ignore_index=2,
        gt_seg_map_loader_cfg=dict(nodata=-1, nodata_replace=2),
    ),
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.9,
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

# This checkpoint config is later overwritten to allow for better logging in mmseg/apis/train.py l. 163
checkpoint_config = dict(
    # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=1350,
    out_dir=save_path,
)

evaluation = dict(interval=10, metric="mIoU", pre_eval=True, save_best="mIoU")
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)

optimizer_config = dict(grad_clip=None)

# epoch_config = dict(
#     epochs=10
# )
runner = dict(max_iters=27000)

workflow = [
    ("train", 1),
    ("val", 1),
]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 40000 iterations according to the `runner.max_iters`.
# workflow = [('train', 1)]

device = "cuda"
norm_cfg = dict(type="BN", requires_grad=True)

model = dict(
    type="TemporalEncoderDecoder",
    pretrained="/p/project/training2308/hls-foundation/epoch-832-loss-0.0473.pt",
    backbone=dict(
        type="PretrainVisionTransformer",
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
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    decode_head=dict(
        num_classes=2,
        in_channels=embed_dim * num_frames,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="DiceLoss", use_sigmoid=False, loss_weight=1),
    ),
    auxiliary_head=dict(
        num_classes=2,
        in_channels=embed_dim * num_frames,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="DiceLoss", use_sigmoid=False, loss_weight=1),
    ),
    train_cfg=dict(frozen=False),
    test_cfg=dict(mode="slide", stride=(128, 128), crop_size=(224, 224), frozen=False),
)

epoch_config = dict(epochs=50)
