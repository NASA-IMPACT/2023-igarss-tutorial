# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='PretrainedVisionTransformer',
        img_size=1024,
        # embed_dim=768,
        # depth=12,
        # num_heads=12,
        mlp_ratio=4,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.1,
        # use_abs_pos_emb=True,
        ),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight=[0.95, 0.05])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight=[0.95, 0.05])),
    # model training and testing settings
    train_cfg=dict(),
    #test_cfg=dict(mode='whole')
    test_cfg=dict(mode='slide', stride=(128,128), crop_size=(224,224))
    )

    # 