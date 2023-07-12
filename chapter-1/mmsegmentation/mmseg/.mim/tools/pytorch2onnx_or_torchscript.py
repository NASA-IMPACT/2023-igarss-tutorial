"""
Script to convert mmsegmentation model state_dict to ONNX/Torchscript formats.

From the repository root directory, run

    python mmsegmentation/tools/pytorch2onnx_or_torchscript.py

ONNX and Torchscript files will be saved under the `pretrain_ckpts/` folder.
"""
import mmcv
import torch

import mmseg
from tools.pytorch2onnx import _demo_mm_inputs, pytorch2onnx
from tools.pytorch2torchscript import pytorch2libtorch

# %%
# Load model configuration file
cfg = mmcv.utils.Config.fromfile(
    filename="/home/workdir/hls-foundation/mmsegmentation/configs/tornado_tracks_config/geospatial_fm_config.py"
)
# Check that path to pretrained weights is in the config
assert (
    cfg.model["pretrained"]
    == "/home/workdir/hls-foundation/pretrain_ckpts/epoch-832-loss-0.0473.pt"
)

# Initialize pretrained model
model: torch.nn.Module = mmseg.models.build_segmentor(
    cfg=cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg"),
)
assert isinstance(model, torch.nn.Module)

# %%
# Saving using mmsegmentation tools
# Save to ONNX
assert model.decode_head.num_classes == 2
size: tuple = (1, 6, 1, 224, 224)  # batch Number, Channels, Time, Height, Width
N, C, T, H, W = size
mm_inputs = {
    "imgs": torch.rand(*size, requires_grad=True),
    "img_metas": [
        {
            "img_shape": (H, W, C),
            "ori_shape": (H, W, C),
            "pad_shape": (H, W, C),
            "filename": "<demo>.png",
            "scale_factor": 1.0,
            "flip": False,
        }
    ],
    "gt_semantic_seg": torch.randint(low=0, high=1, size=size, dtype=torch.uint8),
}
pytorch2onnx(
    model=model,
    mm_inputs=mm_inputs,
    opset_version=12,
    show=True,
    output_file="/home/workdir/hls-foundation/pretrain_ckpts/epoch-832-loss-0.0473.onnx",
)

# Save to Torchscript
pytorch2libtorch(
    model=model,
    input_shape=(1, 6, 1, 224, 224),
    show=True,
    output_file="/home/workdir/hls-foundation/pretrain_ckpts/epoch-832-loss-0.0473-torchscript.pt",
)
