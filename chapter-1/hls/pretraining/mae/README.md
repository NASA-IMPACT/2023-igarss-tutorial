## About this folder

This folder contains the pretraining model.

The model (as well as the code structure) follows [original mae repo](https://github.com/facebookresearch/mae)
with modifications including:
1. replace 2D patch embed with 3D patch embed
2. replace 2D positional embed with 3D positional embed
3. replace 2D patchify and unpatchify with 3D
4. etc.

We also have some experimental versions that includes some features from 
[SatMAE](https://arxiv.org/pdf/2207.08051.pdf) and [RingMo](https://ieeexplore.ieee.org/abstract/document/9844015),
but we are currently using this stable version in this repo for downstream tasks.

