# Chapter 1: Fine-Tune a Pretrained HLS model
In this chapter we take a Pretrained HLS model and fine-tune it for two usecases: flood, burn_scars.

Training and validation files are located at `/p/project/training2308/burn_scars` and `/p/project/training2308/flood`.

The configuration files are located in [Burn scars config](chapter-1/mmsegmentation/configs/burn_scars_config/geospatial_fm_config.py) and [Flood config](chapter-1/mmsegmentation/configs/flood_config/geospatial_fm_config.py)

For burn scars, training can be triggered using `sbatch burn_scars.sh`
For flood, training can be triggered using `sbatch flood.sh`

The fine-tuned models will be stored in `/p/project/training2308/<user>/burn_scars` or `/p/project/training2308/<user>/flood`. *Note: <user> corresponds to your username*
