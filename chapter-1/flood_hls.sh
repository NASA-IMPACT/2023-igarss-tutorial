#!/usr/bin/env bash
#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --account=training2308
#SBATCH --output=output_flood_hls.out
#SBATCH --error=error_flood_hls.er
#SBATCH --time=2:00:00
#SBATCH --job-name=TEST
#SBATCH --gres=gpu:1 --partition=dc-gpu
#SBATCH --hint=nomultithread

module --force purge

ml Stages/2023

source /p/project/training2308/<username>/miniconda3/bin/activate py39

echo "Starting training"
echo "python path:"
echo  `which python`
export CUDA_AVAILABLE_DEVICES="0,1"
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p training2308 \
    --job-name=train \
    --ntasks=2 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u mmsegmentation/tools/train.py mmsegmentation/configs/flood_hls_config/geospatial_fm_config.py --launcher="slurm" --cfg-options 'find_unused_parameters'=True
