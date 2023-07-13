#!/usr/bin/env bash
#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --account=training2308
#SBATCH --output=output.out
#SBATCH --error=error.er
#SBATCH --time=2:00:00
#SBATCH --job-name=TEST
#SBATCH --gres=gpu:1 --partition=dc-gpu
#SBATCH --hint=nomultithread

module --force purge

ml Stages/2023

source /p/project/training2308/gurung1/miniconda3/bin/activate py39

echo "Starting training"

export CUDA_AVAILABLE_DEVICES="0,1"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p training2308 \
    --job-name=train \
    --ntasks=2 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u mmsegmentation/tools/train.py mmsegmentation/configs/burn_scar_config/geospatial_fm_config.py --launcher="slurm" --cfg-options 'find_unused_parameters'=True
