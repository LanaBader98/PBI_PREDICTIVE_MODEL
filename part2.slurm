#!/bin/bash -l
#SBATCH --job-name=part2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/part2_out.log
#SBATCH --error=logs/part2_error.log

module purge
source /ibex/user/baderl/miniforge/bin/activate
conda activate /ibex/user/baderl/conda-environments/chemprop

export PYTHONPATH=/ibex/user/baderl/projects/PBI_project:$PYTHONPATH

cd /ibex/user/baderl/projects/PBI_project/part_2

python predictive_model.py