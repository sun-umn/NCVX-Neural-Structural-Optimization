#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=16
#SBATCH --mem=100g
#SBATCH --tmp=40g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
/home/jusun/dever120/NCVX-Neural-Structural-Optimization
export PATH=/home/jusun/dever120/miniconda3/envs/rydevera/bin:$PATH
python -m tasks run-multi-structure-pipeline --model_size='large' --structure_size='medium'