#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks=8
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
/home/jusun/dever120/NCVX-Neural-Structural-Optimization
export PATH=/home/jusun/dever120/miniconda3/envs/rydevera/bin:$PATH
python tasks.py --problem_name "roof_128x128_0.15" --num_trials=100 --maxit=1500 --requires_flip