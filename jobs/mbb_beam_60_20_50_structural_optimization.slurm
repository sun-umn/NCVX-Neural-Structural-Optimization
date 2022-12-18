#!/bin/bash -l
#SBATCH --time=8:00:00
#SBATCH --ntasks=8
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
/home/jusun/dever120/NCVX-Neural-Structural-Optimization
export PATH=/home/jusun/dever120/miniconda3/envs/rydevera/bin:$PATH
python tasks.py --problem_name "mbb_beam_60x20_0.5" --num_trials=50 --maxit=1500 --resizes
