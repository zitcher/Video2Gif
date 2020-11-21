#!/bin/bash
#SBATCH --job-name=train1
#SBATCH --output=train1.out

#SBATCH --time=12:00:00

#SBATCH -n 1
#SBATCH -p gpu --gres=gpu:1

#SBATCH --mem=16G

module load python/3.7.4
module load cuda/10.1.105
module load cudnn/7.6.5
module load graphviz/2.40.1
source /users/zhoffman/Video2Gif/env/bin/activate
python /users/zhoffman/Video2Gif/train.py