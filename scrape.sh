#!/bin/bash
#SBATCH --job-name=render_images
#SBATCH --output=render3.out

#SBATCH --time=24:00:00

#SBATCH -c 1

#SBATCH --mem=8G

module load python/3.9.0
source /users/zhoffman/Video2Gif/venv/bin/activate
python /users/zhoffman/Video2Gif/yahooscraper.py -s 90000 -e 100000
