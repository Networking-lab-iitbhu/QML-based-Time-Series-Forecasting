#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=5
#SBATCH --time=20:02:20
#SBATCH --job-name=printjob
#SBATCH --error=job.%J.err_node
#SBATCH --output=job.%J.out_node
#SBATCH --partition=cpu

echo "hello world"
module load conda
source /home/apps/bio_tools/conda/etc/profile.d/conda.sh
conda activate qenv
lscpu
free -h
echo "module loaded"
echo "env loaded"
echo "python is running"
python ./main.py --num_latent 3 --num_trash 7
