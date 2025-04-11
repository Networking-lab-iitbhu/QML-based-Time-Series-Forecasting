#!/bin/bash
#SBATCH -N 1
#SBATCH --time=4:50:20
#SBATCH --job-name=lexiql
#SBATCH --error=job.%J.err_node
#SBATCH --output=job.%J.out_node
#SBATCH --partition=cpu

module load miniconda_23.5.2_python_3.11.4
source ./env3/bin/ac
python main.py
