#!/bin/bash
#SBATCH -N 1
#SBATCH --time=10:02:20
#SBATCH --job-name=printjob
#SBATCH --error=job.%J.err_node
#SBATCH --output=job.%J.out_node
#SBATCH --partition=cpu

echo "hello world"
module load miniconda_23.5.2_python_3.11.4
echo "module loaded"
source ./env3/bin/activate
echo "env loaded"
echo "python is running"
python ./main.py
