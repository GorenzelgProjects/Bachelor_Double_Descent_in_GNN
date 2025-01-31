#!/bin/sh

# SET JOB NAME
#BSUB -J dd_gnn_1

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpua100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 4

# gb memory per core
#BSUB -R "rusage[mem=16G]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 24:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

module load python3/3.12.4
source .venv/bin/activate
python double_descent_deep_learning/Train_Cifar_experiment.py

