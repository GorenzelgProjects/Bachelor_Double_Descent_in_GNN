#!/bin/sh

# SET JOB NAME
#BSUB -J GS_PPR

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"

# number of cores to use
#BSUB -n 15

# gb memory per core
#BSUB -R "rusage[mem=8G]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 24:00
#BSUB -o hpc/gat/output_%J.out 
#BSUB -e hpc/gat/error_%J.err 

# -- end of LSF options --

module load python3/3.12.4
source .venv/bin/activate

python GNN_double_descent/model_wrapper_gpu.py --config="GNN_double_descent/configs/config_GS_ppr_1.json" &
python GNN_double_descent/model_wrapper_gpu.py --config="GNN_double_descent/configs/config_GS_ppr_2.json" &
python GNN_double_descent/model_wrapper_gpu.py --config="GNN_double_descent/configs/config_GS_ppr_3.json" &
python GNN_double_descent/model_wrapper_gpu.py --config="GNN_double_descent/configs/config_GS_ppr_4.json" &
python GNN_double_descent/model_wrapper_gpu.py --config="GNN_double_descent/configs/config_GS_ppr_5.json" &
wait
