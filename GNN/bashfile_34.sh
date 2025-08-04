#!/bin/sh

# SET JOB NAME
#BSUB -J MADReg_34

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpua100

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"

# number of cores to use
#BSUB -n 10

# gb memory per core
#BSUB -R "rusage[mem=4G]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 24:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

# -- end of LSF options --

module load python3/3.12.4
source .venv/bin/activate

python main.py --config configs/config_madreg_34.json --trial 1 &
python main.py --config configs/config_madreg_34.json --trial 2 &
python main.py --config configs/config_madreg_34.json --trial 3 &
wait
