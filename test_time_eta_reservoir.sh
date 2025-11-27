#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:0:0
#SBATCH --partition=nlpgpo

source /ubc/cs/home/s/shenranw/.bashrc
source /ubc/cs/home/s/shenranw/scratch/envs/LREATA/.venv/bin/activate

cd /ubc/cs/home/s/shenranw/LREATA
python test_time.py --cfg cfgs/cifar10_c/Standard/eta_reservoir.yaml