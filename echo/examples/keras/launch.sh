#!/bin/bash -l
#SBATCH --account=NAML0001
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128G
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -J hyper_opt
#SBATCH -o hyper_opt.out
#SBATCH -e hyper_opt.err
module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib
python run.py examples/keras/hyperparameter.yml examples/keras/model_config.yml
