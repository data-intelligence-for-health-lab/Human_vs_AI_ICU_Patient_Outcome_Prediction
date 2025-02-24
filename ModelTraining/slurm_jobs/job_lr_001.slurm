#!/bin/bash
#SBATCH --job-name=lrc_hosp_mort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=100:00:00
#SBATCH --mem=500g
#SBATCH --partition=gpu
#SBATCH --output=lr_c_hosp_mort_lrc.out
#SBATCH --error=lr_c_hosp_mort_lrc.err
source ~/miniconda3/bin/activate
conda activate myenv
cd ~/thesis/GithubFolder/jupyterlab/optuna_code/
python3 source_code/main.py --model_type C --outcome HOSP_EXPIRE_FLAG --model_name LR_C --id C001
conda deactivate