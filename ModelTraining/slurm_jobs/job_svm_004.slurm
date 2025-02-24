#!/bin/bash
#SBATCH --job-name=svc_hosp_mort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=100:00:00
#SBATCH --mem=500g
#SBATCH --partition=gpu
#SBATCH --output=svc_hosp_mort.out
#SBATCH --error=svc_hosp_mort.err
source ~/miniconda3/bin/activate
conda activate myenv
cd ~/thesis/GithubFolder/jupyterlab/optuna_code/
python3 source_code/main.py --model_type C --outcome HOSP_EXPIRE_FLAG --model_name SVM_C --id C002
conda deactivate