#!/bin/bash
#
#SBATCH --job-name=GPU
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH --partition=research.q
#SBATCH --gres=gpu:GeForceRTX2070 ## Request 2 GPUs MAX: 8 GPUs

module load cuda/11.2
module load nvidia-hpc-sdk/21.2
module load gcc/10.2.0
module load miniconda/3

# v1.11.0 of Pytorch
# CUDA 10.2
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch 

#conda create --name bert-env --clone datathon
#conda install scikit-learn
#conda install transformers
#conda install -c conda-forge optuna
#conda init bash
eval "$(conda shell.bash activate new-env)"

#conda list

#conda activate datathon
#conda install -n datathon --freeze-installed pandas
#conda list

#python "/home/pfc/atoval/BertModel_threshold.py"
#python "/home/pfc/atoval/BertModel_multiclassification.py"
#python "/home/pfc/atoval/LSTM_multiclassification.py"
#python "/home/pfc/atoval/roBERTa_multiclassification.py"
#python "/home/pfc/atoval/BERT - LRFinder.py"
#python "/home/pfc/atoval/RoBERTa - LRFinder.py"
#python "/home/pfc/atoval/HypertuneBERT.py"

python "/home/pfc/atoval/BertModel_multiclassification.py"
