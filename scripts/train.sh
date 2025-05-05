#!/bin/bash

# Paramètres SLURM
#SBATCH -c 5
#SBATCH --partition=cbio-gpu
#SBATCH --gres=gpu:1
# SBATCH --exclude=node009,node003,node005,node007
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=60000
#SBATCH --time=30-00:00:00 
#SBATCH --output=dinov2_train.out
#SBATCH --job-name=dinov2_train

# Lancer le script Python
python ../dinov2/run/train/train.py \
    --nodes 1 \
    --config-file /cluster/CBIO/home/ablondel1/cell_SSL/cell_SSL/dinov2_script/dinov2/configs/train/vitl16_short_wsidataset.yaml \
    --output-dir /cluster/CBIO/data1/ablondel1/WSI_vesper_data/data_alice/train/dinov2
