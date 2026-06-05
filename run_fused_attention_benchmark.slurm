#!/bin/bash
#SBATCH --job-name=fused_attention_benchmark
#SBATCH --account=amath-ckpt
#SBATCH --partition=ckpt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=fused_attention_output_%j.log

export MPLCONFIGDIR=/gscratch/amath/$USER/.matplotlib
export TORCH_HOME=/gscratch/amath/$USER/.torch

export PYTHONDONTWRITEBYTECODE=1

module load conda
conda activate fused_attention_env

ncu --set roofline -o fused_attention_benchmark python3 fused_attention_benchmark.py
