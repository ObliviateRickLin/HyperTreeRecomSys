#!/bin/bash
#$ -cwd
#$ -j y
#$ -o train_output.$JOB_ID
#$ -l gpu,A6000,cuda=1,h_data=16G,h_rt=24:00:00

# Load modules
module load python/3.9.6
module load cuda

# Setup environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/beauty_tokenizer
mkdir -p data/mlm_data
mkdir -p models/mlm

# Set CUDA config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Initialize tokenizer
python beauty_tokenizer_init.py

# Prepare data
python -m src.data --max_samples 10000 --ensure_all_tokens

# Start training
python -m src.mlm_model \
  --mode train \
  --train_file data/mlm_data/train_mlm.txt \
  --val_file data/mlm_data/val_mlm.txt \
  --tokenizer_path data/beauty_tokenizer \
  --output_dir models/mlm \
  --batch_size 8 \
  --max_length 128 \
  --mlm_probability 0.15 \
  --num_epochs 40 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --use_wandb \
  --wandb_project "amazon-beauty-mlm" \
  --wandb_run_name "mlm-new-head-training-v2"
