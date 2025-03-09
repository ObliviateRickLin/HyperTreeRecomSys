#!/bin/bash
#$ -cwd
#$ -j y
#$ -o hit_train_output.$JOB_ID
#$ -l gpu,A6000,cuda=1,h_data=16G,h_rt=24:00:00

# Load modules
module load python/3.9.6
module load cuda

# Setup environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install geoopt
pip install wandb

# Create directories
mkdir -p data/beauty_hierarchy
mkdir -p models/hit

# Set CUDA config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start training
python -m src.train \
  --meta_file data/meta_Beauty_2014.json.gz \
  --output_dir models/hit \
  --model_name "bert-base-uncased" \
  --batch_size 256 \
  --num_epochs 20 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --clustering_margin 5.0 \
  --centripetal_margin 0.1 \
  --use_wandb \
  --wandb_project "amazon-beauty-hit" \
  --wandb_run_name "hit-training-v1" 