#!/bin/bash

CUDA_VISIBLE_DEVICES=0
export HUB_TOKEN="<your_hf_token>"                       # your hf token
export HUB_REPO_NAME="sdxl-controlnet-object-remover_v1" # models will be uploaded to this repository in hf account
export HUB_DATASET_NAME="Vimax97/Object_remover_v1"      # hf dataset

accelerate launch --mixed_precision="fp16" --num_processes=1 train_controlnet_sdxl.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --controlnet_model_name_or_path="alimama-creative/EcomXL_controlnet_inpaint" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir=./output \
  --train_data_dir=$HUB_DATASET_NAME \
  --resolution=1024 \
  --learning_rate=1e-5 \
  --max_train_steps=5000 \
  --validation_steps=50000 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --gradient_accumulation_steps=4 \
  --checkpoints_total_limit=1 \
  --push_to_hub \
  --hub_token=$HUB_TOKEN \
  --hub_model_id=$HUB_REPO_NAME \
  --report_to="wandb" 
