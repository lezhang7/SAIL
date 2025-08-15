#!/bin/bash
#SBATCH --job-name=vq16_dreamlipcc3m_mlph1024o768_s20b10
#SBATCH --partition=long                    # Ask for unkillable job
#SBATCH --cpus-per-task=16                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=24:00:00                                    
#SBATCH --output=./slurm_logs/train/job_output-%j.txt
#SBATCH --error=./slurm_logs/train/job_error-%j.txt 
# ----------------------TRAIN SETTING------------------------

epoch_num=500
lr=1e-5
bs=32768
width_factor=8
logit_scale=20
logit_bias=-8

text_embedding_list="/home/mila/l/le.zhang/scratch/SAIL/data/tensor_data/text_embedding/NV-Embed-v2/dreamlipcc3m_raw_caption" 
image_embedding_list="/home/mila/l/le.zhang/scratch/SAIL/data/tensor_data/image_embedding/VQ-16/dreamlipcc3m_concat"
extra_text_embedding_list="/home/mila/l/le.zhang/scratch/SAIL/data/tensor_data/text_embedding/NV-Embed-v2/dreamlipcc3m_shortSV_captions"
output_name="sail_vq16_dreamlipcc3m_mlph1024o768_s20b-8"
# ------------------------------------------------------------


python main.py \
    --text-embedding-list $text_embedding_list \
    --image-embedding-list $image_embedding_list \
    --extra-text-embedding-list $extra_text_embedding_list \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 2 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 0 \
    --optimizer lion \
    --output-dim 768 \
    --hidden-dim 1024 \
    --siglip \
    --wd 1e-4 \
    --linear-type vqdensify \
    --log-every-n-steps 5 \
    --wandb-project-name sail_train \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias



if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi
