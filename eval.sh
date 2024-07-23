#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l.3:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/eval/output-%j.txt
#SBATCH --error=./slurm_logs/eval/error-%j.txt 


module load miniconda/3
conda init
conda activate openflamingo
# vision_model="facebook/dinov2-base"
vision_model="facebook/dinov2-large"

# text_model="sentence-transformers/all-mpnet-base-v2"
text_model="Alibaba-NLP/gte-large-en-v1.5"
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"

export HF_AUTH_TOKEN="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"

for train in cc3m_gtedinoL_bs_32768_lion_mean_lr_1e-5_star7_d1024
do
    for epoch in {10..300..10}; 
    do
        for task in imagenet winoground COCO
        do
            checkpoint_path="./logs/${train}/checkpoints/epoch_${epoch}.pt"
            # check if the checkpoint exists
            if [ ! -f $checkpoint_path ]; then
                echo "Checkpoint not found: $checkpoint_path"
                continue
            fi
            echo "Evaluating checkpoint: $checkpoint_path"

            # python eval.py \
            #     --head-weights-path $checkpoint_path \
            #     --task $task \
            #     --vision-model $vision_model \
            #     --text-model $text_model \
            #     --linear-align \
            #     --linear-type star \
            #     --target-dimension 1024

            python eval.py \
                --head-weights-path $checkpoint_path \
                --task $task \
                --vision-model $vision_model \
                --text-model $text_model \
                --linear-align \
                --linear-type star \
                --target-dimension 1024
        done
    done
done