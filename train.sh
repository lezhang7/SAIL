#!/bin/bash
#SBATCH --job-name=gtedino-bs32768
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/train/output-%j.txt
#SBATCH --error=./slurm_logs/train/error-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

# setup training parameters
epoch_num=300
lr=1e-4
bs=65536
bs=32768

timestamp=$(date +%Y%m%d%H%M%S)
job_name=$SLURM_JOB_NAME

export NCCL_DEBUG=INFO

# 检查GPU数量
if [[ "$SLURM_GPUS_ON_NODE" -gt 1 ]]; then
    output_name="${job_name}"
    echo "Running on multiple GPUs"
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
    echo master port is $MASTER_POR
    export WORLD_SIZE=$SLURM_NTASKS_PER_NODE
    echo world size is $WORLD_SIZE
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo master addr is $MASTER_ADDR
    export OMP_NUM_THREADS=12

    torchrun --master_port $MASTER_PORT --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
             --train-data $train_data \
             --seed 42 \
             --resume latest \
             --save-frequency 1 \
             --report-to wandb \
             --warmup 50 \
             --batch-size $bs \
             --lr $lr \
             --wd 0.1 \
             --epochs $epoch_num \
             --workers 0 \
             --beta1 0.9 \
             --beta2 0.98 \
             --eps 1e-06 \
             --log-every-n-steps 10 \
             --wandb-project-name clip_training \
             --name $output_name
else
    echo "Running on a single GPU"
    # Be sure to name the output folder with the text and vision model name
    output_name="llava_vflan_gtedinoL_bs_${bs}_lion_mean"
    python main.py \
           --text-embedding-list /home/mila/l/le.zhang/scratch/light_align/data/text_embedding/gte-large-en-v1.5/LLaVA558K /home/mila/l/le.zhang/scratch/light_align/data/text_embedding/gte-large-en-v1.5/ALLaVAVFLAN \
           --image-embedding-list /home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-large/LLaVA558K /home/mila/l/le.zhang/scratch/light_align/data/image_embedding/dinov2-large/ALLaVAVFLAN  \
           --dataset-type embedding \
           --linear-align \
           --seed 42 \
           --save-frequency 10 \
           --report-to wandb \
           --batch-size $bs \
           --lr $lr \
           --siglip \
           --epochs 300 \
           --workers 0 \
           --wd 1e-07 \
           --log-every-n-steps 10 \
           --wandb-project-name clip_training \
           --name $output_name
fi

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo "Training failed. Cleaning up.../logs/${output_name}"
    # 假设输出文件夹是根据 $output_name 创建的
    rm -rf ./logs/$output_name
fi
