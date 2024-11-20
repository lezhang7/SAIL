#!/bin/bash
#SBATCH --job-name=encode_data
#SBATCH --partition=unkillable                       # Ask for unkillable job
#SBATCH --cpus-per-task=4                         # Ask for 8 CPUs
#SBATCH --gres=gpu:a100l:1                         # Request 2 GPUs
#SBATCH --ntasks-per-node=1                        # Ask for 1 task per node
#SBATCH --mem=32G           
#SBATCH --time=12:00:00                                    
#SBATCH --output=./slurm_logs/encode/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.out
#SBATCH --error=./slurm_logs/encode/%(%Y-%m-%d)T/%x_%j_%A_%a_${data}.err 

module load miniconda/3
conda init
conda activate openflamingo
module load cudatoolkit/12.4.0

#==============================================================================#
#                              VISION MODEL                                      #
#==============================================================================#
vision_model="facebook/dinov2-large"
# Available options:
# vision_model="ijepa-huge"                    # IJEPA
# vision_model="openai/clip-vit-large-patch14" # OpenAI CLIP
# vision_model="mae-base"                      # MAE
# vision_model="dinov1-vitb16"                # DINOv1
# vision_model="aim_1B"                       # AIM
# vision_model="ibot-base"                    # iBOT

#==============================================================================#
#                               TEXT MODEL                                       #
#==============================================================================#
text_model="nvidia/NV-Embed-v2"
# Available options:
# text_model="Alibaba-NLP/gte-large-en-v1.5"          # GTE large
# text_model="openai/clip-vit-large-patch14"          # CLIP
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"   # Qwen2

#==============================================================================#
#                                 DATA                                           #
#==============================================================================#
data="dreamclipcc3m"
# Available options:
# data="dreamclipcc3m"      # DreamCLIP CC-3M
# data="dreamclipcc12mhf"  # DreamCLIP CC-12M high-fidelity
# data="yfcc15m"           # DreamCLIP YFCC-15M

#==============================================================================#
#                                DOMAIN                                         #
#                         image or text encoding                                #
#==============================================================================#

domain="image" # "image" or "text", each time we only encode one modality

#==============================================================================#
#                             BATCH SIZE                                         #
#==============================================================================#
batch_size=2048 # adjust based on GPU memory
#==============================================================================#
#                           Additional options                                  #
#==============================================================================#
# Available options:
# HQ Long captions:  longIB_captions, longSV_captions, longLLA_captions
# HQ Short captions: shortIB_captions, shortSV_captions, shortLLA_captions
# Raw caption:    raw_caption
# agg_mode: "concat" (concatenate cls with all patch tokens and average pool) or "cls" (use cls token only)
source_caption="longSV_captions"
agg_mode="concat"


# Program 
gpu_count=$SLURM_GPUS_ON_NODE

if [ "$gpu_count" -eq 4 ]; then # If use multiple GPUs, please make sure the end_index is integer multiple of batch_size to avoid overite and cause image-text mismatch
    echo "bash output: Running tasks in parallel on multiple GPUs..."
    echo "bash output: Using vision model: $vision_model"
    echo "bash output: Using text model: $text_model"
    echo "bash output: Processing dataset: $data"
    echo "bash output: Using domain: $domain"
    echo "bash output: Each GPU will use save batch size of $batch_size"
    echo "bash output: Using source caption: $source_caption"
    CUDA_VISIBLE_DEVICES=0 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --end_index 6144000 --source_caption $source_caption &
    CUDA_VISIBLE_DEVICES=1 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 6144000 --end_index 12288000 --source_caption $source_caption &
    CUDA_VISIBLE_DEVICES=2 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 12288000 --end_index 18432000 --source_caption $source_caption &
    CUDA_VISIBLE_DEVICES=3 python encode.py --domain $domain --vision_model_name $vision_model --text_model_name $text_model --batch_size $batch_size --data $data --resume --start_index 18432000 --source_caption $source_caption &
    wait
else
    echo "bash output: Running tasks sequentially on a single GPU..."
    echo "bash output: Using vision model: $vision_model"
    echo "bash output: Using text model: $text_model"
    echo "bash output: Processing dataset: $data"
    echo "bash output: Using domain: $domain"
    echo "bash output: Using batch size: $batch_size"
    echo "bash output: Using source caption: $source_caption"
    python encode.py \
    --domain $domain \
    --vision_model_name $vision_model \
    --text_model_name $text_model \
    --batch_size $batch_size \
    --data $data \
    --resume \
    --source_caption $source_caption \
    --agg_mode $agg_mode
fi