#!/bin/bash
#SBATCH --job-name=scraping
#SBATCH --partition=long-cpu                    # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=256G           
#SBATCH --time=48:00:00                                    
#SBATCH --output=./slurm_logs/download_mllm_enhanced_data/job_output-%j.txt
#SBATCH --error=./slurm_logs/download_mllm_enhanced_data/job_error-%j.txt 

export NUMEXPR_MAX_THREADS=32

# steal from https://github.com/ExplainableML/flair/blob/main/preprocess/scraping_cc3m.sh
img2dataset \
    --url_list ./datasets/cc3m_3long_3short_1raw_captions_url.parquet \
    --input_format "parquet" \
    --url_col "Image Path" \
    --output_format "webdataset" \
    --output_folder ./datasets/dreamlipcc3m \
    --processes_count 16 \
    --thread_count 32 \
    --number_sample_per_shard 5000 \
    --save_additional_columns '["raw_caption", "shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]'