#!/bin/bash
#SBATCH --job-name=download_mllm_enhanced_data
#SBATCH --partition=long-cpu                    # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=256G           
#SBATCH --time=96:00:00                                    
#SBATCH --output=./slurm_logs/download_mllm_enhanced_data/job_output-%j.txt
#SBATCH --error=./slurm_logs/download_mllm_enhanced_data/job_error-%j.txt 

# download the data from the url to datasets folder
mkdir -p ./datasets


# Download and convert YFCC15M data if not already present
if [ ! -f "./datasets/yfcc15m_3long_3short_1raw_captions_url.csv" ]; then
    echo "Downloading yfcc15m_3long_3short_1raw_captions_url.csv"   
    wget -P ./datasets https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/yfcc15m_3long_3short_1raw_captions_url.csv
else
    echo "yfcc15m_3long_3short_1raw_captions_url.csv already exists, skipping download"
fi

# Download and convert CC3M data if not already present 
if [ ! -f "./datasets/cc3m_3long_3short_1raw_captions_url.csv" ]; then
    echo "Downloading cc3m_3long_3short_1raw_captions_url.csv"
    wget -P ./datasets https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc3m_3long_3short_1raw_captions_url.csv
else
    echo "cc3m_3long_3short_1raw_captions_url.csv already exists, skipping download"
fi

# Download and convert CC12M data if not already present
if [ ! -f "./datasets/cc12m_3long_3short_1raw_captions_url.csv" ]; then
    echo "Downloading cc12m_3long_3short_1raw_captions_url.csv"
    wget -P ./datasets https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc12m_3long_3short_1raw_captions_url.csv
else
    echo "cc12m_3long_3short_1raw_captions_url.csv already exists, skipping download"
fi

# Convert CSV files to parquet if parquet files don't exist
for dataset in "yfcc15m" "cc3m" "cc12m"; do
    if [ ! -f "./datasets/${dataset}_3long_3short_1raw_captions_url.parquet" ] && [ -f "./datasets/${dataset}_3long_3short_1raw_captions_url.csv" ]; then
        python convert_to_parquet.py -i "./datasets/${dataset}_3long_3short_1raw_captions_url.csv"
    else
        echo "Parquet file for ${dataset} already exists or CSV file missing, skipping conversion"
    fi
done



export NUMEXPR_MAX_THREADS=32
# steal from https://github.com/ExplainableML/flair/blob/main/preprocess/scraping_cc3m.sh
# Process each dataset
for dataset in "cc3m" "cc12m" "yfcc15m"; do
    output_folder="./datasets/dreamlip${dataset}"
    if [ ! -d "$output_folder" ]; then
        echo "Processing ${dataset} dataset..."
        img2dataset \
            --url_list "./datasets/${dataset}_3long_3short_1raw_captions_url.parquet" \
            --input_format "parquet" \
            --url_col "Image Path" \
            --output_format "webdataset" \
            --output_folder "$output_folder" \
            --processes_count 16 \
            --thread_count 32 \
            --number_sample_per_shard 5000 \
            --save_additional_columns '["raw_caption", "shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]'
    else
        echo "Output folder $output_folder already exists, skipping ${dataset} dataset processing"
    fi
done