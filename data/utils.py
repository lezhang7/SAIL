import os
import json
import csv
import torch
from tqdm import tqdm
import itertools    
import time
import pandas as pd
from data.data_config import DATADIR


# instruction = "Summarize the following sentences: "
instruction = ""

def load_data(args):
    data_name, source_caption, domain = args.data, args.source_caption, args.domain
    if data_name in DATADIR:
        data_config = DATADIR[data_name]
        data_file = data_config['annotation']
    else:
        data_file = data_name
    if data_file.endswith('.json'):
        return load_json_data(data_file, data_config['imagedir'])
    elif data_file.endswith('.jsonl'):
        return load_jsonl_data(data_file, data_config['imagedir'])
    elif data_file.endswith('.csv'):
        return load_csv_data(data_file, data_config['imagedir'], source_caption, domain)
    elif data_file.endswith('.tar') or '.tar' in data_file:
        return load_wds_data(data_file, domain, args.batch_size)
    else:
        raise ValueError('Unsupported data format')

def load_json_data(data_file, image_dir):
    with open(data_file, 'r') as f:
        data = json.load(f)
    sentences = [sample['conversations'][-1]['value'] for sample in data]
    images = [os.path.join(image_dir, sample['image']) for sample in data]
    return sentences, images

def load_jsonl_data(data_file, image_dir):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    sentences = [sample['conversations'][-1]['value'] for sample in data]
    images = [os.path.join(image_dir, sample['image']) for sample in data]
    return sentences, images

def read_csv_column(file_path, column_name, chunksize=None):
    start_time = time.time()  # Start timing
    column_data = []
    
    # Choose reading method based on whether chunksize is used
    if chunksize:
        # Read in chunks
        for chunk in pd.read_csv(file_path, usecols=[column_name], chunksize=chunksize):
            column_data.extend(chunk[column_name].tolist())
    else:
        # Read directly
        column_data = pd.read_csv(file_path, usecols=[column_name])[column_name].tolist()
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    
    # Output results and runtime
    print(f"Time taken to read '{column_name}' column: {elapsed_time:.2f} seconds")
    return column_data

def load_csv_data(data_file, image_dir, source_caption, domain):
    # Read image paths and captions using read_csv_column function
    images = []
    sentences = []
    if domain == 'image':
        image_paths = read_csv_column(data_file, 'Image Path')
        for image_path in image_paths:
            try:
                full_image_path = os.path.join(image_dir, image_path)
                images.append(full_image_path)
            except Exception as e:
                print(f"Error processing image path: {e}")
                images.append(None)  # Append None for failed image paths
    elif domain == 'text':
        sentences = read_csv_column(data_file, source_caption)
        sentences = [instruction + sentence for sentence in sentences]
    
    return sentences, images

def load_wds_data(data_file, domain, batch_size):
    """
    Load data from WebDataset format (.tar files)
    
    Args:
        data_file: Path to WebDataset .tar file or URL pattern (e.g., 'datasets/cc3m_recap/cc3m-train-{0000..0575}.tar')
        data_config: Configuration dictionary containing dataset details
        source_caption: Caption field to use
        domain: 'text' or 'image' domain to process
        
    Returns:
        sentences, images: Lists of sentences and image paths/tensors
    """
    import webdataset as wds
    
    # Handle sharded dataset patterns
    print(f"Loading WebDataset from: {data_file}")
    # Create dataset with appropriate handler for the specific domain
    if domain == 'image':
        # For image domain, extract image data
        dataset = (
            wds.WebDataset(data_file)
            .select(lambda x: 'jpg' in x or 'png' in x)
            .decode('pil')
            .to_tuple('jpg;png')
            .batched(batch_size)
        )
        
    elif domain == 'text':
        # For text domain, extract caption data
        dataset = (
            wds.WebDataset(data_file)
            .decode()
            .to_tuple('json')
            .batched(batch_size)
        )
 
    return dataset
