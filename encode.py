import torch
import os
import json
import logging
import csv
import time
from tqdm import tqdm
from model import SentenceEmbedding, ImageEmbedding
from train.logger import setup_logging
from data.utils import load_data
from data.image_dataset import create_image_dataloader
import warnings
setup_logging(log_file=None, level=logging.INFO)
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
   )
    parser.add_argument('--vision_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--text_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--resume', action='store_true', help='Resume from existing embeddings')
    parser.add_argument('--domain', type=str, choices=['text', 'image'], required=True, help='Domain to encode')
    parser.add_argument('--source_caption', type=str, choices=['raw_caption', 'shortIB_captions', 'longIB_captions', 'shortSV_captions', 'longSV_captions', 'shortLLA_captions', 'longLLA_captions','caption'], required=True, help='Source caption')
    parser.add_argument('--save_name', type=str, default=None, help='Save name')
    parser.add_argument('--batch_size', type=int, default=32, help='Save batch size')
    parser.add_argument('--agg_mode', type=str, default='concat', help='Aggregation mode')
    parser.add_argument('--throughput', action='store_true', help='Calculate throughput')
    return parser.parse_args()



def encode_save(batch_data, encode_function, output_path, throughput=False):
    start_time = time.time()
    with torch.cuda.amp.autocast():  # Enable automatic mixed precision
        with torch.no_grad():
            batch_embeddings = encode_function(batch_data).cpu()  # Encode the entire batch

    end_time = time.time()
    
    # Update timing stats
    batch_time = end_time - start_time
 
    
    # Calculate and log throughput
    current_throughput = len(batch_data) / batch_time
    if throughput:
        logging.info(f"throughput: {current_throughput:.2f} samples/sec")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(batch_embeddings.half(), output_path)

def process_batch(dataset, domain, output_dir, index, encode_function, resume, throughput=False, source_caption=None):
    idx = 0
    desc = "Processing images" if domain == "image" else "Processing text"
    
    for batch in tqdm(dataset, desc=desc):
        output_path = os.path.join(output_dir, f'{index}_{idx}.pt')
        if resume and os.path.exists(output_path):
                idx += 1
                continue
        if domain == "text":
            assert source_caption is not None, "source_caption is required for text domain"
            batch_data = [json_data[source_caption] for json_data in batch[0]]
        else:  # image domain
            batch_data = batch[0]
        encode_save(batch_data, encode_function, output_path=output_path, throughput=throughput)
        idx += 1
            


@torch.no_grad()
def encode_data(args, dataset):
    index = args.data.split('/')[-1].split('.tar')[0]
    data_name = args.data.split('/')[-2]
    if args.domain == 'text':
        model_name = args.text_model_name.split('/')[-1]
        logging.info(f'Encoding text data {args.data} with model {args.text_model_name} of batch size {args.batch_size}...')
        if args.save_name:
            output_dir = os.path.join('./data/tensor_data/text_embedding', model_name, data_name +'_'+ args.source_caption + '_' + args.save_name)
        else:
            output_dir = os.path.join('./data/tensor_data/text_embedding', model_name, data_name +'_'+ args.source_caption)
        model = SentenceEmbedding(args.text_model_name).half().to('cuda')  # Move model to GPU and convert to FP16
        encode_fn = model.get_sentence_embeddings
        source_caption = args.source_caption
    else:  # image domain
        model_name = args.vision_model_name.split('/')[-1]
        logging.info(f'Encoding image data {args.data} with model {args.vision_model_name} of batch size {args.batch_size}...')
        output_dir = os.path.join('./data/tensor_data/image_embedding', model_name, data_name + '_' + args.agg_mode)
        model = ImageEmbedding(args.vision_model_name, agg_mode=args.agg_mode).half().to('cuda')  # Move model to GPU and convert to FP16
        encode_fn = model.get_visual_embeddings_from_list
        source_caption = None

    if not args.resume and os.path.exists(output_dir):
        logging.info(f'{output_dir} already exists, skipping...')
        exit()
    
    logging.info(f"output_dir: {output_dir}")
        
    model.eval()
    process_batch(dataset, args.domain, output_dir, index, encode_fn, args.resume, args.throughput, source_caption)


def main():
    args = parse_args()
    dataset = load_data(args)
    if args.throughput:
        logging.warning("Only measure throughput, not saving embeddings")
    # assert begin and end 
    
    encode_data(args, dataset)

if __name__ == "__main__":
    main()