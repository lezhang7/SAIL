import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import glob
from natsort import natsorted
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def custom_collate_fn(batch):
    if len(batch[0]) == 3:
        text_vectors, image_vectors, extra_text_vectors = zip(*batch)
    else:
        text_vectors, image_vectors = zip(*batch)
        extra_text_vectors = None

    text_vectors = pad_sequence(text_vectors, batch_first=True, padding_value=0)
    image_vectors = pad_sequence(image_vectors, batch_first=True, padding_value=0)
    
    if extra_text_vectors:
        extra_text_vectors = pad_sequence(extra_text_vectors, batch_first=True, padding_value=0)
        return text_vectors, image_vectors, extra_text_vectors
    else:
        return text_vectors, image_vectors

def load_vectors(embedding_list: list[str]) -> list[torch.Tensor]:
    files = []
    for dir_path in embedding_list:
        files.extend(natsorted(glob.glob(os.path.join(dir_path, "*.pt"))))
    vectors = []
    for file in tqdm(files, desc="Loading embedding data", unit="file"):
        vectors.extend(torch.load(file, weights_only=True).to(torch.float16))
    return vectors

class VLEmbeddingDataset(Dataset):
    def __init__(self, text_embedding_list, image_embedding_list, extra_text_embedding_list=None, train_num_samples=None):

        self.text_vectors, self.image_vectors = self._load_image_text_vectors(image_embedding_list, text_embedding_list)
        n_img, n_txt = len(self.image_vectors), len(self.text_vectors)
        assert n_img > 0 and n_txt > 0 and n_txt % n_img == 0, f"text vectors length ({n_txt}) is not a multiple of image vectors length ({n_img})"

        if extra_text_embedding_list:
            print(f"Loading extra text vectors from {extra_text_embedding_list}")
            self.extra_text_vectors, _ = self._load_image_text_vectors(text_embedding_list = extra_text_embedding_list)
            assert len(self.extra_text_vectors) == len(self.text_vectors), f"extra text vectors {extra_text_embedding_list} length {len(self.extra_text_vectors)} is not equal to text vectors length {len(self.text_vectors)}"
    
        if train_num_samples is not None:
            num_samples = len(self.text_vectors)
            random_indices = np.random.choice(num_samples, train_num_samples, replace=False)
            self.text_vectors = [self.text_vectors[i] for i in random_indices]
            self.image_vectors = [self.image_vectors[i] for i in random_indices]
            if extra_text_embedding_list:
                self.extra_text_vectors = [self.extra_text_vectors[i] for i in random_indices]
            print(f"Random Selecting {train_num_samples} samples as training data")

        self.image_num = len(self.image_vectors)
        self.text_num = len(self.text_vectors)

        self.visual_dim = self.image_vectors[0].shape[-1] if len(self.image_vectors[0].shape) == 1 else self.image_vectors[0].shape[1]
        self.text_dim = self.text_vectors[0].shape[-1] if len(self.text_vectors[0].shape) == 1 else self.text_vectors[0].shape[1]
        
    def _load_image_text_vectors(self, image_embedding_list = None, text_embedding_list = None):
        assert image_embedding_list is not None or text_embedding_list is not None, "Either image_embedding_list or text_embedding_list must be provided"
        if image_embedding_list is not None:
            image_vectors = load_vectors(image_embedding_list)
        else:
            image_vectors = []
        if text_embedding_list is not None:
            text_vectors = load_vectors(text_embedding_list)
        else:
            text_vectors = []
        return text_vectors, image_vectors

    def __len__(self):
        return self.text_num
    
    def __getitem__(self, idx):
        # multiple text for one image
        if idx >= self.image_num:
            img_idx = idx % self.image_num
        else:
            img_idx = idx
        
        if hasattr(self, 'extra_text_vectors'):
            return self.text_vectors[idx], self.image_vectors[img_idx], self.extra_text_vectors[idx]
        else:
            return self.text_vectors[idx], self.image_vectors[img_idx]

if __name__ == "__main__":

    text_embedding_dir = ['/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/validation']
    image_embedding_dir = ['/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/validation']
    print("Loading dataset...")
    # dataset = LazyVLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    dataset = VLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    print("Dataset loaded.")
    # 创建DataLoader
    breakpoint()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    for batch in tqdm(dataloader):
        if len(batch) == 3:
            text_vectors, image_vectors, extra_text_vectors = batch
        else:
            text_vectors, image_vectors = batch
            extra_text_vectors = None
        print(text_vectors.shape, image_vectors.shape)

