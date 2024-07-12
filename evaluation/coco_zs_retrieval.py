import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import clip
from tqdm import tqdm
import torch.nn as nn
from .utils import get_model_device

    
def coco_collate_fn(batch):
    text_list = []
    image_list = []
    for item in batch:
        image, text = item
        text_list.append(text)
        image_list.append(image)
    # print(image_list)
    images = torch.cat(image_list)
    # print(images.shape)
    images = {'pixel_values': images}
    return text_list, images

# Encodes all text and images in a dataset
def encode_dataset(model, dataset: dutils.Dataset, device, batch_size = 16):
    with torch.no_grad():
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        dataloader = dutils.DataLoader(dataset, collate_fn=coco_collate_fn, batch_size=batch_size, shuffle=False)

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0
        captions_per_image = 5

        for text, images in tqdm(dataloader):
            images = {key: value.to(device) for key, value in images.items()} # B x 3 x 224 x 224
            batch_size = len(text)
            text_list = []
            for i in text:
                text_list.extend(i[:captions_per_image])
            text_tokens = model.text_model.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device) # (B*5) x 77
            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image
                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1
            image_encodings.append(model.encode_image(images))
            text_encodings.append(model.encode_text(text_tokens))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        # Normalise encodings
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def recall_at_k(clip, dataset: dutils.Dataset, device, k_vals: List[int], batch_size: int):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(clip, dataset, device, batch_size=batch_size)
 
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)


    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    print("Done.")
    return text_to_image_recall, image_to_text_recall

class Processor():
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, images):
        images = self.processor(images, return_tensors="pt")['pixel_values']
        return images

def coco_eval(
        model: nn.Module,
        bs: int = 1024, 
        coco_root: str = "/home/mila/l/le.zhang/scratch/datasets",
        coco_ann_file: str = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/annotations/captions_val2017.json",
        k_vals: List[int] = [1, 5, 10]
):
    model.eval()
    device = get_model_device(model)
    processor = Processor(model.vision_model.image_processor)
    dataset = CocoCaptions(
        root=coco_root,
        annFile=coco_ann_file,
        transform=processor
        # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
    )
    t2i, i2t = recall_at_k(model, dataset, device, k_vals=k_vals, batch_size=bs)
    result_dict = {}
    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")
        result_dict[f"T2I R@{k}"] = x

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")
        result_dict[f"I2T R@{k}"] = x
    

    return result_dict

if __name__ == "__main__":
        # Change these to path of local COCO dataset:
    linear = True
    coco_root = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/val2017"
    coco_ann_file = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/annotations/captions_val2017.json"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if linear:
        model = VLContrastModel(text_model_name='sentence-transformers/all-mpnet-base-v2', vision_model_name='facebook/dinov2-base', device=device, linear=True)
        weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data_only_linear_head/checkpoint_52.pth'
    else:
        model = VLContrastModel(text_model_name='sentence-transformers/all-mpnet-base-v2', vision_model_name='facebook/dinov2-base', device=device, linear=False)
        weights_path='/home/mila/l/le.zhang/scratch/light_align/output/raw_data_linear_shared_head/checkpoint_72.pth'
    checkpoint = torch.load(weights_path)
    model.vlhead.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    processor = Processor(model.vision_model.image_processor)
    dataset = CocoCaptions(
        root=coco_root,
        annFile=coco_ann_file,
        transform=processor
        # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
    )

    k_vals=[1, 5, 10, 50]

    t2i, i2t = recall_at_k(model, dataset, k_vals=k_vals, batch_size=512)

    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")

