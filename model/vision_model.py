from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
import torch
import torch.nn as nn
from PIL import Image, ImageFile    
import os
from typing import List, Tuple, Dict, Any, Union, Optional
from torchvision import transforms as pth_transforms
from transformers import ImageProcessingMixin
try:
    from transformers.image_processing_base import BaseBatchFeature
except:
    pass
from packaging import version
import transformers
from .dinoforseg import Dinov2EncoderForSegmentation

from .mae import get_mae_vit
from .ibot import get_ibot_vit
from .ijepa import get_ijepa_vit
from .vq_model import get_vq_model

from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def modify_vit(type_: str) -> None:
    from transformers.models.dinov2 import modeling_dinov2
    if type_ == 'seg':
        print("Using Dinov2EncoderForSegmentation")
        modeling_dinov2.Dinov2Encoder = Dinov2EncoderForSegmentation
    else:
        pass

class CustomImageProcessor(ImageProcessingMixin):
    def __init__(self, size=224, is_vq=False):
        super().__init__()
        if is_vq:
            self.image_processor = pth_transforms.Compose([
                pth_transforms.Resize(256, interpolation=pth_transforms.InterpolationMode.BICUBIC),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.image_processor = pth_transforms.Compose([
                pth_transforms.Resize(256, interpolation=pth_transforms.InterpolationMode.BICUBIC),
                pth_transforms.CenterCrop(size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    def __call__(self, images, return_tensors=None):
        if not isinstance(images, list):
            images = [images]
        processed_images = [self.image_processor(image) for image in images]
        processed_images = torch.stack(processed_images)
        if return_tensors == 'pt':
            return processed_images
        else:
            return processed_images


class ImageEmbedding(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device=None, seg: bool = False, agg_mode='concat'):
        super(ImageEmbedding, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg = seg
        self.agg_mode = agg_mode
        self.model_name = model_name

        if any(x in model_name for x in ['ibot', 'mae', 'dinov1', 'ml-aim', 'ijepa', 'VQ']):
            # load from local
            if 'ibot' in model_name:
                self.model = get_ibot_vit(model_name)
            elif 'mae' in model_name:
                self.model = get_mae_vit(model_name)
            elif 'ijepa' in model_name:
                self.model = get_ijepa_vit(model_name)
            elif 'VQ' in model_name:
                self.model = get_vq_model(model_name)
            # load from torch hub
            elif 'dinov1' in model_name:
                if 'vitb16' in model_name:
                    self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
                    self.model.embed_dim = 768
                elif 'resnet' in model_name:
                    self.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
                    self.model.embed_dim = 2048
            elif 'ml-aim' in model_name:
                if '1B' in model_name:
                    self.model = torch.hub.load("apple/ml-aim", "aim_1B")
                    self.model.embed_dim = 2048
                elif '600M' in model_name:
                    self.model = torch.hub.load("apple/ml-aim", "aim_600M")
                    self.model.embed_dim = 1536
                else:
                    raise ValueError(f"Invalid model name: {model_name}")
            else:
                raise ValueError(f"Invalid model name: {model_name}")
            self.model = self.model.to(self.device).half()
            self.model.dtype = torch.float16
            self.image_processor = CustomImageProcessor(is_vq= 'VQ' in model_name)
        
        # load from huggingface
        elif not seg and any(x in model_name.lower() for x in ['dinov2']) and version.parse(transformers.__version__) >= version.parse("4.46.0"):
            # check if huggingface version is greater than 4.38.0
            print("Using model with SDPA")
            self.SDPA = True
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.float16)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        elif any(x in model_name.lower() for x in ['clip']):
            self.model = CLIPVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        elif any(x in model_name.lower() for x in ['aimv2']):
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            if seg:
                modify_vit('seg')
            self.SDPA = False
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.float16)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)


    def get_visual_embeddings_from_list(self, images: List[Image.Image]):
        images_tensor = self.image_processor(images, return_tensors="pt").to(self.device, dtype=self.model.dtype)
        with torch.autocast(device_type=self.device, dtype=self.model.dtype):
            with torch.no_grad():
                return self.forward(images_tensor)
    
    def get_visual_embeddings(self, images_tensor: List[torch.Tensor]):
        """
        Process pre-loaded image tensors (e.g., from WebDataset)
        
        Args:
            images_tensor: List of image tensors already loaded
            
        Returns:
            Embeddings from the vision model
        """
        # Convert list of tensors to batch if needed
        if isinstance(images_tensor, list):
            # Check if images need preprocessing
            if len(images_tensor[0].shape) == 3:  # If [C, H, W] format
                # Apply normalization if images are already in tensor format but not normalized
                normalized_images = []
                for img_tensor in images_tensor:
                    # Convert to float and normalize if needed
                    if img_tensor.max() > 1.0:  # Images in 0-255 range
                        img_tensor = img_tensor.float() / 255.0
                    
                    # Apply normalization with ImageNet stats
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
                    normalized_images.append((img_tensor - mean) / std)
                
                # Stack into batch
                inputs = torch.stack(normalized_images).to(self.device, dtype=self.model.dtype)
            else:
                # If already batched or in proper format
                inputs = torch.stack(images_tensor).to(self.device, dtype=self.model.dtype)
        else:
            # If already a batched tensor
            inputs = images_tensor.to(self.device, dtype=self.model.dtype)
            
        with torch.autocast(device_type=self.device, dtype=self.model.dtype):
            with torch.no_grad():
                return self.forward(inputs)

    def forward(self, inputs, patch_mode=False, attetion_type='qk', ignore_residual=False):
        """
        Get embeddings from the vision encoder.

        Args:
            nocls: if True, return mean of patch tokens, else concat CLS token and mean of patch tokens.
            patch_mode: if True, return all patch tokens.
            ignore_residual: if True, skip the residual connection in the last layer.
        """
        if self.seg:
            self.model.encoder.attetion_type = attetion_type
            self.model.encoder.ignore_residual = ignore_residual
                
        if any(x in self.model_name.lower() for x in ['mae']):
            if isinstance(inputs, torch.Tensor):
                outputs = self.model.forward_features(inputs)
            else:
                outputs = self.model.forward_features(inputs['pixel_values'])
        else:
            if isinstance(inputs, torch.Tensor):
                if any(x in self.model_name.lower() for x in ['vq']):
                    outputs = self.model.encode(inputs)
                    
                else:
                    outputs = self.model(inputs)
            elif isinstance(inputs, dict) or isinstance(inputs, BaseBatchFeature):
                if any(x in self.model_name.lower() for x in ['ibot', 'dinov1', 'ml-aim', 'ijepa']):
                    outputs = self.model(inputs['pixel_values'])
                else:
                    # huggingface transformer vision model
                    outputs = self.model(**inputs)
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
            

        # extract the embeddings
        if any(x in self.model_name.lower() for x in ['ijepa']):
            embedding = outputs.mean(dim=1)
        elif any(x in self.model_name.lower() for x in ['vq']):
            embedding = outputs[0]
        elif any(x in self.model_name.lower() for x in ['clip']):
            embedding = outputs.pooler_output
        elif any(x in self.model_name.lower() for x in ['aimv2']):
            embedding = torch.mean(outputs.last_hidden_state, dim=1)
        else:
            sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

            if patch_mode:
                patch_tokens = sequence_output[:, 1:]
                cls_token = sequence_output[:, 0].unsqueeze(1).repeat(1, patch_tokens.shape[1], 1)
                embedding = torch.cat([cls_token, patch_tokens], dim=-1)
            else:
                if any(x in self.model_name.lower() for x in ['ibot', 'mae', 'dinov1']):
                    embedding = outputs
                elif any(x in self.model_name.lower() for x in ['aim']):
                    embedding = outputs[1]
                else:
                    cls_token = sequence_output[:, 0]
                    patch_tokens = sequence_output[:, 1:]
                    if self.agg_mode == 'patch': 
                        embedding = patch_tokens.mean(dim=1)
                    elif self.agg_mode == 'cls':
                        embedding = cls_token
                    elif self.agg_mode == 'concat':
                        embedding = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
                    else:
                        raise ValueError(f"Invalid agg_mode: {self.agg_mode}")

        return embedding


if __name__ == "__main__":
    processor = ImageEmbedding()
    image_dirs = ["path/to/dir1", "path/to/dir2"]
    hidden_states = processor.get_visual_embeddings_from_list(image_dirs)
    for states in hidden_states:
        print(list(states.shape))
    breakpoint()
