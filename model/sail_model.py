from transformers import AutoImageProcessor, Dinov2Model
import torch
from PIL import Image
import os
from typing import List, Tuple, Dict, Any, Union, Optional
import torch.nn as nn
from .language_model import SentenceEmbedding
from .vision_model import ImageEmbedding
import torch.nn.functional as F
import torch
from transformers.activations import ACT2FN
from .linear import StarMLP, ShareLockMLP, VQDenseNet
from functools import partial
import numpy as np


class AlignmentLayer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        linear_type: str = "star",
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = 10.0,
        logit_bias: float = -10.0,
        only_vision: bool = False,
        width_factor: int = 8,
    ):
        super(AlignmentLayer, self).__init__()
        assert only_vision or text_dim is not None, "text_dim must be provided if only_vision is False"

        self.cast_dtype = cast_dtype
        self.linear_type = linear_type
        
        # Set up vision mapping network
        if linear_type == "vqdensify":
            self.vision_mapping_network = VQDenseNet(hidden_dim,hidden_dim)
        else:
            if linear_type == "star":
                VisionLayer = partial(StarMLP, width_factor=width_factor, activation=nn.ReLU6())
            else:
                VisionLayer = nn.Linear
            self.vision_mapping_network = VisionLayer(vision_dim, hidden_dim)

        self.vision_pre_layernorm = nn.LayerNorm(vision_dim)
        self.vision_post_layernorm = nn.LayerNorm(hidden_dim)
        self.vision_output_proj = nn.Linear(hidden_dim, output_dim)

        # Text mapping network always uses StarMLP if vqdensify, otherwise follows linear_type
        if not only_vision:
            if linear_type == "vqdensify":
                self.text_mapping_network = StarMLP(
                    text_dim, 
                    hidden_dim,
                    width_factor=width_factor,
                    activation=nn.ReLU6()
                )
            else:
                self.text_mapping_network = self.vision_mapping_network.__class__(
                    text_dim, output_dim
                )
            self.text_pre_layernorm = nn.LayerNorm(text_dim)
            self.text_post_layernorm = nn.LayerNorm(hidden_dim)
            self.text_output_proj = nn.Linear(hidden_dim, output_dim)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self._initialize_weights(logit_scale, logit_bias)

    def _initialize_weights(self, scale: float, bias: float):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Initialize logit_scale and logit_bias
        logit_scale_init = torch.log(torch.tensor(scale))                           
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(torch.tensor(bias))
    
    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()
    
    @property
    def get_logit_bias(self):
        return self.logit_bias
     
    def forward(self, image_features=None, text_features=None, extra_text_features=None, compute_logits=False):

        if image_features is None and text_features is None:
            raise ValueError(
                "At least one of image_features and text_features should be provided."
            )
        if image_features is not None:
            image_features = image_features.to(self.cast_dtype)
            image_features = self.vision_pre_layernorm(image_features)
            image_features = self.vision_mapping_network(image_features)
            image_features = self.vision_post_layernorm(image_features)
            image_features = self.vision_output_proj(image_features)
        else:
            image_features = None

        if text_features is not None:
            text_features = text_features.to(self.cast_dtype)
            text_features = self.text_pre_layernorm(text_features)
            text_features = self.text_mapping_network(text_features)
            text_features = self.text_post_layernorm(text_features)
            text_features = self.text_output_proj(text_features)
        else:
            text_features = None

        if extra_text_features is not None:
            extra_text_features = extra_text_features.to(self.cast_dtype) 
            extra_text_features = self.text_pre_layernorm(extra_text_features)
            extra_text_features = self.text_mapping_network(extra_text_features)
            extra_text_features = self.text_post_layernorm(extra_text_features)
            extra_text_features = self.text_output_proj(extra_text_features)
        else: 
            extra_text_features = None

        if compute_logits and image_features is not None and text_features is not None and image_features.nelement() > 0 and text_features.nelement() > 0:
            logits_per_text = (
                torch.matmul(
                    F.normalize(text_features, dim=-1),
                    F.normalize(image_features, dim=-1).t(),
                )
                * self.logit_scale.exp()
                + self.logit_bias
            )
        else:
            logits_per_text = None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "extra_text_features": extra_text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.logit_scale.exp(),
            "logit_bias": self.logit_bias
        }

class ShareLockAlignmentLayer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        text_dim: int = None,
        output_dim: int = None,
        linear_type: str = None,
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = np.log(1 / 0.07),
        logit_bias: float = -10.0,
        *args,
        **kwargs,
    ):
        super(ShareLockAlignmentLayer, self).__init__()
        self.cast_dtype = cast_dtype
        # no vision alignment layer overhead
        output_dim = vision_dim 
        hidden_dim = 4096
        self.text_mapping_network = ShareLockMLP(
            text_dim, hidden_dim, output_dim
        )
     
        self.text_pre_layernorm = nn.LayerNorm(text_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)
        self._initialize_weights(logit_scale, logit_bias)

    def _initialize_weights(self, scale: float, bias: float):
    
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)                     

    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()
    
    @property
    def get_logit_bias(self):
        return self.logit_bias

    def forward(self, image_features=None, text_features=None, extra_text_features=None, compute_logits=False):
    
        if image_features is None and text_features is None:
            raise ValueError(
                "At least one of image_features and text_features should be provided."
            )

        if text_features is not None:
            text_features = text_features.to(self.cast_dtype)
            text_features = self.text_pre_layernorm(text_features)
            text_features = self.text_mapping_network(text_features)
        else:
            text_features = None

        if compute_logits and image_features is not None and text_features is not None and image_features.nelement() > 0 and text_features.nelement() > 0:
            logits_per_text = (
                torch.matmul(
                    F.normalize(text_features, dim=-1),
                    F.normalize(image_features, dim=-1).t(),
                )
                * self.logit_scale.exp()
                + self.logit_bias
            )
        else:
            logits_per_text = None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.get_logit_scale,
            "logit_bias": self.get_logit_bias
        }


        

class SAILModel(nn.Module):
    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        output_dim: int,
        vlhead_weights_path: str = None,
        linear_type: str = "linear",
        cast_dtype: Optional[torch.dtype] = None,
        seg: bool = False,
        agg_mode: str = 'concat',
        width_factor: int = 8,
        sharelock: bool = False,
    ):
        super(SAILModel, self).__init__()
        self.text_model = SentenceEmbedding(text_model_name)
        self.vision_model = ImageEmbedding(vision_model_name, seg=seg, agg_mode=agg_mode)
        if any(x in vision_model_name for x in ['mae','ibot','dinov1','ml-aim','ijepa','clip','aimv2']) or 'patch' in agg_mode or 'cls' in agg_mode:
            if hasattr(self.vision_model.model, 'config'):
                vision_dim = self.vision_model.model.config.hidden_size
            else:
                vision_dim = self.vision_model.model.embed_dim
        else:
            vision_dim = self.vision_model.model.config.hidden_size * 2
        LayerClass = ShareLockAlignmentLayer if sharelock else AlignmentLayer
        self.vlhead = LayerClass(
            vision_dim = vision_dim,
            text_dim = self.text_model.model.config.hidden_size,
            output_dim = output_dim,
            linear_type = linear_type,
            cast_dtype = cast_dtype,
            width_factor = width_factor,
        )  
        if vlhead_weights_path is not None:
            self.load_vlhead_weights(  )

    def freeze_except_vlhead(self):
        self._freeze_parameters(self.vision_model)
        self._freeze_parameters(self.text_model)
        self._unfreeze_parameters(self.vlhead)

    def _freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def load_vlhead_weights(self, vlhead_weights_path):
        weights = torch.load(vlhead_weights_path)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        msg = self.vlhead.load_state_dict(weights, strict=False)
        print(f"Loaded VL head weights from {vlhead_weights_path}, {msg}")

    @property
    def image_processor(self):
        return self.vision_model.image_processor

    @torch.no_grad()
    def encode_image(
        self,
        image,
        normalize: bool = False,
        is_pre_encoded: bool = False,
        return_encoded: bool = False,
        patch_mode: bool = False,
        attetion_type: str = 'qk',
        ignore_residual: bool = False,
    ):
        if is_pre_encoded:
            features = image
        else:
            features = self.vision_model(image, patch_mode=patch_mode, attetion_type=attetion_type, ignore_residual=ignore_residual)
        outputs = self.vlhead(image_features=features)
        image_features = outputs["image_features"]
        if return_encoded:
            return (
                F.normalize(image_features, dim=-1) if normalize else image_features
            ), features
        else:
            return F.normalize(image_features, dim=-1) if normalize else image_features

    @torch.no_grad()
    def encode_text(
        self,
        text,
        text_list: List[str] = None,
        normalize: bool = False,
        is_pre_encoded: bool = False,
        return_encoded: bool = False,
    ):
        if is_pre_encoded:
            features = text
        else:
            if isinstance(text, str):
                features = self.text_model.get_sentence_embeddings([text])
            elif hasattr(self.text_model.model, 'config') and 'NV' in self.text_model.model.config.name_or_path:
                features = self.text_model.model.encode(text_list, max_length=1024).half()
            elif 'clip' in self.text_model.model_name:
                features = self.text_model.get_sentence_embeddings(text_list)
            else:
                features = self.text_model(text)
                
        outputs = self.vlhead(text_features=features)
        text_features = outputs["text_features"]
        if return_encoded:
            return (
                F.normalize(text_features, dim=-1) if normalize else text_features
            ), features
        else:
            return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        images, 
        texts, 
        text_list: List[str] = None,
        is_pre_encoded: bool = False,
        return_encoded: bool = False,
    ):
        if is_pre_encoded:
            # if its pre-encoded, we do not need to return encoded features
            norm_image_features = self.encode_image(
                images,
                normalize=True,
                is_pre_encoded=True,
                return_encoded=False,
            )
            norm_text_features = self.encode_text(
                texts,
                normalize=True,
                is_pre_encoded=True,
                return_encoded=False,
            )
        elif return_encoded:
            # if we need to return encoded features, then it is not pre-encoded
            norm_image_features, encoded_image_features = self.encode_image(
                images,
                normalize=True,
                is_pre_encoded=False,
                return_encoded=True,
            )
            norm_text_features, encoded_text_features = self.encode_text(
                texts,
                text_list=text_list,
                normalize=True,
                is_pre_encoded=False,
                return_encoded=True,
            )
        else:
            # if we do not need to return encoded features and it is not pre-encoded, usuallu during training
            norm_image_features = self.encode_image(images, normalize=True)
            norm_text_features = self.encode_text(texts, text_list=text_list, normalize=True)

        # Log the sizes of embeddings
        logits_per_text = (
            torch.matmul(norm_text_features, norm_image_features.t())
            * self.vlhead.logit_scale.exp()
            + self.vlhead.logit_bias
        )
        if return_encoded:
            return {
                "image_features": norm_image_features,
                "text_features": norm_text_features,
                "logits_per_text": logits_per_text,
                "logit_scale": self.vlhead.get_logit_scale,
                "logit_bias": self.vlhead.get_logit_bias,
                "encoded_image_features": encoded_image_features,
                "encoded_text_features": encoded_text_features,
            }
        else:
            return {
                "image_features": norm_image_features,
                "text_features": norm_text_features,
                "logits_per_text": logits_per_text,
                "logit_scale": self.vlhead.get_logit_scale,
                "logit_bias": self.vlhead.get_logit_bias,
            }

if __name__ == "__main__":
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count
    from thop import profile
    # 初始化模型
    model = AlignmentLayer(
        vision_dim=2048,
        text_dim=1024,  
        output_dim=1024, 
        linear_type="raw"    
    )
    
    print(model)
    # 假设输入图像和文本特征
    image_features = torch.randn(1, 2048)  
    text_features = torch.randn(1, 1024)   


    # output = model(image_features=image_features, text_features=text_features)

    flops, params = profile(model, inputs=(image_features,text_features))

    print(f"FLOPs: {flops/1e6}")
    print(f"Params: {params/1e6}")
    
