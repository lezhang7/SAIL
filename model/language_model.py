import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPTextModel
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional, Union


def get_embedding_strategy(model_name):
    for strategy, models in MODEL_EMBEDDING_TYPE.items():
        if model_name in models:
            return strategy
    raise ValueError(f"No embedding strategy found for model: {model_name}")

def last_token_pool(model_output: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = model_output.last_hidden_state
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pooling(model_output: torch.Tensor, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def bos_pooling(model_output: torch.Tensor, attention_mask):
    return model_output.last_hidden_state[:,0]

MODEL_EMBEDDING_TYPE={
    'mean': ['sentence-transformers/all-mpnet-base-v2'],
    'BOS' : ['Alibaba-NLP/gte-large-en-v1.5','Alibaba-NLP/gte-base-en-v1.5'],
    "LastToken": ['Alibaba-NLP/gte-Qwen2-7B-instruct',"Alibaba-NLP/gte-Qwen2-1.5B-instruct","Qwen/Qwen3-Embedding-0.6B"],
}

POOL_CLASSES = {
    'mean': mean_pooling,
    'BOS' : bos_pooling,
    'LastToken' : last_token_pool,
}   

class SentenceEmbedding(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        super(SentenceEmbedding, self).__init__()
        self.model_name = model_name
        if not any(x in model_name for x in ['NV', 'clip']):
            self.pooling = POOL_CLASSES[get_embedding_strategy(model_name)]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'clip' in model_name:
            self.model = CLIPTextModel.from_pretrained(model_name, device_map=self.device).half()
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=self.device).half()
    
    def mean_pooling(self, model_output: torch.Tensor, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_sentence_embeddings(self, sentences: List[str]):
        # Tokenize sentences
        if 'NV' in self.model_name: 
            with torch.autocast(device_type=self.device.type, dtype=self.model.dtype):  # or bfloat16
                embeddings = self.model.encode(sentences, max_length=1024).half()
                return embeddings
        elif 'clip' in self.model_name:
            with torch.autocast(device_type=self.device.type, dtype=self.model.dtype):  # or bfloat16
                inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
                return embeddings
        else:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(self.device)
            return self.forward(encoded_input)
        
    def forward(self, inputs):
        # Compute token embeddings
        if 'clip' in self.model_name:
            with torch.autocast(device_type=self.device.type, dtype=self.model.dtype):  # or bfloat16
                model_output = self.model(**inputs)
            sentence_embeddings = model_output.pooler_output
        else:
            with torch.autocast(device_type=self.device.type, dtype=self.model.dtype):  # or bfloat16
                model_output = self.model(**inputs)
            sentence_embeddings = self.pooling(model_output, inputs['attention_mask'])
        return sentence_embeddings

# Usage example
if __name__ == "__main__":
    sentences = ['This is an example sentence', 'Each sentence is converted']
    
    embedder = SentenceEmbedding()
    sentence_embeddings = embedder.get_sentence_embeddings(sentences)
    
    print("Sentence embeddings:")
    print(sentence_embeddings)
    print(sentence_embeddings.shape)
