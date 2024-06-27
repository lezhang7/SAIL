import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class SentenceEmbedding(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', device=None):
        super(SentenceEmbedding, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_sentence_embeddings(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        return self.forward(encoded_input)
    
    def forward(self, inputs):
        # Compute token embeddings
        model_output = self.model(**inputs)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
      
        return sentence_embeddings

# Usage example
if __name__ == "__main__":
    sentences = ['This is an example sentence', 'Each sentence is converted']
    
    embedder = SentenceEmbedding()
    sentence_embeddings = embedder.get_sentence_embeddings(sentences)
    
    print("Sentence embeddings:")
    print(sentence_embeddings)
    print(sentence_embeddings.shape)
