import torch
from torch import nn
from typing import Optional, Callable
import torch.nn.functional as F

# v7  
class StarMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width_factor: int = 4,
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.f1 = nn.Linear(input_dim, width_factor * input_dim)
        self.f2 = nn.Linear(input_dim, width_factor * input_dim)
        self.act = activation  # 传入的激活函数
        self.g = nn.Linear(width_factor * input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.f1(hidden_states), self.f2(hidden_states)
        x1 = torch.clamp(x1, min=-1e3, max=1e3)
        x2 = torch.clamp(x2, min=-1e3, max=1e3)
        if self.act:
            x = self.act(x1) * x2
        else:
            x = x1 * x2
        x = self.g(x)

        assert not torch.isnan(x).any(), "Output contains NaN"
        assert not torch.isinf(x).any(), "Output contains infinite values"

        return x


class ShareLockMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super(ShareLockMLP, self).__init__()
        
        # Define the layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class VQDenseNetLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VQDenseNetLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.mlp = StarMLP(input_dim, output_dim)

    def forward(self, x, residual=False):
        if residual:
            x = x + self.mlp(self.layer_norm(x))
        else:
            x = self.mlp(self.layer_norm(x))
        return x


class VQDenseNet(nn.Module):
    """
    Network that processes 8x16x16 feature maps with position embeddings and GLU-based MLP
    """
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input dimensions: 8 channels, 16x16 spatial
        self.num_channels = 8
        self.spatial_size = 16
        self.input_size = self.num_channels * self.spatial_size * self.spatial_size  # 8 * 16 * 16 = 2048
        
        # Learnable position embeddings for each 16x16 feature map
        self.position_embeddings = nn.Parameter(
            torch.randn(self.num_channels, self.spatial_size, self.spatial_size) * 0.02
        )
        
        self.layers = nn.ModuleList()
        self.layers.append(VQDenseNetLayer(self.input_size, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(VQDenseNetLayer(hidden_dim, hidden_dim))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: Tensor of shape (batch_size, 8, 16, 16)
        
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        batch_size = feature_map.size(0)
        
        # Add position embeddings to each 16x16 feature map
        # feature_map: (B, 8, 16, 16), position_embeddings: (8, 16, 16)
        feature_map = feature_map + self.position_embeddings.unsqueeze(0)
        
        # Linearize (flatten) the feature map
        hidden_states = feature_map.view(batch_size, -1)  # (B, 8*16*16)
        
        # Pass through GLU-based MLP layers with residual connections

        for i, layer in enumerate(self.layers):
            if i == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(hidden_states, residual=True)
        
        return hidden_states
