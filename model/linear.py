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
        intermediate_dim: Optional[int] = None,
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.f1 = nn.Linear(input_dim, 2 * input_dim)
        self.f2 = nn.Linear(input_dim, 2 * input_dim)
        self.act = activation  # 传入的激活函数
        self.g = nn.Linear(2 * input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.f1(hidden_states), self.f2(hidden_states)
        x1 = torch.clamp(x1, min=-1e6, max=1e6)
        if self.act:
            x = self.act(x1) * x2
        else:
            x = x1 * x2
        x = self.g(x)

        assert not torch.isnan(x).any(), "Output contains NaN"
        assert not torch.isinf(x).any(), "Output contains infinite values"

        return x





# v2
class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear1(x) * F.silu(self.linear2(x))

class SiglipMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        intermediate_dim: Optional[int] = None,
    ):
        super().__init__()
        intermediate_dim = intermediate_dim if intermediate_dim is not None else output_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return  self.proj(hidden_states)

# v6 fast weight programming
# class StarMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         compress_dim: Optional[int] = 256,
#         intermediate_dim: Optional[int] = None,
#     ):
#         super().__init__()
#         intermediate_dim = intermediate_dim if intermediate_dim is not None else output_dim
#         self.compression = nn.Linear(input_dim, compress_dim)
#         self.Wa = nn.Linear(compress_dim, compress_dim, bias=False)
#         self.Wb = nn.Linear(compress_dim, compress_dim, bias=False)
#         self.g = nn.Linear(compress_dim, output_dim, bias=False)
#         self.act = nn.ReLU6()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.compression(x)
#         a = self.Wa(x)  # N x d
#         b = self.Wb(x)  # N x d
#         x = torch.einsum('bij,bj->bi', torch.sigmoid(a.unsqueeze(-1) * b.unsqueeze(1)), x)
#         x = self.g(self.act(x))

#         assert not torch.isnan(x).any(), "Output contains NaN"
#         assert not torch.isinf(x).any(), "Output contains infinite values"

#         return x