
import math
import numpy as np
import torch
import torch.nn as nn

from torch_scatter import scatter

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

def subtract_cog(x, num_atoms):
    batch = torch.arange(num_atoms.size(0), device=num_atoms.device
                         ).repeat_interleave(num_atoms, dim=0)
    cog = scatter(x, batch, dim=0, reduce='mean'
                  ).repeat_interleave(num_atoms, dim=0)
    return x - cog
        
class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)