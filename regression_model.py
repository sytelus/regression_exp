from torch import nn
import torch

class RegressionModel(nn.Module):
  def __init__(self, model_size:int):
    super().__init__()
    self.model_size = model_size
    self.A = nn.Parameter(torch.randn(1,model_size))

  def forward(self, x):
    return self.A.mm(x.t()).t()