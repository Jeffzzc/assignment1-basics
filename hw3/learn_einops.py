import torch
from einops import rearrange, einsum

A = torch.randn(3,4)
B = torch.randn(4,3)

C = A @ B

C_einsum = torch.einsum('ik,kj->ij', A, B)

print(C)
print(C_einsum)