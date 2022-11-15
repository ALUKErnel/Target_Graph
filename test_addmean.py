import torch

t1 = torch.tensor([[1,2,1],[3,4,4]])
t2 = torch.tensor([[5,7,2],[1,10,20]])

t3 = (t1 + t2)/2

print(t3)
