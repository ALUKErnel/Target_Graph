import torch

# adj = torch.tensor([1.0,0.0])
# adj1 = torch.tensor([1,0])
# attn= torch.tensor([3.2,4.4])
# attn.masked_fill_(1-adj1, -999)
# print(attn)


a = torch.tensor([[1,0],[1,1]])
b = torch.ones(2,2) - a
print(b)