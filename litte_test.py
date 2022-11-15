import torch

import torch.nn.functional as F

data = torch.tensor([
            [1., 2., 3., 4.],
            [ 2., 4., 6., 8.],
            [ 3., 6., 9., 12.]
        ])

normed = torch.norm(data, p=2, dim=-1, keepdim=True)

print(normed)

normed_data = data / normed 


print(normed_data)


normed_data2 = F.normalize(data, p=2, dim=-1)

print(normed_data2)

