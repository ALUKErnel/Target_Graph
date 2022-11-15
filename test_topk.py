import torch

# matrix = torch.tensor([[0.4,0.2,0.1,0.7],
#                        [0.3,0.2,0.9,1.0],
#                        [1.2,0.1,0,0],
#                        [0.7,1.4,0,0]])

# k = 3

# value, index = matrix.topk(3,dim=-1,largest=True, sorted=True)

# print(value, index)

# zero_tensor = torch.zeros([4,4])

# mask = zero_tensor.scatter_(-1, index, 1)
# print(mask)
# zero_tensor1 = torch.zeros([4,4])
# mask_e = torch.where(matrix > 0, mask, zero_tensor1)
# print(mask_e)

def get_target_relation(weight_matrix):
    '''
    从GAT得到的weight matrix转换为target relation matrix 
    暂时定为threshold 后续可以考虑 top K 等
    固定的threshold会导致 和 weight不吻合的问题  weight的变化范围很大 1 0.1 0.001 的情况都存在
    先选择 top K 试试看
    
    '''
    # target_relation_matrix = torch.where(weight_matrix>opt.weight_threshold, 1, 0)
    zero_tensor1 = torch.zeros([num_target, num_target])
    _, index = weight_matrix.topk(tk, dim=-1, largest=True, sorted=True)
    target_relation_matrix = zero_tensor1.scatter_(-1, index, 1)
    zero_tensor2 = torch.zeros([num_target, num_target])
    target_relation_matrix = torch.where(weight_matrix > 0, target_relation_matrix, zero_tensor2)
    
    return target_relation_matrix # [n x n]

num_target = 4
tk = 3
if __name__ == "__main__":
    weight_matrix = torch.tensor([[0.4,0.2,0.1,0.7],
                       [0.3,0.2,0.9,1.0],
                       [1.2,0.1,0,0],
                       [0.7,1.4,0,0]])
    mask = get_target_relation(weight_matrix)
    print(mask)
    