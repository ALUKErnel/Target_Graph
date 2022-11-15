from models import *
import torch
import time
from sklearn.metrics import f1_score, confusion_matrix, classification_report
if __name__ == "__main__":
    # adj = torch.tensor([[0,1,1,1],
    #                     [4,0,1,0],
    #                     [1,1,0,0],
    #                     [1,0,0,0]])
    
    adj = torch.ones([100,100])

    # features = torch.ones([4,40])
    
    
    # emb_size = 40
    # gnn_dims = "10,10"
    # att_heads = "4,4" 
    # attn_dropout = 0.2
    # concat_dropout = 0.2
    # leaky_alpha = 0.3
    # G = GAT(emb_size, gnn_dims, att_heads, attn_dropout, concat_dropout, leaky_alpha) 
    
    # node_features, weight = G(features, adj)
    # print(node_features)
    # print(weight)
    
    # t1 = torch.tensor([0,1,0,0])
    # print(t1)
    # t2 = torch.tensor([1,0,0,0])
    # middle = torch.matmul(t1.T, adj)
    # final = torch.matmul(middle, t2)
    # print(final)  
    
    # N = 300
    # target_label_1 = torch.tensor([1,0,3,1,3,4,5,6,1,1])
    # target_label = torch.cat([target_label_1]*30, dim=-1).reshape(N,-1)
    
    # print(target_label.shape)
    
    # e11 = torch.cat([target_label.repeat(1,N).view(N*N, 1), target_label.repeat(N,1)],dim=1).view(N,N,-1)
    # # print(e11)
    # print(e11.shape)
    
    # # eq = torch.tensor([[[1,1],[1,0],[1,3]],
    # #                  [[0,1],[0,0],[0,3]],
    # #                  [[3,1],[3,0],[3,3]]])

    # # print(eq.shape)    
    # mask = torch.zeros([N,N])
    
    # start_time = time.time()
    
    
    # for i in range(N):
    #     for j in range(N):
    #         index_pair = e11[i][j]
    #         mask[i][j] = adj[index_pair[0], index_pair[1]]
            
    # end_time = time.time()
            
    # print("duration is: {} ".format(end_time - start_time))
    
    # print(mask)
    
    # labels = torch.tensor([1,0,2]).reshape(-1,1)
    # one_hot = torch.zeros([3,4]).scatter_(1,labels, 1)
    # print(one_hot)
    
    
    # index = torch.tensor([1,0,2])
    # dic = torch.tensor([[0,0],[1,1],[2,2]])
    # print(dic[index])
    
    y_true = torch.tensor([1,0,1,1,1,1,0]).reshape(-1,1)
    y_pred = torch.tensor([1,1,1,1,0,1,0]).reshape(-1,1)
    f1_1 = f1_score(y_true == 1, y_pred == 1, labels=True)
    f1_0 = f1_score(y_true == 0, y_pred == 0, labels=True)
    print(f1_1, f1_0)
    print("classification report: \n", classification_report(y_true, y_pred, digits=4))
    
    
    
    
    