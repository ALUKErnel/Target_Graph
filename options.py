import argparse
import torch
import os
parser = argparse.ArgumentParser()

# path
parser.add_argument('--data_dir', default='./dataset/')
parser.add_argument('--model_save_file', default='./save/1114_GAT1_tk10_mean_adjust_symm_force_contargetlang_1')

# random seed 
parser.add_argument('--random_seed', type=int, default=1)

# data readin
parser.add_argument('--num_train_lines', type=int, default=0)  # set to 0 to use all training data
parser.add_argument('--max_seq_len', type=int, default=1000) # Dataset readin (settle)
parser.add_argument('--num_target', type=int, default=0) # set 0 to select all targets

# preprocessing
parser.add_argument('--sim_threshold', type=float, default=0.4)
parser.add_argument('--measurement', type=str, default='cosine similarity')

# X EmbeddingModule
parser.add_argument('--tokenized_max_len', type=int, default=120) # Dataset readin 
parser.add_argument('--emb_size', type=int, default=768)

# G GAT 
parser.add_argument('--gnn_dims', type=str, default='192')
parser.add_argument('--att_heads', type=str, default='4')
parser.add_argument('--attn_dropout', type=float, default=0.2)
parser.add_argument('--concat_dropout', type=float, default=0.2)
parser.add_argument('--leaky_alpha', type=float, default=0.2)

# weight -> target_relation_matrix
parser.add_argument('--weight_threshold', type=float, default=0.3)
parser.add_argument('--tk', type=int, default=10) 

# P StanceClassifier
parser.add_argument('--P_layers', type=int, default=2)
parser.add_argument('--P_bn', default=True)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.2)

# D Domain Classifier
parser.add_argument('--D_layers', type=int, default=2)
parser.add_argument('--D_bn', default=True)

# concat for stance/domain classifier
parser.add_argument('--concat_stance', default=True)
parser.add_argument('--concat_domain', default=False)

# contrastive criterion
parser.add_argument('--temperature', type=float, default=0.3)

# loss balancing
parser.add_argument('--beta_l', type=float, default=0.7)
parser.add_argument('--beta_t', type=float, default=0.3)
parser.add_argument('--alpha', type=float, default=0.8)

# training 
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_size_target', type=int, default=100)
parser.add_argument('--max_epoch', type=int, default=15)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--local_rank', type=int, default=0)

opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'

