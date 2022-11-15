
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel, TrOCRConfig
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import random
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
from tqdm import tqdm 
import logging

from data_prep.xstance_dataset_domain import get_datasets_main, get_datasets_target
# from data_prep.xstance_dataset import get_datasets_target

from models import *
from criterions import *
from options import opt
from utils2 import * 



from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
# torch.distributed.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# print("local_rank: ",opt.local_rank)
# torch.cuda.set_device(opt.local_rank)
# opt.device = torch.device("cuda", opt.local_rank)

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)


if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, '1031_train_1000_b32.txt'))
log.addHandler(fh)


log.info('Fine-tuning mBERT with options:')
log.info(opt)

    
# opt.batch_size_target 
def get_target_feature(datasets_target_list, all_targets_list, model_target, tokenizer):
    all_target_feature = torch.zeros([len(all_targets_list), opt.emb_size]).to(opt.device)     
    for ti in range(len(all_targets_list)):
        log.info(f"{ti}th target dataset begin encode")
        curr_dataset = datasets_target_list[ti]
        data_loader = DataLoader(curr_dataset, opt.batch_size_target, shuffle=False)
        cls_list = []
        model_target.eval()
        iter1 = iter(data_loader)
        max_iter1 = len(data_loader) // opt.batch_size_target 
        for i, (inputs, y) in tqdm(enumerate(iter1), total=max_iter1):
            model_target.zero_grad()
            tokenized_inputs = tokenizer(inputs[1], padding=True, truncation=True, max_length=200, return_tensors="pt").to(opt.device)
            outputs = model_target(**tokenized_inputs)
            output_cls = outputs.pooler_output
            
            cls_list.append(output_cls)
        
        cls_list_tensor = torch.vstack(cls_list)
        cls_mean = torch.mean(cls_list_tensor, dim=0)
        all_target_feature[ti] = cls_list_tensor[0]
        
    
    return all_target_feature   # n x emb_size


# opt.sim_threshold
# opt.measurement 
def calculate_similarity(all_target_feature, measurement):
    mask = (torch.ones([opt.num_target, opt.num_target])- torch.eye(opt.num_target)).to(opt.device)
    if measurement == "dot product":
        similarity_matrix = torch.matmul(all_target_feature, all_target_feature.T) 
        similarity_matrix = similarity_matrix * mask
        
        
    elif measurement == "cosine similarity":
        norm_all_target_feature = all_target_feature / torch.norm(all_target_feature, p=2, dim=-1, keepdim=True)
        similarity_matrix = torch.matmul(norm_all_target_feature, norm_all_target_feature.T) 
        similarity_matrix = similarity_matrix * mask
    
    elif measurement == "fully-connected":
        similarity_matrix = mask 
        
    else:
        print("Error measurement in calculating similarity!")
        similarity_matrix = mask 
    
    return similarity_matrix
        
def calculate_adj(similarity_matrix, threshold):
    adj = torch.where(similarity_matrix>threshold, 1, 0)
    
    return adj 

# opt.weight_threshold
def get_target_relation(weight_matrix):
    '''
    从GAT得到的weight matrix转换为target relation matrix 
    暂时定为threshold 后续可以考虑 top K 等
    # target_relation_matrix = torch.where(weight_matrix>opt.weight_threshold, 1, 0)
    固定的threshold会导致 和 weight不吻合的问题  weight的变化范围很大 1 0.1 0.001 的情况都存在
    先选择 top K 试试看
    
    '''
    zero_tensor1 = torch.zeros([opt.num_target, opt.num_target]).to(opt.device)
    _, index = weight_matrix.topk(opt.tk, dim=-1, largest=True, sorted=True)
    target_relation_matrix = zero_tensor1.scatter_(-1, index, 1)
    zero_tensor2 = torch.zeros([opt.num_target, opt.num_target]).to(opt.device)
    target_relation_matrix = torch.where(weight_matrix > 0, target_relation_matrix, zero_tensor2)
    
    return target_relation_matrix # [n x n]

def get_target_mask_batch(target_relation_matrix, target_label):
    batch_size = target_label.shape[0]
    target_label = target_label.reshape(-1, 1) 
    index_matrix = torch.cat([target_label.repeat(1,batch_size).view(batch_size* batch_size, 1), target_label.repeat(batch_size,1)], dim=1).view(batch_size, batch_size,-1)
    # [ batch_size x batch_size x 2 ]
    
    target_relation_mask = torch.zeros([batch_size, batch_size]).to(opt.device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            index_vec = index_matrix[i][j]
            target_relation_mask[i][j] = target_relation_matrix[index_vec[0], index_vec[1]]
            
    return target_relation_mask
    
    
def get_metrics_f1(y_true, y_pred):
    average = 'macro'
    # f1 = f1_score(y_true, y_pred, average=average)
    f1_1 = f1_score(y_true == 1, y_pred == 1, labels=True)
    log.info('favor f1: {}'.format(100 * f1_1))
    f1_0 = f1_score(y_true == 0, y_pred == 0, labels=True)
    log.info('against f1: {}'.format(100 * f1_0))
    f1_avg = (f1_1 + f1_0) / 2
    # print("classification report: \n", classification_report(y_true, y_pred, digits=4))


    return f1_avg 

def train(opt):

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # configuration = BertConfig(max_position_embeddings=256)
    model_target = BertModel.from_pretrained("bert-base-multilingual-cased")
    model_target = model_target.to(opt.device) 
    freeze_net(model_target)
    
    train_file_path = os.path.join(opt.data_dir, "train.jsonl")
    valid_file_path = os.path.join(opt.data_dir, "valid.jsonl")
    test_file_path = os.path.join(opt.data_dir, "test.jsonl")
    
    
    # opt.num_target = len(all_targets_list)
    
    # train_dataset, valid_dataset, test_dataset = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, opt.num_train_lines, opt.max_seq_len, all_targets_list)
    
    all_targets_list,  train_dataset, valid_dataset, test_dataset = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, opt.num_train_lines, opt.max_seq_len)
    opt.num_target = len(all_targets_list)
    
    
    datasets_target_list = get_datasets_target(train_file_path, valid_file_path, test_file_path, tokenizer, opt.num_train_lines, opt.max_seq_len)
    
    log.info("Done loading datasets.")
    
    opt.num_labels = train_dataset.num_labels 
    opt.num_domains =train_dataset.num_domains
    
    # 由于没有引入LSTM等RNN模型，暂时不需要自定义data_loader中的collate方式 
    
    train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True) # 记得并行加sampler的时候 shuffle改为False 
    valid_loader = DataLoader(valid_dataset, opt.batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, opt.batch_size, shuffle=False) 
    
    
    log.info('Done constructing DataLoader. ')
    
    # opt.tokenized_max_len
    X = EmbeddingModule(opt.tokenized_max_len)
    X = X.to(opt.device)
    
    G = GAT(opt.emb_size, opt.gnn_dims, opt.att_heads, opt.attn_dropout, opt.concat_dropout, opt.leaky_alpha) # 参数改一下
    G = G.to(opt.device)
    
    P = StanceClassifier(opt.P_layers, opt.hidden_size, opt.num_labels, opt.concat_stance, opt.dropout, opt.P_bn)
    P = P.to(opt.device)
    
    # D = DomainClassifier(opt.D_layers, opt.hidden_size, opt.num_domains, opt.concat_domain, opt.dropout, opt.D_bn) # num_labels 
    # D = D.to(opt.device)

    CTL = ConTargetLoss(opt.temperature)
    CLL = ConLangLoss(opt.temperature)
    
    optimizer = optim.Adam(list(X.parameters()) + list(G.parameters()) + list(P.parameters()), lr=opt.learning_rate)
    
    log.info('Done loading models. ')
    
    
    # before training 
    
    # initialize target features 
    all_target_feature = get_target_feature(datasets_target_list, all_targets_list, model_target, tokenizer)
    # cal adj matrix 
    similarity_matrix = calculate_similarity(all_target_feature, opt.measurement)
    adj = calculate_adj(similarity_matrix, opt.sim_threshold) # original adj, to be optimized 
    
    # all_target_feature = torch.randn(opt.num_target, opt.emb_size).to(opt.device)
    # adj = (torch.ones([opt.num_target, opt.num_target]) - torch.eye(opt.num_target)).to(opt.device)
    
    
    
    # training 
    best_f1 = 0.0
    
    
    
    for epoch in range(opt.max_epoch):
        X.train()
        G.train()
        P.train()
        train_iter = iter(train_loader)
        correct, total = 0, 0
        max_iter_per_epoch = len(train_dataset) // opt.batch_size   
        for i, (inputs,y_t, y, y_l, y_d) in tqdm(enumerate(train_iter), total=max_iter_per_epoch):
            
            X.zero_grad()
            G.zero_grad()
            P.zero_grad()
            
            y = y.to(opt.device)
            y_t = y_t.to(opt.device)
            y_l = y_l.to(opt.device)
            
            embeds = X(inputs)
            node_features, weight = G(all_target_feature, adj)
            weight = weight.detach() # weight 是否需要detach
            
            features = torch.cat([embeds, node_features[y_t]], -1)
            # features = embeds
            outputs_stance = P(features)
            loss_ce_stance = F.nll_loss(outputs_stance, y)
            
            loss_con_lang = CLL(features, y, y_l)
            
            
            target_relation_matrix = get_target_relation(weight)
            target_mask = get_target_mask_batch(target_relation_matrix, y_t)
            loss_con_target = CTL(features, y, y_t, target_mask)
            
            loss_con = opt.beta_l * loss_con_lang + opt.beta_t * loss_con_target
            loss = opt.alpha * loss_ce_stance + (1 - opt.alpha) * loss_con
            
            
            # correct 
            _, pred = torch.max(outputs_stance, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # print("loss: ", loss.item())
            loss.backward()
            optimizer.step()
            
            
        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        log.info('Training Accuracy: {}%'.format(100.0*correct/total))
        
        log.info('Evaluating on valid set:')
        acc, f1 = evaluate(opt, valid_loader, X,G,P,node_features)
        
        if f1 > best_f1:
            log.info('Best f1 has been updated as {}'.format(f1))
            best_f1 = f1
            
        log.info('Evaluating on test set:')
        acc, f1 = evaluate(opt, test_loader, X,G,P, node_features)
        
        
    log.info('Best valid f1 is {}'.format(best_f1))
    
            

def evaluate(opt, data_loader, X,G,P, node_features):
    X.eval()
    G.eval()
    P.eval()
    iter1 = iter(data_loader)
    correct, total = 0, 0
    preds = []
    labels = []
    with torch.no_grad():
        for inputs, y_t, y, y_l, y_d in tqdm(iter1):
            
            y = y.to(opt.device)
            y_t = y_t.to(opt.device)
            y_l = y_l.to(opt.device)
            
            embeds = X(inputs)
            features = torch.cat([embeds, node_features[y_t]], -1)
            outputs_stance = P(features)
            _, pred = torch.max(outputs_stance, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            preds.append(pred)
            labels.append(y)
    
    y_pred = torch.cat(preds, dim=0).cpu()
    y_true = torch.cat(labels, dim=0).cpu()
    f1 = get_metrics_f1(y_true, y_pred)
    accuracy = correct / total
    log.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
    log.info('f1 on {} samples: {}'.format(total, 100.0*f1))
    return accuracy, f1
    
 

if __name__ == '__main__':
    
    train(opt)

    
