

import torch
from torch import autograd, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import XLMConfig, XLMTokenizer, XLMModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel
from layers import * 
from options import opt
import random
import numpy as np


# contrastive language alignment (different language and the same stance label)

class ConLangLoss(nn.Module): 
    def __init__(self, temperature, scale_by_temperature=True):
        super(ConLangLoss,self).__init__()
        self.temperature = temperature
        self.scale = scale_by_temperature
    
    def forward(self, features, labels, labels_l, mask=None):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        
        labels = labels.contiguous().view(-1, 1)
        labels_l = labels_l.contiguous().view(-1, 1)

        mask_y = torch.eq(labels, labels.T).float().to(opt.device)

        mask_l = torch.eq(labels_l, labels_l.T).float().to(opt.device)
        
        # 20221023
        #logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)


        # mask
        mask_mask_lang = 1 - mask_l # lang不同的样本 （对角线为0）
        positive_mask = mask_y  * mask_mask_lang
        
        # positive_mask = mask_y - torch.eye(batch_size).to(opt.device) # change back
        
        negative_mask = 1 - positive_mask 
        negative_mask = negative_mask - torch.eye(batch_size).to(opt.device)
        

        #
        num_positives_per_row = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(exp_logits * negative_mask, dim=1, keepdim=True) + \
            torch.sum(exp_logits * positive_mask, dim=1, keepdim=True)

        log_probs = logits - torch.log(denominator) # ??

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positive_mask, dim=1)[num_positives_per_row > 0] / \
            num_positives_per_row[num_positives_per_row>0]
        
        
        loss = - log_probs 
        if self.scale:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    
    
    

# contrastive target alignment (similiar target and the same stance label)

class ConTargetLoss(nn.Module): 
    def __init__(self, temperature, scale_by_temperature=True):
        super(ConTargetLoss,self).__init__()
        self.temperature = temperature
        self.scale = scale_by_temperature
    
    def forward(self, features, labels, labels_l, labels_t, target_positive_mask):
        
        # target_positive_mask 由GAT输出的weight矩阵得到，
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        
        labels = labels.contiguous().view(-1, 1)
        labels_l = labels_l.contiguous().view(-1, 1)
        labels_t = labels_t.contiguous().view(-1, 1)

        mask_y = torch.eq(labels, labels.T).float().to(opt.device)
        mask_y -= torch.eye(batch_size).to(opt.device)
        mask_l = torch.eq(labels_l, labels_l.T).float().to(opt.device)
        mask_l = torch.ones(batch_size,batch_size).to(opt.device) - mask_l
        
        # 20221023
        #logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)


        # mask
        
        # mask_mask = target_positive_mask - torch.eye(batch_size).to(opt.device) # 由target的label计算而得 
        middle = mask_y * mask_l
        positive_mask = middle * target_positive_mask # target相似(相同），标签相同  （去掉对角线）

        negative_mask = 1 - positive_mask # 其他为负样本（去掉对角线）
        negative_mask = negative_mask - torch.eye(batch_size).to(opt.device)
        
        #
        num_positives_per_row = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(exp_logits * negative_mask, dim=1, keepdim=True) + \
            torch.sum(exp_logits * positive_mask, dim=1, keepdim=True)

        log_probs = logits - torch.log(denominator) # ??

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positive_mask, dim=1)[num_positives_per_row > 0] / \
            num_positives_per_row[num_positives_per_row>0]
        
        
        loss = - log_probs 
        if self.scale:
            loss *= self.temperature
        loss = loss.mean()
        return loss