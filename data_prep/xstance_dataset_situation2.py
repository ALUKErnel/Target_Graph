import os
import numpy as np

import pickle
import torch
from torch.utils.data import Dataset
import time
import jsonlines
import jsonlines

# test for tokenizer 
import torch
from transformers import XLMConfig, XLMTokenizer, XLMModel

# train中 语言信息：de fr 混合 domain信息：N个domain同时存在 
# 按照split获取 
class XStanceDataset(Dataset):
    def __init__(self,
                 split, 
                 settype,
                 file_path,
                 tokenizer,
                 num_train_lines,
                 max_seq_len,
                 all_targets_list,
                 src_targets_list, 
                 tgt_targets_list
                 ): # TODO: max_seq_len的处理方式待定
        self._settype = settype # test set type 
        self._max_seq_len = max_seq_len
        self.raw_X_question = []
        self.raw_X_comment = [] 
        self.question_id = []
        # self.X_question = [] # (ids, lengths)
        # self.X_comment = [] # (ids, lengths)
        self.X = [] # (ids, lengths)
        # self.X_embedding = [] 
        self.Y = []
        self.Y_t = []
        self.Y_l = []
        self.Y_d = []
        self.num_labels = 2  # stance label 个数
        
        self.label_dict = {"FAVOR": 1, "AGAINST": 0}
        self.domain_dict = { "Foreign Policy": 4, "Immigration": 5}
        self.num_domains = len(self.domain_dict)
                            #{"Digitisation": 0}
                            # "Economy": 1
                            # "Education": 2
                            # "Finances": 3
                            # "Foreign Policy": 4, 
                            # "Immigration": 5,
                            # "Infrastructure & Environment": 6, 
                            # "Security": 7,
                            # "Society": 8, 
                            # "Welfare": 9
                            # } 
        self.lang_dict = {"de": 1, "fr": 0} 
        self.all_targets_list = all_targets_list
        self.src_targets_list = src_targets_list
        self.tgt_targets_list = tgt_targets_list
        self.num_domains = 0
        
        with jsonlines.open(file_path, 'r') as inf:
            print("fine ")
            cnt = 0
            for i, answer in enumerate(inf):
                # only take the lang instances
                if split == "test": 
                    # 选取规则
                    if answer["test_set"] != self._settype: # 选取cross-lingual下的new_comments_defr
                        continue
                    if answer["language"] != "fr":  # 只选取fr的数据
                        continue
                    
                    topic = answer["topic"] 
                    if topic not in self.domain_dict.keys():  # 只选取规定范围内的domain
                        continue
                    
                    question_id = answer["question_id"]
                    if question_id not in self.tgt_targets_list:
                        continue
                    
                    lang = answer["language"] 

                    question = answer["question"]
                    self.raw_X_question.append(question)
                    self.question_id.append(question_id)

                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)
                   
                    
                    self.X.append((question, comment[:self._max_seq_len]))   # 在这里直接对string进行截取
                    
                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
                    lang_index = self.lang_dict[lang]
                    domain_index = self.domain_dict[topic]
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    self.Y_d.append(domain_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break

 
                elif split == "train":  # 要两种语言 
                    # 选取规则
                    topic = answer["topic"]
                    if topic not in self.domain_dict.keys():
                        continue

                    lang = answer["language"] 
                    question_id = answer["question_id"]
                    if lang == "de" and question_id in self.src_targets_list:
                        question = answer["question"]
                        self.raw_X_question.append(question)
                        self.question_id.append(question_id)

                        comment = answer["comment"]
                        self.raw_X_comment.append(comment)
                    
                        self.X.append((question, comment[:self._max_seq_len]))
                
                        label = answer.get("label", None)
                    
                        label_index = self.label_dict[label]
                        lang_index = self.lang_dict[lang]
                        domain_index = self.domain_dict[topic]
                    
                        self.Y.append(label_index)
                        self.Y_l.append(lang_index)
                        self.Y_d.append(domain_index)
                
                        cnt += 1
                        if num_train_lines > 0 and cnt >= num_train_lines:
                            break
                        
                    if lang == "fr" and question_id in self.tgt_targets_list:
                        question = answer["question"]
                        self.raw_X_question.append(question)
                        self.question_id.append(question_id)

                        comment = answer["comment"]
                        self.raw_X_comment.append(comment)
                    
                        self.X.append((question, comment[:self._max_seq_len]))
                
                        label = answer.get("label", None)
                    
                        label_index = self.label_dict[label]
                        lang_index = self.lang_dict[lang]
                        domain_index = self.domain_dict[topic]
                    
                        self.Y.append(label_index)
                        self.Y_l.append(lang_index)
                        self.Y_d.append(domain_index)
                
                        cnt += 1
                        if num_train_lines > 0 and cnt >= num_train_lines:
                            break
                        
                    
                else: # split == "valid": 只要fr
                    
                    lang = answer["language"]
                    if lang != "fr":
                        continue
                    
                    topic = answer["topic"]
                    if topic not in self.domain_dict.keys():
                        continue
                    
                    question_id = answer["question_id"]
                    if question_id not in self.tgt_targets_list:
                        continue

                    
                    question = answer["question"]
                    self.raw_X_question.append(question)
                    self.question_id.append(question_id)

                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len]))
                    

                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
                    lang_index = self.lang_dict[lang]
                    domain_index = self.domain_dict[topic]
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    self.Y_d.append(domain_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break


    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y_t[idx], self.Y[idx],self.Y_l[idx], self.Y_d[idx])   
    
    def get_all_questions(self):
        
        question_set = set()
        for q_id in self.question_id:
            question_set.add(q_id)
            
        return question_set
    
    def to_Y_t(self, sorted_questions):   # all_targets_list
        sorted_questions_list = list(sorted_questions)
        sorted_questions_dict = {}
        for i, q in enumerate(sorted_questions_list):
            sorted_questions_dict[q] = i
            
        for q in self.question_id:
            label_t = sorted_questions_dict[q]
            self.Y_t.append(label_t)
            


# 按照target获取的数据集类
class XStanceDataset_target(Dataset):
    def __init__(self,
                 split, 
                 settype,
                 file_path,
                 lang,
                 target, 
                 both,
                 tokenizer,
                 num_train_lines,
                 max_seq_len
                 ): 
        self._settype = settype # test set type 
        self._max_seq_len = max_seq_len
        self._target = target # 获取指定target的样本
        self._lang = lang # 获取指定target和lang的样本 # 
        self._both = both 
        self.raw_X_question = []
        self.raw_X_comment = [] 
        self.X_question = [] # (ids, lengths)
        self.X_comment = [] # (ids, lengths)
        self.X = [] # (ids, lengths)
        self.X_embedding = [] 
        self.Y = []
        self.Y_d = []
        self.Y_l = []
        self.num_labels = 2  # stance label 个数
        self.label_dict = {"FAVOR": 1, "AGAINST": 0}
        self.domain_dict = {"Foreign Policy": 4, "Immigration": 5}
        self.lang_dict = {"de": 1, "fr": 0} 
        

        
        with jsonlines.open(file_path, 'r') as inf:
            print("fine target ")
            cnt = 0
            for i, answer in enumerate(inf):
                # only take the lang instances
                if split == "test": 
                    # 选取规则
                    if answer["test_set"] != self._settype: # 选取cross-lingual下的new_comments_defr
                        continue
                    if answer["language"] != "fr":  # 只选取fr的数据
                        continue
                    
                    topic = answer["topic"] 
 
                    if topic not in self.domain_dict.keys():  # 只选取规定范围内的domain
                        continue
                    
                    if answer["question_id"] != self._target:
                        continue
                    
                    lang = answer["language"] 

                    question = answer["question"]
                    self.raw_X_question.append(question)
                   

                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)
                   
                    
                    self.X.append((question, comment[:self._max_seq_len]))   # 在这里直接对string进行截取
                    
                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
                    domain_index = self.domain_dict[topic]
                    lang_index = self.lang_dict[lang]
                    
                    self.Y.append(label_index)
                    self.Y_d.append(domain_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break

 
                elif split == "train":  # 要两种语言 
                    # 选取规则
                    topic = answer["topic"]
                    question_id = answer["question_id"]
                    lang = answer["language"]


                    if topic not in self.domain_dict.keys():
                        continue
                    
                    if not self._both and lang == self._lang and question_id == self._target:
                        question = answer["question"]
                        self.raw_X_question.append(question)

                        comment = answer["comment"]
                        self.raw_X_comment.append(comment)
                    
                        self.X.append((question, comment[:self._max_seq_len]))
                    

                        label = answer.get("label", None)
                    
                        label_index = self.label_dict[label]
                        domain_index = self.domain_dict[topic]
                        lang_index = self.lang_dict[lang]
                    
                        self.Y.append(label_index)
                        self.Y_d.append(domain_index)
                        self.Y_l.append(lang_index)
                
                        cnt += 1
                        if num_train_lines > 0 and cnt >= num_train_lines:
                            break
                    if self._both and question_id == self._target:
                        question = answer["question"]
                        self.raw_X_question.append(question)

                        comment = answer["comment"]
                        self.raw_X_comment.append(comment)
                    
                        self.X.append((question, comment[:self._max_seq_len]))
                    

                        label = answer.get("label", None)
                    
                        label_index = self.label_dict[label]
                        domain_index = self.domain_dict[topic]
                        lang_index = self.lang_dict[lang]
                    
                        self.Y.append(label_index)
                        self.Y_d.append(domain_index)
                        self.Y_l.append(lang_index)
                
                        cnt += 1
                        if num_train_lines > 0 and cnt >= num_train_lines:
                            break
                        
                    
                else: # split == "valid": 只要fr
                    
                    if answer["language"] != "fr":
                        continue
                    
                    topic = answer["topic"]
                    
                    if topic not in self.domain_dict.keys():
                        continue
                    
                    if answer["question_id"] != self._target:
                        continue

                    lang = answer["language"] 
                    
                    question = answer["question"]
                    self.raw_X_question.append(question)
                    
                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len]))
                    

                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
                    domain_index = self.domain_dict[topic]
                    lang_index = self.lang_dict[lang]
                    
                    self.Y.append(label_index)
                    self.Y_d.append(domain_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break


    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])   # 其实此处的读入没必要读入其他标签信息 
    
    def get_target(self):
        return self._target


def get_datasets_main(train_file_path,
                 valid_file_path,
                 test_file_path,
                 tokenizer,
                 num_train_lines,
                 max_seq_len,
                 all_targets_list,
                 src_targets_list,
                 tgt_targets_list
                 ):
    # 按照每个target分别获取样本
    
    train_dataset = XStanceDataset('train', None, train_file_path, tokenizer, num_train_lines, max_seq_len, all_targets_list, src_targets_list, tgt_targets_list)
    valid_dataset = XStanceDataset('valid', None, valid_file_path, tokenizer, 0, max_seq_len, all_targets_list, src_targets_list, tgt_targets_list)
    test_dataset = XStanceDataset('test', 'new_comments_defr', test_file_path, tokenizer, 0, max_seq_len, all_targets_list, src_targets_list, tgt_targets_list)
        
    train_dataset.to_Y_t(all_targets_list)        
    valid_dataset.to_Y_t(all_targets_list)
    test_dataset.to_Y_t(all_targets_list)
    
    return all_targets_list, train_dataset, valid_dataset, test_dataset

def get_datasets_target_combine(train_file_path,
                        valid_file_path,
                        test_file_path,
                        tokenizer,
                        num_train_lines,
                        max_seq_len,
                        all_targets_list,
                        src_targets_list,
                        tgt_targets_list
                        ):
    dataset_list = []
    
    for t in all_targets_list:
        if t in src_targets_list and t in tgt_targets_list:
            train_dataset_each_target = XStanceDataset_target('train', None, train_file_path, "de", t, True, tokenizer, 0, max_seq_len)
            dataset_list.append(train_dataset_each_target)
        if t in src_targets_list and t not in tgt_targets_list:
            train_dataset_each_target = XStanceDataset_target('train', None, train_file_path, "de", t, False, tokenizer, 0, max_seq_len)
            dataset_list.append(train_dataset_each_target)
        if t not in src_targets_list and t in tgt_targets_list:
            train_dataset_each_target = XStanceDataset_target('train', None, train_file_path, "fr", t, False, tokenizer, 0, max_seq_len)
            dataset_list.append(train_dataset_each_target)
    
    return dataset_list 


def get_datasets_target_sep(train_file_path,
                        valid_file_path,
                        test_file_path,
                        tokenizer,
                        num_train_lines,
                        max_seq_len,
                        all_targets_list,
                        src_targets_list,
                        tgt_targets_list
                        ):
    dataset_list = []
    a = 1
    for t in all_targets_list:
        if t in src_targets_list and t in tgt_targets_list:
            train_dataset_each_target = XStanceDataset_target('train', None, train_file_path, "de", t, True, tokenizer, 0, max_seq_len)
            dataset_list.append(train_dataset_each_target)
        if t in src_targets_list and t not in tgt_targets_list:
            train_dataset_each_target = XStanceDataset_target('train', None, train_file_path, "de", t, False, tokenizer, 0, max_seq_len)
            dataset_list.append(train_dataset_each_target)
        if t not in src_targets_list and t in tgt_targets_list:
            train_dataset_each_target = XStanceDataset_target('train', None, train_file_path, "fr", t, False, tokenizer, 0, max_seq_len)
            dataset_list.append(train_dataset_each_target)
    
    return dataset_list 
        
    
def calculate_num_per_target(dataset, num_target):
    
    
    target_src = dict(zip(range(num_target),[0]*num_target))
    target_tgt = dict(zip(range(num_target),[0]*num_target))
    for line in dataset:
        if line[3] == 1:
            target_src[line[1]] += 1 
        else: 
            target_tgt[line[1]] += 1
    return target_src, target_tgt
            
            

if __name__ == "__main__":
    data_dir = "./dataset/" # mention the work dir

    tokenizer = ""

    train_file_path = os.path.join(data_dir, "train.jsonl")
    valid_file_path = os.path.join(data_dir, "valid.jsonl")
    test_file_path = os.path.join(data_dir, "test.jsonl")

    max_seq_len = 500 # 全部读入
    num_train_lines = 0
   
    all_targets_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3224, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471]
    src_targets_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497]
    tgt_targets_list = [64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3224, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471]
   
    all_targets_list,  train_dataset, valid_dataset, test_dataset = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len, all_targets_list, src_targets_list, tgt_targets_list)
    
    
    # num_target = len(all_targets_list)
    
    # print(all_targets_list)
    # print(len(all_targets_list))
    
    # print(len(train_dataset))
    # print(len(valid_dataset))
    # print(len(test_dataset))
    
    # target_src, target_tgt = calculate_num_per_target(train_dataset, num_target)
    # print(target_src)
    # print(target_tgt)
    
    # src_sum = 0
    # tgt_sum = 0
    # for key, values in target_src.items():
    #     src_sum += values
    # for key, values in target_tgt.items():
    #     tgt_sum += values
        
    # print(src_sum)
    # print(tgt_sum)
    
    dataset_list = get_datasets_target_combine(train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len, all_targets_list, src_targets_list, tgt_targets_list)
    
    # print(len(src_dataset_list))
    # print(len(tgt_dataset_list))
    # src_sum = 0
    # tgt_sum = 0
    # for d in src_dataset_list:
    #     src_sum += len(d)
    #     print(d.get_target(), len(d))
    
    # for d in tgt_dataset_list:
    #     tgt_sum += len(d)
    #     print(d.get_target(), len(d))
        
    # print(src_sum)
    # print(tgt_sum)
        
    for d in dataset_list:
        print(d.get_target(), len(d))
        
    
    # # with jsonlines.open("target_num.jsonl", 'w') as f:
    # #     f.write(target_src)
    # #     f.write(target_tgt)
    
    
    


    



    



    


                
            
