import os
import numpy as np

import pickle
import torch
from torch.utils.data import Dataset
import time
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
                 all_targets_list): # TODO: max_seq_len的处理方式待定
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
        self.num_labels = 2  # stance label 个数
        self.label_dict = {"FAVOR": 1, "AGAINST": 0}
        self.domain_dict = {"Digitisation": 0,
                            "Economy": 1, 
                            "Education": 2,
                            "Finances": 3,
                            "Foreign Policy": 4, 
                            "Immigration": 5,
                            "Infrastructure & Environment": 6, 
                            "Security": 7,
                            "Society": 8, 
                            "Welfare": 9
                            } 
        self.lang_dict = {"de": 1, "fr": 0} 
        self.all_targets_list = all_targets_list
        
        
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
                    if question_id not in self.all_targets_list:
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
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break

 
                elif split == "train":  # 要两种语言 
                    # 选取规则
                    topic = answer["topic"]


                    if topic not in self.domain_dict.keys():
                        continue

                    lang = answer["language"] 
                    
                    question = answer["question"]
                    question_id = answer["question_id"]
                    self.raw_X_question.append(question)
                    self.question_id.append(question_id)

                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)
                    
                    self.X.append((question, comment[:self._max_seq_len]))
                    

                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
                    lang_index = self.lang_dict[lang]
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                
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
                    if question_id not in self.all_targets_list:
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
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break


    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y_t[idx], self.Y[idx],self.Y_l[idx])   
    
    def get_all_questions(self):
        
        question_set = set()
        for q_id in self.question_id:
            question_set.add(q_id)
            
        return question_set
    
    def to_Y_t(self, sorted_questions):  
        sorted_questions_list = list(sorted_questions)
        sorted_questions_dict = {}
        for i, q in enumerate(sorted_questions_list):
            sorted_questions_dict[q] = i
            
        for q in self.question_id:
            label_t = sorted_questions_dict[q]
            self.Y_t.append(label_t)
            
        
    
    



# 按照target获取 
class XStanceDataset_target(Dataset):
    def __init__(self,
                 split, 
                 settype,
                 file_path,
                 target, 
                 tokenizer,
                 num_train_lines,
                 max_seq_len): 
        self._settype = settype # test set type 
        self._max_seq_len = max_seq_len
        self._target = target # 获取指定target的样本
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
        self.domain_dict = {"Digitisation": 0,
                            "Economy": 1, 
                            "Education": 2,
                            "Finances": 3,
                            "Foreign Policy": 4, 
                            "Immigration": 5,
                            "Infrastructure & Environment": 6, 
                            "Security": 7,
                            "Society": 8, 
                            "Welfare": 9
                            } 
        self.lang_dict = {"de": 1, "fr": 0} 

        
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
                 max_seq_len):
    # 按照每个target分别获取样本
    
    train_dataset = XStanceDataset('train', None, train_file_path, tokenizer, num_train_lines, max_seq_len, [])
        
    all_targets_list = list(train_dataset.get_all_questions())
    all_targets_list.sort()
        
    print(all_targets_list)
        
    valid_dataset = XStanceDataset('valid', None, valid_file_path, tokenizer, 0, max_seq_len, all_targets_list)
    test_dataset = XStanceDataset('test', 'new_comments_defr', test_file_path, tokenizer, 0, max_seq_len, all_targets_list)
        
    train_dataset.to_Y_t(all_targets_list)        
    valid_dataset.to_Y_t(all_targets_list)
    test_dataset.to_Y_t(all_targets_list)
    
    return all_targets_list, train_dataset, valid_dataset, test_dataset
    

if __name__ == "__main__":
    data_dir = "./dataset/" # mention the work dir

    tokenizer = ""

    train_file_path = os.path.join(data_dir, "train.jsonl")
    valid_file_path = os.path.join(data_dir, "valid.jsonl")
    test_file_path = os.path.join(data_dir, "test.jsonl")

    max_seq_len = 500 # 全部读入
    num_train_lines = 20
   
   
    all_targets_list,  train_dataset, valid_dataset, test_dataset = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len)
    
    print(all_targets_list)
    print(len(all_targets_list))
    print(len(train_dataset))
    
    print(len(valid_dataset))
    print(len(test_dataset))
    


    



    



    


                
            
