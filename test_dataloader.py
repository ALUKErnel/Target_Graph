
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
from tqdm import tqdm 
import logging

from data_prep.xstance_dataset import get_datasets
from data_prep.xstance_dataset import get_datasets_target

from models import *
from criterions import *
from options import opt
from utils2 import * 



from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

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
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'mbert_0707_src+tgt.txt'))
log.addHandler(fh)


log.info('Fine-tuning mBERT with options:')

log.info(opt)
train_file_path = os.path.join(opt.data_dir, "train.jsonl")
valid_file_path = os.path.join(opt.data_dir, "valid.jsonl")
test_file_path = os.path.join(opt.data_dir, "test.jsonl")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
train_dataset, valid_dataset, test_dataset = get_datasets(train_file_path, valid_file_path, test_file_path, tokenizer, 500)


train_loader = DataLoader(train_dataset, 16, shuffle=True) # 记得并行加sampler的时候 shuffle改为False 
valid_loader = DataLoader(valid_dataset, 16, shuffle=False) 
test_loader = DataLoader(test_dataset, 16, shuffle=False) 

max_iter_per_epoch = 5

train_iter = iter(train_loader)
for i, (inputs, y) in tqdm(enumerate(train_iter), total=max_iter_per_epoch):
    print(input)
    print(y)
    print(type(y))