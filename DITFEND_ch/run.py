from unicodedata import category
from gensim.models.keyedvectors import Vocab
from torch.serialization import save
from transformers.file_utils import CONFIG_NAME
import torch
import tqdm
import pickle
import logging
import os
import time
import json
from copy import deepcopy
import numpy as np
import math
import random

from utils.utils import Averager, data2gpu, Recorder
from utils.dataloader import bert_data, w2v_data
from models.metamodel import Trainer as MetaTrainer

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.meta_task = config['meta_task']
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.weight_decay = config['weight_decay']
        self.finetune_lr = config['finetune_lr']
        self.local_lr = config['local_lr']
        self.global_lr = config['global_lr']
        self.epoch = config['epoch']
        self.emb_type = config['emb_type']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert_emb = config['bert_emb']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.log_dir = config['log_dir']
        self.save_param_dir = config['save_param_dir']

        self.train_path = self.root_path + 'train_id.pkl'
        self.val_path = self.root_path + 'val_id.pkl'
        self.test_path = self.root_path + 'test_id.pkl'

        self.category_dict = {
            "科技": 0,  
            "军事": 1,  
            "教育考试": 2,  
            "灾难事故": 3,  
            "政治": 4,  
            "医药健康": 5,  
            "财经商业": 6,  
            "文体娱乐": 7,  
            "社会生活": 8,  
        }
        if(self.meta_task == 3):
            self.category_dict = {
                "政治": 0,  
                "医药健康": 1,  
                "文体娱乐": 2,  
            }
        if(self.meta_task == 6):
            self.category_dict = {
                "教育考试": 0,
                "灾难事故": 1,
                "医药健康": 2,
                "财经商业": 3, 
                "文体娱乐": 4,
                "社会生活": 5, 
            }

    def get_dataloader(self):
        if self.emb_type == 'bert' and os.path.isfile('bert_dataloader_' + str(self.meta_task) + '.pkl'):
            with open('bert_dataloader_' + str(self.meta_task) + '.pkl', 'rb') as f:
                split_train_loader, train_loader, val_loader, test_loader = pickle.load(f)
        elif self.emb_type == 'w2v' and os.path.isfile('w2v_dataloader_' + str(self.meta_task) + '.pkl'):
            with open('w2v_dataloader_' + str(self.meta_task) + '.pkl', 'rb') as f:
                split_train_loader, train_loader, val_loader, test_loader = pickle.load(f)
        else:
            if self.emb_type == 'bert':
                loader = bert_data(max_len = self.max_len, batch_size = self.batchsize, vocab_file = self.vocab_file,
                            category_dict = self.category_dict, task_num = self.meta_task, num_workers=self.num_workers)
            elif self.emb_type == 'w2v':
                loader = w2v_data(max_len=self.max_len, vocab_file=self.vocab_file, emb_dim = self.emb_dim,
                        batch_size=self.batchsize, category_dict=self.category_dict, task_num = self.meta_task, num_workers= self.num_workers)
            split_train_loader = loader.load_data_split(self.train_path, True)
            train_loader = loader.load_data(self.train_path, True)
            val_loader = loader.load_data(self.val_path, False)
            test_loader = loader.load_data(self.test_path, False)
            if(self.emb_type == 'bert'):
                with open('bert_dataloader_' + str(self.meta_task) + '.pkl', 'wb') as f:
                    pickle.dump([split_train_loader, train_loader, val_loader, test_loader], f)
            elif (self.emb_type == 'w2v'):
                with open('w2v_dataloader_' + str(self.meta_task) + '.pkl', 'wb') as f:
                    pickle.dump([split_train_loader, train_loader, val_loader, test_loader], f)
        return split_train_loader, train_loader, val_loader, test_loader

    def main(self):
        split_train_dataloader, train_dataloader, val_dataloader, test_dataloader = self.get_dataloader()
        train_param = {
            'lr': [self.global_lr] * 10,
        }
        param = train_param
        best_param = []
        json_path = './logs/json/' + self.model_name + '.json'
        json_result = []
        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            for i, v in enumerate(vs):
                self.global_lr = v
                trainer = MetaTrainer(base_model_name = self.model_name,
                                    emb_dim=self.emb_dim,
                                    mlp_dims = self.mlp_dims,
                                    task_num = self.meta_task,
                                    bert_emb = self.bert_emb,
                                    batchsize = self.batchsize,
                                    use_cuda = self.use_cuda,
                                    local_lr = self.local_lr,
                                    global_lr = self.global_lr,
                                    dropout = self.dropout,
                                    split_train_loader = split_train_dataloader,
                                    train_loader = train_dataloader,
                                    val_loader = val_dataloader,
                                    test_loader= test_dataloader,
                                    category_dict=self.category_dict,
                                    weight_decay = self.weight_decay,
                                    log_dir = self.log_dir,
                                    save_param_dir = self.save_param_dir,
                                    emb_type = self.emb_type,
                                    early_stop = self.early_stop,
                                    epoches = self.epoch)
                metrics, best_model_path = trainer.train_stage()
                json_result.append(metrics)
                if(metrics['metric'] > best_metric['metric']):
                    best_metric['metric'] = metrics['metric']
                    best_v = v
            best_param.append({p: best_v})
            print("best metric: ", best_metric)
            print("best model path: ", best_model_path)
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent = 4, ensure_ascii = False)
