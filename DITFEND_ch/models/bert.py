import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, get_pretty_result, metrics, Recorder
import pandas as pd

class BertFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, task_num, bert_emb, dropout):
        super(BertFNModel, self).__init__()
        self.task_num = task_num
        self.bert = BertModel.from_pretrained(bert_emb).requires_grad_(False)

        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = True
        #     if name.startswith("encoder.layer.11") \
        #             or name.startswith('encoder.layer.10') \
        #             or name.startswith('encoder.layer.9') \
        #             or name.startswith('encoder.layer.8') \
        #             or name.startswith('encoder.layer.7'):# \
        #             or name.startswith('encoder.layer.6')\
        #             or name.startswith('encoder.layer.5') \
        #             or name.startswith('encoder.layer.4')\
        #             or name.startswith('encoder.layer.3'):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        self.mlp = MLP(emb_dim, mlp_dims, dropout)
        self.attention = MaskAttention(emb_dim)
    
    def get_param(self):
        return list(self.mlp.parameters()) + list(self.attention.parameters())

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        # bert_feature = self.bert(inputs, attention_mask = masks).last_hidden_state[:, 0]
        bert_feature = self.bert(inputs, attention_mask = masks)[0]
        bert_feature, _ = self.attention(bert_feature, masks)
        #bert_feature = bert_feature[:, 0, :]
        output = self.mlp(bert_feature)
        return torch.sigmoid(output.squeeze(1))


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 task_num,
                 bert_emb,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 log_dir,
                 save_param_dir, 
                 early_stop = 5,
                 epoches = 100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert_emb = bert_emb
        self.dropout = dropout
        self.task_num = task_num
        
        if os.path.exists(log_dir):
            self.log_dir = log_dir
        else:
            self.log_dir = os.makedirs(log_dir)

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = os.makedirs(save_param_dir)
        
        self.model = BertFNModel(self.emb_dim, self.mlp_dims, self.task_num, self.bert_emb, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        

    def train(self, finetune_domain=-1):
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                optimizer.zero_grad()
                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_bert_' + self.emb_type + '_' + str(self.task_num) + 'domains_finetune' + str(finetune_domain) + '.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        if self.epoches > 0:
            self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_bert_' + self.emb_type + '_' + str(self.task_num) + 'domains_finetune' + str(finetune_domain) + '.pkl')))
        results = self.test(self.test_loader)
        pretty_results = get_pretty_result(results)
        with open(os.path.join(self.log_dir, 'bert_' + str(self.task_num) + '_finetune_' + str(finetune_domain) + '_results.txt'), 'w') as results_file:
            print(pretty_results, file = results_file)
        print(pretty_results)
        return pretty_results, os.path.join(self.save_param_dir, 'parameter_bert_' + self.emb_type + '_' + str(self.task_num) + 'domains_finetune' + str(finetune_domain) + '.pkl')

    def test(self, dataloader):
        test_df = pd.DataFrame(columns = ['content', 'label', 'pred', 'category'])
        test_df_wrong = pd.DataFrame(columns = ['content', 'label', 'pred', 'category'])
        content_id = []
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_content_id = batch_data['content_id']
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_pred = self.model(**batch_data)

                content_id.extend(batch_content_id.detach().cpu().numpy().tolist())
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)
