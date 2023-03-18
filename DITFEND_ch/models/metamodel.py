from unicodedata import category
import torch
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from models.textcnn import TextCNNModel
from models.bigru import BiGRUModel
from models.bert import BertFNModel
import tqdm
from utils.utils import data2gpu, metrics, Averager, Recorder, metrics_weight, get_pretty_result
import os
import pandas as pd

class MetaModel(torch.nn.Module):
    def __init__(self, embed_dim, mlp_dims, task_num, bert_emb, emb_type, dropout, use_cuda, local_lr, global_lr,
                 weight_decay, base_model_name):
        super(MetaModel, self).__init__()
        self.base_model_name = base_model_name
        if base_model_name == 'textcnn':
            self.base_model = TextCNNModel(emb_dim = embed_dim, mlp_dims = mlp_dims, task_num = task_num, bert_emb = bert_emb, dropout = dropout, emb_type = emb_type)
        elif base_model_name == 'bigru':
            self.base_model = BiGRUModel(emb_dim = embed_dim, mlp_dims = mlp_dims, num_layers=1, task_num = task_num, bert_emb = bert_emb, dropout = dropout, emb_type = emb_type)
        elif base_model_name == 'bert':
            self.base_model = BertFNModel(emb_dim = embed_dim, mlp_dims = mlp_dims, task_num = task_num, bert_emb = bert_emb, dropout = dropout)
        if use_cuda:
            self.base_model = self.base_model.cuda()
        self.use_cuda = use_cuda
        self.local_lr = local_lr
        self.meta_optimizer = torch.optim.Adam(params = self.base_model.get_param(), lr = global_lr, weight_decay = weight_decay)

    def forward(self, x):
        return self.base_model(x)
      
class Trainer():
    def __init__(self,
                 base_model_name,
                 emb_dim,
                 mlp_dims,
                 task_num,
                 bert_emb,
                 batchsize,
                 use_cuda,
                 local_lr,
                 global_lr,
                 dropout,
                 split_train_loader,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 log_dir,
                 save_param_dir,
                 emb_type = 'w2v',
                 early_stop = 5,
                 epoches = 100):
        self.base_model_name = base_model_name
        self.local_lr = local_lr
        self.global_lr = global_lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.split_train_loader = split_train_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.batchsize = batchsize

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert_emb = bert_emb
        self.dropout = dropout
        self.emb_type = emb_type
        self.task_num = task_num

        if os.path.exists(log_dir):
            self.log_dir = log_dir
        else:
            self.log_dir = os.makedirs(log_dir)

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = os.makedirs(save_param_dir)
        
        self.model = MetaModel(self.emb_dim, self.mlp_dims, self.task_num, self.bert_emb, self.emb_type, self.dropout, self.use_cuda, self.local_lr, self.global_lr, self.weight_decay, self.base_model_name)

    def global_update(self, support_sets, query_sets, alpha):
        losses_q = []
        for i in range(len(support_sets)): 
            support_set = support_sets[i]
            query_set = query_sets[i]
            sup_loss = self.local_update(support_set, alpha = alpha)
            label = query_set['label']
            query_domain_label = query_set['category']

            if self.base_model_name == 'eann':
                query_set_y_pred, query_domain_pred = self.model.base_model(**query_set, alpha = -1)
                loss_fn = torch.nn.BCELoss()
                loss_q = loss_fn(query_set_y_pred, label.float()) + F.nll_loss(F.log_softmax(query_domain_pred, dim=1), query_domain_label)
            elif self.base_model_name == 'eddfn':
                query_set_y_pred, query_rec_feature, query_bert_feature, query_domain_pred = self.model.base_model(**query_set, alpha = 1)
                loss_fn = torch.nn.BCELoss()
                loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)
                loss_q = loss_fn(query_set_y_pred, label.float()) + loss_mse(query_rec_feature, query_bert_feature) + 0.1 * F.nll_loss(F.log_softmax(query_domain_pred, dim=1), query_domain_label)
            else:
                query_set_y_pred = self.model.base_model(**query_set)
                loss_fn = torch.nn.BCELoss()
                loss_q = loss_fn(query_set_y_pred, label.float())

            losses_q.append(loss_q)
        losses_q =torch.stack(losses_q).mean(0)
        self.model.meta_optimizer.zero_grad()
        losses_q.backward()
        self.model.meta_optimizer.step()
        fast_parameters = self.model.base_model.get_param()
        for weight in fast_parameters:
            weight.fast = None
        return losses_q

    def local_update(self, support_set, alpha):
        fast_parameters = self.model.base_model.get_param()
        for weight in fast_parameters:
            weight.fast = None
        label = support_set['label']
        domain_label = support_set['category']
        
        support_set_y_pred= self.model.base_model(**support_set)
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(support_set_y_pred, label.float())

        self.model.base_model.zero_grad()
        grad = torch.autograd.grad(loss, fast_parameters, create_graph = True, allow_unused = True)
        for k, weight in enumerate(fast_parameters):
            if grad[k] is None:
                continue
            if weight.fast is None:
                weight.fast = weight - self.local_lr * grad[k]
            else:
                weight.fast = weight.fast - self.local_lr * grad[k]
        return loss

    def train_stage(self):
        recorder = Recorder(self.early_stop)
        best_metric = 0
        for epoch_i in range(self.epoches):
            alpha = max(2. / (1. + np.exp(-10 * epoch_i / self.epoches)) - 1, 1e-1)
            print("Training Epoch {}:".format(epoch_i + 1))
            self.model.base_model.train()
            avg_loss = Averager()
            task_ids = list(self.category_dict.values())
            max_num = 0
            data_train_iter = []
            task_num = len(task_ids)
            for task_i in range(task_num):
                data_train_iter.append(iter(self.split_train_loader[task_i]))
                if(len(self.split_train_loader[task_i].dataset) > max_num):
                    max_num = len(self.split_train_loader[task_i].dataset)
            for i in tqdm.tqdm(range(int(max_num / self.batchsize))):
               
                batch_data_sup = []
                batch_data_qry = []
                for task_i in range(task_num):
                    try:
                        batch_data_train_d = data_train_iter[task_i].next()
                    except Exception as error:
                        data_train_iter[task_i] = iter(self.split_train_loader[task_i])
                        batch_data_train_d = data_train_iter[task_i].next()
                    batch_data_train_d = data2gpu(batch_data_train_d, use_cuda = self.use_cuda)
                    batch_data_sup.append(batch_data_train_d)
                    try:
                        batch_data_train_d = data_train_iter[task_i].next()
                    except Exception as error:
                        data_train_iter[task_i] = iter(self.split_train_loader[task_i])
                        batch_data_train_d = data_train_iter[task_i].next()
                    batch_data_train_d = data2gpu(batch_data_train_d, use_cuda = self.use_cuda)
                    batch_data_qry.append(batch_data_train_d)
                loss = self.global_update(batch_data_sup, batch_data_qry, alpha)
                avg_loss.add(loss.item())
            print("training epoch {}; loss {}".format(epoch_i + 1, avg_loss.item()))
            
            results = self.test(self.val_loader, test = False)
            mark = recorder.add(results)
            if mark == 'save':
                best_metric = results['metric']
                torch.save(self.model.base_model.state_dict(),
                        'parameter_{}_{}_{}_{}.pkl'.format(self.model.base_model_name, self.task_num, self.emb_type, best_metric))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.base_model.load_state_dict(torch.load('parameter_{}_{}_{}_{}.pkl'.format(self.model.base_model_name, self.task_num, self.emb_type, best_metric)))
        print("begin test............")
        results = self.test(self.test_loader, test = True)
        # print(results)
        pretty_results = get_pretty_result(results)
        with open(os.path.join(self.log_dir, self.base_model_name + '_' + self.emb_type + '_' + str(self.task_num) + '_' + '_test_results.txt'), 'w') as results_file:
            print(pretty_results, file = results_file)
        results_file.close()
        print(pretty_results)
        return results, 'parameter_{}_{}_{}_{}.pkl'.format(self.model.base_model_name, self.task_num, self.emb_type, best_metric)

    
    def test(self, dataloader, test = False):
        pred = []
        label = []
        category = []
        self.model.base_model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_pred = self.model.base_model(**batch_data)
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        result = metrics(label, pred, category, self.category_dict)

        return result
