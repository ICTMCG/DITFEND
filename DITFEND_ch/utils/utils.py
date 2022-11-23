import pickle
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import torch
from prettytable import PrettyTable
import pandas as pd

class Recorder_target():
    def __init__(self, early_stop, target_domain):
        self.max = {'fscore': 0, 'precision': 0, 'recall': 0, 'acc': 0, 'auc': 0}
        self.cur = {'fscore': 0, 'precision': 0, 'recall': 0, 'acc': 0, 'auc': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop = early_stop
        self.target_domain = target_domain
    
    def add(self, x):
        self.cur = x[self.target_domain]
        self.curindex += 1
        print("current", self.cur)
        return self.judge()
    
    def judge(self):
        if self.cur['fscore'] > self.max['fscore']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_stop:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

class Recorder():

    def __init__(self, early_stop):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop = early_stop

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_stop:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

def metrics_weight(y_true, y_pred):
    metrics = {}
    metrics['auc'] = roc_auc_score(y_true, y_pred, average='weighted')
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics['metric'] = f1_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['acc'] = accuracy_score(y_true, y_pred)
    return metrics

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }

    metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred)
    
    for c, res in res_by_category.items():
        #precision, recall, fscore, support = precision_recall_fscore_support(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), zero_division=0)
        try:
            metrics_by_category[c] = {
                'precision': precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'recall': recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'fscore': f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'auc': metrics_by_category[c]['auc'],
                'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
            }
        except Exception as e:
            metrics_by_category[c] = {
                'precision': 0,
                'recall': 0,
                'fscore': 0,
                'auc': 0,
                'acc': 0
            }
    return metrics_by_category

def get_pretty_result(results):
    tb = PrettyTable()
    tb.field_names = ['category', 'precision', 'recall', 'fscore', 'auc', 'acc']
    overall_precision = 0
    overall_recall = 0
    overall_fscore = 0
    overall_auc = 0
    overall_acc = 0
    for key, value in results.items():
        if(key == 'auc'):
            overall_auc = format(value, ".4f")
        elif(key == 'metric'):
            overall_fscore = format(value, ".4f")
        elif(key == 'recall'):
            overall_recall = format(value, ".4f")
        elif(key == 'precision'):
            overall_precision = format(value, ".4f")
        elif(key == 'acc'):
            overall_acc = format(value, ".4f")
        else:
            tb.add_row([key]+ list(value.values()))
    tb.add_row(["overall", overall_precision, overall_recall, overall_fscore, overall_auc, overall_acc])

    return tb

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content_id': batch[0].cuda(),
            'content': batch[1].cuda(),
            'content_masks': batch[2].cuda(),
            'label': batch[3].cuda(),
            'category': batch[4].cuda()
            }
    else:
        batch_data = {
            'content_id': batch[0],
            'content': batch[1],
            'content_masks': batch[2],
            'label': batch[3],
            'category': batch[4]
            }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
