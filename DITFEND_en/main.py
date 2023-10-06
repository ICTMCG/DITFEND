import os
import torch
import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings('ignore')
parser.add_argument('--model_name', default='bert')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--finetune_epoches', type = int, default = 2)
parser.add_argument('--finetune_domain', type = int, default = -1)
parser.add_argument('--target_domain', type = int, default = -1)
parser.add_argument('--meta_task', type = int, default = 3)
parser.add_argument('--batchsize', type = int, default = 64)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--bert_vocab_file', default='../pretrained_model/roberta-base/vocab.json')
parser.add_argument('--root_path', default='/data/')
parser.add_argument('--bert_emb', default='../pretrained_model/roberta-base')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='1')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=300)
parser.add_argument('--finetune_lr', type = float, default = 0.0003)
parser.add_argument('--local_lr', type = float, default = 0.0002)
parser.add_argument('--global_lr', type=float, default=0.001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='/data/nanqiong/data/emb/fasttext/cc.en.300.bin')
parser.add_argument('--log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
