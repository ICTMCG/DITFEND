import os
import torch
import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--finetune_epoches', type = int, default = 2)
parser.add_argument('--finetune_domain', type = int, default = -1)
parser.add_argument('--meta_task', type = int, default = 9)
parser.add_argument('--batchsize', type = int, default = 64)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--bert_vocab_file', default='hf1/chinese-roberta-wwm-ext/vocab.txt')
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--bert_emb', default='hf1/chinese-roberta-wwm-ext')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='3')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--finetune_lr', type = float, default = 0.0009)
parser.add_argument('--local_lr', type = float, default = 0.0002)
parser.add_argument('--global_lr', type=float, default=0.0009) 
parser.add_argument('--emb_type', default='w2v')
parser.add_argument('--w2v_vocab_file', default='../pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if args.finetune_domain == -1:
    from run import Run
else:
    from finetune import Run

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('finetune lr: {}; local lr: {}; global lr: {}; \
           model name: {}; meta_task: {}; batchsize: {}; epoch: {}; \
               gpu: {}; emb_dim: {}'.format(args.finetune_lr, 
                                            args.local_lr,
                                            args.global_lr,
                                            args.model_name, 
                                            args.meta_task, 
                                            args.batchsize, 
                                            args.epoch, 
                                            args.gpu, 
                                            emb_dim))


config = {
        'use_cuda': True,
        'meta_task': args.meta_task,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert_emb': args.bert_emb,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': emb_dim,
        'finetune_lr': args.finetune_lr,
        'local_lr': args.local_lr, 
        'global_lr': args.global_lr,
        'epoch': args.epoch,
        'finetune_epoches': args.finetune_epoches,
        'finetune_domain': args.finetune_domain,
        'model_name': args.model_name,
        'seed': args.seed,
        'log_dir': args.log_dir,
        'save_param_dir': args.save_param_dir
        }


if __name__ == '__main__':
    Run(config = config
        ).main()
