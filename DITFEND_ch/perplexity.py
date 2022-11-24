import math
import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import tqdm
import argparse
import pandas as pd

class Perplexity:
    def __init__(self, path, gpu):
        super(Perplexity, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForMaskedLM.from_pretrained(path)
        self.device = torch.device('cuda:{}'.format(gpu))
        self.model.to(self.device)

    def cal_pp(self, text, char):
        self.model.eval()
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segment_ids]).to(self.device)
        masked_index = tokenized_text.index('[MASK]')
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        prob = torch.softmax(predictions[0][0][masked_index], dim = 0)
        for index in range(prob.size(0)):
            pred_char = self.tokenizer.convert_ids_to_tokens([index])[0]
            if(pred_char == char):
                return prob[index]
        return 0.0001

    def cal_sentence_pp(self, text):
        sum = 0
        text = text[:170]
        for i, ch in enumerate(text):
            masked_text = '[CLS]{}[MASK]{}[SEP]'.format(text[:i], text[i+1:])
            sum = sum + math.log(self.cal_pp(masked_text, ch), 2)
        return -1.0 * sum / len(text)

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./finetune_politics/')
parser.add_argument('--in_file', default='./data/train.csv')
parser.add_argument('--out_file', default='./data/perplexity/train_with_politics_perplexity.csv')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

path = args.path
in_file = args.in_file
out_file = args.out_file
gpu = args.gpu
PP = Perplexity(path, gpu)
if __name__ == "__main__":
    with open(in_file, 'r') as fr:
        data = pd.read_csv(fr)
        data.insert(len(data.columns), 'perplexity', 0.0)
        for i in tqdm.tqdm(range(data.shape[0])):
            data.iloc[i, 5] = PP.cal_sentence_pp(data.iloc[i, 1])
        with open(out_file, 'w') as fw:
            data.to_csv(fw)
 
