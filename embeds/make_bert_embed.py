import numpy as np
import torch
import sys
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

cuda_device = torch.device('cuda:0')
cpu_device = torch.device('cpu')
pretrain_model_path = sys.argv[1]
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
model = BertModel.from_pretrained(pretrain_model_path).to(cuda_device)


def process(batch_text):
    batch_ids = []
    for single_text in batch_text:
        token_ids = tokenizer.encode(
            single_text, pad_to_max_length=True,
            max_length=128, return_tensors='pt')
        batch_ids.append(token_ids)
    batch_ids = torch.cat(batch_ids).to(cuda_device)
    with torch.no_grad():
        hidden_states = model(batch_ids)[0][:, 0, :]
    return hidden_states.to(cpu_device)


embed = []
with open('../data/entity2text.txt', 'r') as f:
    batch, batch_size = [], 600
    for line in tqdm(f.readlines(), desc='encoding texts'):
        entity, text = line.strip().split('\t')
        batch.append(text)
        if len(batch) == batch_size:
            embed.append(process(batch))
            batch = []
if len(batch) > 0:
    embed.append(process(batch))
embed = torch.cat(embed).numpy()
np.savez_compressed('emb_raw.npz', embed=embed)
