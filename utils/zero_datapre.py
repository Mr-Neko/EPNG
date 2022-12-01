import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
# from data.dataset_bert import PTData
from torch.utils.data import DataLoader
# from utils.parser import parse_args, load_config


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').cuda()

model.eval()
with open('/home/jjy/nips22/dataset/annotations/refcoco.json', 'r') as f:
    datas = json.load(f)


split = datas['testA']
for data in split:

    caption = data['refs']
    temp = tokenizer.encode_plus(
                                        caption, 
                                        add_special_tokens=True, 
                                        max_length=50, 
                                        padding='max_length')

    token_id = torch.LongTensor(temp['input_ids']).cuda().unsqueeze(dim=0).cuda()
    segment_id = torch.LongTensor(temp['token_type_ids']).cuda().unsqueeze(dim=0).cuda()
    token_mask = torch.LongTensor(temp['attention_mask']).cuda().unsqueeze(dim=0).cuda()



    with torch.no_grad():
        output = model(token_id, segment_id, token_mask)
    
    print(output.pooler_output.shape)