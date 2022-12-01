import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import h5py
import torch
import tqdm
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from data.dataset_bert import PTData
from torch.utils.data import DataLoader
from utils.parser import parse_args, load_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(data):
    id = []
    caption = []
    noun_vector = []

    for item in data:
        id.append(item[0])
        caption.append(item[1])
        noun_vector.append(item[2])

    return id, caption, noun_vector


class Bert(nn.Module):
    def __init__(self, model_name, freeze) -> None:
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _detect_subwords(self, tokenized_text):
        count = 0
        index = []
        for token in tokenized_text:
            if '##' in token:
                index.append(count + 1)
            count += 1

        t1 = []
        t2 = []
        for i in index:
            t1.append(i)
            if i + 1 not in index:
                t2.append([t1[0], t1[-1]])
                t1 = []
        return t2

    def _subword_filter(self, token_vecs: torch.Tensor, index):

        sublist = []
        if len(index):
            pos = 1
            for i in index:
                temp = token_vecs[pos: i[0] - 1]
                average = token_vecs[i[0] - 1: i[1] + 1].mean(dim=0).unsqueeze(0)
                if torch.isnan(average).any():
                    print('average wrong!')
                    print(i)
                pos = i[1] + 1
                sublist.extend([temp, average])
            sublist.append(token_vecs[pos:])
            output = torch.cat(sublist, dim=0)
            return output
        return token_vecs[1:]
    
    def forward(self, captions):
        
        token_ids = []
        segment_ids = []
        token_masks = []
        for caption in captions:

            temp = self.tokenizer.encode_plus(
                                                caption, 
                                                add_special_tokens=True, 
                                                max_length=300, 
                                                padding='max_length')

            token_ids.append(torch.LongTensor(temp['input_ids']).to(device).unsqueeze(dim=0))
            segment_ids.append(torch.LongTensor(temp['token_type_ids']).to(device).unsqueeze(dim=0))
            token_masks.append(torch.LongTensor(temp['attention_mask']).to(device).unsqueeze(dim=0))

        token_id = torch.cat(token_ids, dim=0).to(device)
        segment_id = torch.cat(segment_ids, dim=0).to(device)
        token_mask = torch.cat(token_masks, dim=0).to(device)

        output = self.model(token_id, segment_id, token_mask)
        hidden_state = output[2]

        token_embeddings = torch.stack(hidden_state, dim=0)
        token_embeddings = token_embeddings.permute(1, 2, 0, 3).contiguous()

        token_vecs_sum = token_embeddings[:, :, -4:, :]
        token_vecs_sum = torch.mean(token_vecs_sum, dim=2).squeeze(dim=2)
        
        '''
        outputs = []
        for b in range(bsz):
            tokenized_text = self.tokenizer.convert_ids_to_tokens(token_id[b])
            index = self._detect_subwords(tokenized_text)
            output = self._subword_filter(token_vecs_sum[b], index)
            
            len = output.shape[0]
            if len < 300:
                output = torch.cat((output, torch.zeros((300-len, 768)).to(device)), dim=0)
            
            assert output.shape[0] == 300

            output = output.unsqueeze(dim=0)
            outputs.append(output)
        '''
        return token_vecs_sum

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    
    train_dataset = PTData(cfg, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=4
    )
    '''

    val_dataset = PTData(cfg, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=4
    )
    '''
    lg_backbone = Bert('bert-base-uncased', True).to(device)
    pbar = tqdm.tqdm(total=len(train_loader))
    hdfile = h5py.File('/home/jjy/nips22/dataset/features/train2017_full.hdf5', 'w')
    for i, (id, caption, noun_vector) in enumerate(train_loader):

        bsz = len(caption)
        with torch.no_grad():
            text_feature = lg_backbone(caption)
            text_feature.requires_grad = False
            for b in range(bsz):

                hdfile.create_dataset(str(id[b]), data=text_feature[b].cpu().numpy())
        pbar.update(1)

    hdfile.close()

