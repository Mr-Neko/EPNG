from lib2to3.pgen2 import token
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from models.val_v7 import MainModule
from data.dataset_zero import PTData
from utils.parser import parse_args, load_config
from utils.util import IoU, EMA, cos_similar
from utils import dist
from utils import loss

import time

from transformers import BertTokenizer, BertModel
import tqdm
import torch
import os.path as osp
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bert_process(bert_model, bert_tokenizer, captions):
    token_ids = []
    segment_ids = []
    token_masks = []

    for caption in captions:

        temp = bert_tokenizer.encode_plus(
                                            caption[0], 
                                            add_special_tokens=True, 
                                            max_length=512, 
                                            padding='max_length')

        token_ids.append(torch.LongTensor(temp['input_ids']).to(device).unsqueeze(dim=0))
        segment_ids.append(torch.LongTensor(temp['token_type_ids']).to(device).unsqueeze(dim=0))
        token_masks.append(torch.LongTensor(temp['attention_mask']).to(device).unsqueeze(dim=0))

    token_id = torch.cat(token_ids, dim=0).to(device)
    segment_id = torch.cat(segment_ids, dim=0).to(device)
    token_mask = torch.cat(token_masks, dim=0).to(device)

    output = bert_model(token_id, segment_id, token_mask)

    
    hidden_states = output[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = token_embeddings.permute(1, 2, 0, 3).contiguous()

    token_vecs_sum = token_embeddings[:, :, -4:, :]
    token_vecs_sum = torch.mean(token_vecs_sum, dim=2).squeeze(dim=2)

    pos = token_mask.nonzero()[-1, 1].item() + 1
    
    return token_vecs_sum[:, 1 :pos-1, :].mean(dim=1).unsqueeze(dim=1)


def val(model, val_loader, cfg, query, matcher, unknown_word):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()

    bert.eval()
    pbar = tqdm.tqdm(len(val_loader))
    Iou_mask2noun = []

    times = 0
    model.eval()
    with torch.no_grad():
        for i, (image, segms, captions, h, w) in enumerate(val_loader):
            
            gt_noun = bert_process(bert, tokenizer, captions)

            image = image.to(device)
            segms = segms.to(device)

            bsz = image.shape[0]

            start = time.time()
            mask, _, gt_noun = model(image, gt_noun)
            end = time.time()

            times += end - start
            mask = mask.to(device)

            for b in range(bsz):

                pred_mask = torch.nn.functional.interpolate(mask, (h[b].item(), w[b].item()), mode='bilinear', align_corners=True)
                pred_mask.squeeze_(dim=0)
                pred_mask.squeeze_(dim=0)

                # pred_mask = mask[b, :]
                pred_mask[pred_mask >= 0.3] = 1
                pred_mask[pred_mask < 0.3] = 0
                tgt_mask = segms[b]
                
                Iou_mask2noun.append(IoU(pred_mask, tgt_mask).unsqueeze(0))


            pbar.update(1)

    Iou_mask2noun = torch.cat(Iou_mask2noun, dim=0)

    average_accuracy = Iou_mask2noun.mean().item()
    Iou_mask2noun = Iou_mask2noun.cpu().numpy()

    predictions = (Iou_mask2noun >= 0.1).astype(int)
    TP = np.sum(predictions)
    p_01 = TP / len(predictions)

    predictions = (Iou_mask2noun >= 0.4).astype(int)
    TP = np.sum(predictions)
    p_04 = TP / len(predictions)

    predictions = (Iou_mask2noun >= 0.2).astype(int)
    TP = np.sum(predictions)
    p_02 = TP / len(predictions)

    predictions = (Iou_mask2noun >= 0.3).astype(int)
    TP = np.sum(predictions)
    p_03 = TP / len(predictions)

    predictions = (Iou_mask2noun >= 0.5).astype(int)
    TP = np.sum(predictions)
    p_05 = TP / len(predictions)

    times = times / len(predictions)
    return average_accuracy, p_01, p_02, p_03, p_04, p_05, times


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dist.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # device_ids = [0, 1, 2, 3]

    val_dataset_1 = PTData(cfg, 'refcoco', 'testA')
    val_loader_1 = DataLoader(
        val_dataset_1,
        batch_size=1,
        shuffle=False
    )

    val_dataset_2 = PTData(cfg, 'refcoco', 'testB')
    val_loader_2 = DataLoader(
        val_dataset_2,
        batch_size=1,
        shuffle=False
    )

    val_dataset_3 = PTData(cfg, 'refcoco+', 'testA')
    val_loader_3 = DataLoader(
        val_dataset_3,
        batch_size=1,
        shuffle=False
    )

    val_dataset_4 = PTData(cfg, 'refcoco+', 'testB')
    val_loader_4 = DataLoader(
        val_dataset_4,
        batch_size=1,
        shuffle=False
    )

    val_dataset_5 = PTData(cfg, 'refcocog', 'test')
    val_loader_5 = DataLoader(
        val_dataset_5,
        batch_size=1,
        shuffle=False
    )

    model = MainModule(cfg.MODEL.ML, cfg.MODEL.TD).cuda()


    if dist.is_master_proc():
        print("Model:\n{}".format(model))
        print("Params: {:,}".format(np.sum([p.numel() for p in model.parameters()]).item()))
        print("Mem: {:,} MB".format(torch.cuda.max_memory_allocated() / 1024 ** 3))
        print("nvidia-smi")
        os.system("nvidia-smi")


    '''
    if not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    writer = SummaryWriter(cfg.OUTPUT_DIR + '/tensorboard_log')

    # Load a checkpoint to resume training if applicable.
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'best_checkpoint.pth')
    start_epoch = 0
    if osp.exists(checkpoint_path):
        print('Resuming training: loading model from: {0}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])
    '''
    try:
        # Perform the training loop
        query = torch.randn((1, cfg.MODEL.NUM_QUERY, cfg.MODEL.EMBED_DIM))
        # query.require_grad = False

        for epoch in range(0, 60):
            # Shuffle the dataset
            # Train for one epoch
            verage_accuracy, p_01, p_02, p_03, p_04, p_05, times = val(model, val_loader_1, cfg, query, None, None)
            print(verage_accuracy, p_01, p_02, p_03, p_04, p_05, times)

            verage_accuracy, p_01, p_02, p_03, p_04, p_05, times = val(model, val_loader_2, cfg, query, None, None)
            print(verage_accuracy, p_01, p_02, p_03, p_04, p_05, times)

            verage_accuracy, p_01, p_02, p_03, p_04, p_05, times = val(model, val_loader_3, cfg, query, None, None)
            print(verage_accuracy, p_01, p_02, p_03, p_04, p_05, times)

            verage_accuracy, p_01, p_02, p_03, p_04, p_05, times = val(model, val_loader_4, cfg, query, None, None)
            print(verage_accuracy, p_01, p_02, p_03, p_04, p_05, times)

            verage_accuracy, p_01, p_02, p_03, p_04, p_05, times = val(model, val_loader_5, cfg, query, None, None)
            print(verage_accuracy, p_01, p_02, p_03, p_04, p_05, times)
            break
            train_loss, word_loss, dice_loss = train_epoch(model, train_loader, optimizer, epoch, cfg, query,
                                                                 matcher_word, unknown_word, scheduler)
            ema.update()
            ema.apply_shadow()
            average_accuracy = val(model, val_loader, cfg, query, matcher_word, unknown_word)

            if dist.is_master_proc():

                writer.add_scalar('data/loss', train_loss, global_step=epoch)
                writer.add_scalar('data/loss_word', word_loss, global_step=epoch)
                writer.add_scalar('data/loss_dice', dice_loss, global_step=epoch)

                writer.add_scalar('data/accuracy', average_accuracy, global_step=epoch)

                with open(osp.join(cfg.OUTPUT_DIR, "log.txt"), "a") as f:
                    f.write("Epoch: {:d}, loss: {:f}, acc: {:f}\n".format(epoch, train_loss, average_accuracy))

            if dist.is_master_proc():
                if best_val_score is None or average_accuracy > best_val_score:
                    best_val_score = average_accuracy
                    checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'best_checkpoint.pth')
                    checkpoint = {
                        "epoch": epoch,
                        "model_state": model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "accuracy": average_accuracy,
                        'scheduler': scheduler.state_dict()
                    }
                    torch.save(checkpoint, checkpoint_path)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("forkserver")
    args = parse_args()
    cfg = load_config(args)
    train(cfg)
