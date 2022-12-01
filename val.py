import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from models.transformer_text_query_v7 import MainModule
from data.dataset_mix_val import PTData
from utils.parser import parse_args, load_config
from utils.util import IoU, EMA, cos_similar
from utils import dist
from utils import loss

import time
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


def average(input):

    accuracy = []
    average_accuracy = 0
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (input >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)

        accuracy.append(a)

    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])):
        average_accuracy += (np.abs(t[1] - t[0])) * accuracy[i]
    
    return average_accuracy


def val(model, val_loader, cfg, query, matcher, unknown_word):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm.tqdm(len(val_loader))
    Iou_mask2noun = []
    single = []
    plural = []
    thing = []
    stuff = []

    model.eval()
    with torch.no_grad():
        for i, (image, segms, noun_index, gt_noun, is_thing, is_single, h, w) in enumerate(val_loader):

            if h.item() >= w.item():
            
                image = torch.nn.functional.interpolate(image, (1216, 1216), mode='bilinear')
            
            else:
            
                image = torch.nn.functional.interpolate(image, (1216, 1216), mode='bilinear')

            image = image.to(device)
            noun_index = noun_index.to(device)
            segms = segms.to(device)

            gt_noun = gt_noun.to(device)
            # gt_noun[:, 0] = unknown_word.unsqueeze(dim=0)

            bsz = image.shape[0]

            mask, _, gt_noun = model(image, gt_noun, noun_index)
            mask = torch.nn.functional.interpolate(mask, (h.item(), w.item()), mode='bilinear')
            mask = mask.to(device)

            
            mask[mask >= 0.5] = 1  
            mask[mask < 0.5] = 0
            

            for b in range(bsz):
                
                max_pos = np.argwhere(noun_index[b].cpu().numpy() == 0)[0][0]

                pred_mask = mask[b, 1:max_pos+1]
                tgt_mask = segms[b, :max_pos]
                    
                for j in range(max_pos):

                    iou = IoU(pred_mask[j], tgt_mask[j]).item()

                    if is_thing[0, j] == 0:

                        stuff.append(iou)
                    else:

                        thing.append(iou)

                    if is_single[0, j] > 1:
                        plural.append(iou)
                    else:
                        single.append(iou)
                        
                    Iou_mask2noun.append(iou)

            pbar.update(1)

    Iou_mask2noun = np.array(Iou_mask2noun)
    single = np.array(single)
    plural = np.array(plural)
    thing = np.array(thing)
    stuff = np.array(stuff)

    ac_av = average(Iou_mask2noun)
    ac_s = average(single)
    ac_p = average(plural)
    ac_t = average(thing)
    ac_st = average(stuff)
    return ac_av, ac_t, ac_st, ac_s, ac_p


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dist.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # device_ids = [0, 1, 2, 3]

    val_dataset = PTData(cfg, train=False)
    val_loader = DataLoader(
        val_dataset,
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

    if not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    writer = SummaryWriter(cfg.OUTPUT_DIR + '/tensorboard_log')

    # Load a checkpoint to resume training if applicable.
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'best_checkpoint.pth')
    start_epoch = 0
    if osp.exists(checkpoint_path):
        print('Resuming training: loading model from: {0}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch'] + 1
        model_final_path = osp.join(cfg.OUTPUT_DIR, 'best_checkpoint.pth')
        if osp.exists(model_final_path):
            model_final = torch.load(model_final_path)
            best_val_score = model_final['accuracy']
        else:
            best_val_score = None
    
    else:
        best_val_score = None


    try:
        # Perform the training loop
        # query.require_grad = False

        ac_av, ac_t, ac_st, ac_s, ac_p = val(model, val_loader, cfg, None, None, None)
        print(ac_av, ac_t, ac_st, ac_s, ac_p)


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("forkserver")
    args = parse_args()
    cfg = load_config(args)
    train(cfg)
