import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from models.EPNG import MainModule
from data.dataset_mix import PTData
from utils.parser import parse_args, load_config
from utils.util import IoU, EMA, cos_similar
from utils import dist
from utils import loss

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


def lambda_lr(s):
    base_lr = 0.00001
    print("s:", s)

    if s < 5:
        lr = base_lr
    elif s < 10:
        lr = base_lr / 2
    elif s < 20:
        lr = base_lr / 5
    else:
        lr = 5e-7
        # s += 1
    return lr

def train_epoch(model, loader, optimizer, epoch, cfg, query, matcher, unknown_word, scheduler):
    epoch_loss = 0
    epoch_dice = 0
    epoch_word = 0

    # random_size = [[640, 640]]
    # random_size = [[416, 416], [480, 480], [512, 512], [576, 576], [640, 640], [704, 704], [768, 768], [832, 832], [896, 896]]
    pbar = tqdm.tqdm(total=len(loader))

    model.train()

    for i, (image, segms, noun_index, gt_noun) in enumerate(loader):
        
        image = image.to(device)

        bsz = image.shape[0]

        gt_noun = gt_noun.to(device)
        # gt_noun[:, 0] = unknown_word.unsqueeze(dim=0)
        noun_index = noun_index.to(device)

        segms = segms.to(device)

        '''
        h, w = random_size[np.random.randint(0, len(random_size))]

        image = torch.nn.functional.interpolate(image, (h, w), mode='bilinear')
        segms = torch.nn.functional.interpolate(segms, (h, w), mode='bilinear')

        segms[segms>=0.5] = 1
        segms[segms<0.5] = 0
        '''
        
        mask, image_feature, gt_noun = model(image, gt_noun, noun_index)

        mask = mask.to(device)

        all_loss = 0    
        loss_dice_s = 0
        loss_word_s = 0

        for b in range(bsz):

            max_pos = np.argwhere(noun_index[b].cpu().numpy() == 0)[0][0]

            pred_mask = mask[b, 1:max_pos+1]
            tgt_mask = segms[b, :max_pos]

            # loss_word = loss.constrative_loss(word_embed[b], b_gt_noun, pred_index, tgt_index, device).mean()

            # loss_word_s += loss_word

            loss_dice = loss.dice_loss(pred_mask, tgt_mask).mean()
            loss_entrophy = loss.entropy_loss(pred_mask, tgt_mask).mean()
            loss_constrative = loss.constrative_loss(image_feature[b], gt_noun[b], tgt_mask)

            all_loss += 2 * loss_entrophy + 2*loss_dice + loss_constrative

        all_loss = all_loss / bsz
        # loss_word_s = loss_word_s / bsz
        # loss_dice_s = loss_dice_s / bsz

        '''
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        '''

        all_loss.backward()
        if (i + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if cfg.NUM_GPUS > 1:
            all_loss = dist.all_reduce([all_loss])[0]
            # loss_dice_s = dist.all_reduce([loss_dice_s])[0]
            # loss_word_s = dist.all_reduce([loss_word_s])[0]


        epoch_loss += all_loss.item()
        # epoch_word += loss_word_s.item()
        # epoch_dice += loss_dice_s.item()

        if dist.is_master_proc():
            pbar.update(1)
            if i % 50 == 0:
                print(' [{:5d}] ({:5d}/{:5d}) | ms/batch |'
                      ' loss {:.6f} | lr {:.7f}'.format(
                    epoch, i, len(loader),
                    all_loss.item(),
                    optimizer.param_groups[0]["lr"]))
    pbar.close()

    scheduler.step()
    if dist.is_master_proc():
        checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
        checkpoint = {
            "epoch": epoch,
            "model_state": model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
    return epoch_loss / i, epoch_word / i, epoch_dice / i


def val(model, val_loader, cfg, query, matcher, unknown_word):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm.tqdm(len(val_loader))
    Iou_mask2noun = []

    model.eval()
    with torch.no_grad():
        for i, (image, segms, noun_index, gt_noun) in enumerate(val_loader):

            image = image.to(device)
            noun_index = noun_index.to(device)
            segms = segms.to(device)

            gt_noun = gt_noun.to(device)
            # gt_noun[:, 0] = unknown_word.unsqueeze(dim=0)

            bsz = image.shape[0]

            mask, _, gt_noun = model(image, gt_noun, noun_index)

            mask = mask.to(device)

            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0

            for b in range(bsz):
                
                max_pos = np.argwhere(noun_index[b].cpu().numpy() == 0)[0][0]

                pred_mask = mask[b, 1:max_pos+1]
                tgt_mask = segms[b, :max_pos]
                    
                for j in range(max_pos):
                    Iou_mask2noun.append(IoU(pred_mask[j], tgt_mask[j]).unsqueeze(0))


            pbar.update(1)

    Iou_mask2noun = torch.cat(Iou_mask2noun, dim=0)

    if cfg.NUM_GPUS > 1:
        Iou_mask2noun = dist.all_gather([Iou_mask2noun])[0]

    Iou_mask2noun = Iou_mask2noun.cpu().numpy()
    accuracy = []
    average_accuracy = 0
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (Iou_mask2noun >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)

        accuracy.append(a)

    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])):
        average_accuracy += (np.abs(t[1] - t[0])) * accuracy[i]

    return average_accuracy


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dist.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # device_ids = [0, 1, 2, 3]

    train_dataset = PTData(cfg, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=(False if cfg.NUM_GPUS > 1 else True),
        sampler=(DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None),
        num_workers=4
    )

    val_dataset = PTData(cfg, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=(1 if cfg.NUM_GPUS > 1 else cfg.TRAIN.BATCH_SIZE),
        shuffle=False,
        sampler=(DistributedSampler(val_dataset) if cfg.NUM_GPUS > 1 else None)
    )

    cur_device = torch.cuda.current_device()
    model = MainModule(cfg.MODEL.ML, cfg.MODEL.TD)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device).cuda(device=cur_device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cur_device], output_device=cur_device,
                                                      find_unused_parameters=True)

    ema = EMA(model.module, 0.999)
    ema.register()

    matcher_word = loss.HungarianMatcher(1, 0).cuda(device=cur_device)
    matcher_word = torch.nn.DataParallel(matcher_word, device_ids=[cur_device], output_device=cur_device)

    if dist.is_master_proc():
        print("Model:\n{}".format(model))
        print("Params: {:,}".format(np.sum([p.numel() for p in model.parameters()]).item()))
        print("Mem: {:,} MB".format(torch.cuda.max_memory_allocated() / 1024 ** 3))
        print("nvidia-smi")
        os.system("nvidia-smi")

    def optimizer_wrapper(Optim, **kwargs):
        def init_func(model):
            return Optim(model.parameters(), **kwargs)

        return init_func

    optimizers = {
        "adamax": (
            optimizer_wrapper(optim.Adamax, lr=cfg.TRAIN.LR),
            lambda optim: optim.param_groups[0]["lr"],
        ),
        "adam": (
            optimizer_wrapper(optim.Adam, lr=cfg.TRAIN.LR, weight_decay=0.01),
            lambda optim: optim.param_groups[0]["lr"],
        ),
        "sgd": (
            optimizer_wrapper(optim.SGD, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.LR/10),
            lambda optim: optim.param_groups[0]["lr"],
        ),
    }

    # ignored_params = list(map(id, model.module.vs_backbone.parameters())) # 返回的是parameters的 内存地址
    # base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters()) 
    optimizer, _ = optimizers["adam"]
    optimizer = optimizer(model)
    '''
    optimizer = optim.AdamW([
        {'params': base_params},
        {'params': model.module.vs_backbone.parameters(), 'lr': 0.1*cfg.TRAIN.LR}], lr=cfg.TRAIN.LR, weight_decay=0.01)
    '''
    scheduler = LambdaLR(optimizer, lambda_lr)

    if not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    writer = SummaryWriter(cfg.OUTPUT_DIR + '/tensorboard_log')

    # Load a checkpoint to resume training if applicable.
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
    start_epoch = 0
    if osp.exists(checkpoint_path):
        print('Resuming training: loading model from: {0}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler'])
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
        query = torch.randn((1, cfg.MODEL.NUM_QUERY, cfg.MODEL.EMBED_DIM))
        unknown_word = torch.randn((1, cfg.MODEL.EMBED_DIM))
        # query.require_grad = False

        for epoch in range(start_epoch, 60):
            # Shuffle the dataset
            # Train for one epoch
            if cfg.NUM_GPUS > 1:
                train_loader.sampler.set_epoch(epoch)

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
    torch.multiprocessing.set_start_method("forkserver")
    args = parse_args()
    cfg = load_config(args)
    torch.multiprocessing.spawn(
        dist.run,
        nprocs=cfg.NUM_GPUS,
        args=(
            cfg.NUM_GPUS,
            train,
            args.init_method,
            0,
            1,
            "nccl",
            cfg,
        ),
        daemon=False,
    )
