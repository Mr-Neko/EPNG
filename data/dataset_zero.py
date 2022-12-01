import numpy
import torch
from torch.utils.data import Dataset
import json
import os.path as osp
from PIL import Image
import skimage.io as io
from torchvision import transforms
import numpy as np
from transformers import BertTokenizer
import h5py

class PTData(Dataset):
    def __init__(self, cfg, dataset, split) -> None:
        super(PTData, self).__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        
        self.num_box = cfg.DATA.NUM_BOXS
        self.h = cfg.DATA.IMAGE_H
        self.w = cfg.DATA.IMAGE_W

        self.transform = transforms.Compose([
			   transforms.Resize((self.h, self.w), interpolation=transforms.InterpolationMode.BICUBIC),
               self._convert_image_to_rgb,
               transforms.ToTensor(),
               transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.annotation_path = osp.join(cfg.DATA.ANNO_PATH, '{0:s}.json'.format(self.dataset))

        self.panoptic_path = osp.join(cfg.DATA.ANNO_PATH, 'panoptic_segmentation', self.dataset)

        self.image_path = osp.join(cfg.DATA.IMAGE_PATH, 'train2014')


        self.annotations = self.load_json(self.annotation_path)[self.split]

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, indexes):

        annotation = self.annotations[indexes]

        pure_image = Image.open(osp.join(self.image_path, 'COCO_train2014_{0:012d}.jpg'.format(annotation['iid'])))
        image = self.transform(pure_image)
        pure_image = np.array(pure_image)

        segms = np.load(osp.join(self.panoptic_path, '{:s}.npy'.format(str(annotation['mask_id'])))).astype(np.float32)
        segms = torch.from_numpy(segms)[None, None, :, :]

        h, w = segms.shape[2], segms.shape[3]

        # segms = torch.nn.functional.interpolate(segms, (self.h, self.w), mode='bilinear')

        # segms = segms.squeeze(dim=0).squeeze(dim=0)
        # segms[segms >= 0.5] = 1
        # segms[segms < 0.5] = 0
        segms.int()
        caption = annotation['refs']

        return pure_image, image, segms, caption, h, w




    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
        