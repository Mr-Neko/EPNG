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
    def __init__(self, cfg, train) -> None:
        super(PTData, self).__init__()

        self.cfg = cfg
        self.train = train
        
        self.num_box = cfg.DATA.NUM_BOXS
        self.h = cfg.DATA.IMAGE_H
        self.w = cfg.DATA.IMAGE_W
        if self.train:
            self.split = cfg.DATA.TRAIN_SPLIT
        else:
            self.split = cfg.DATA.VAL_SPLIT

        self.transform = transforms.Compose([
			   transforms.Resize((self.h, self.w), interpolation=transforms.InterpolationMode.BICUBIC),
               self._convert_image_to_rgb,
               transforms.ToTensor(),
               transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.annotation_path = osp.join(cfg.DATA.ANNO_PATH, 'e2e_{:s}.json'.format(self.split))
        self.panoptic_ann_path = osp.join(cfg.DATA.ANNO_PATH, 'panoptic_{:s}.json'.format(self.split))
        self.panoptic_path = osp.join(cfg.DATA.ANNO_PATH, 'panoptic_segmentation', self.split)
        self.language_path = osp.join(cfg.DATA.FEATURE_PATH, '{:s}.hdf5'.format(self.split))
        self.image_path = osp.join(cfg.DATA.IMAGE_PATH, self.split)

        annotations = self.load_json(self.annotation_path)
        panoptic_ann = self.load_json(self.panoptic_ann_path)

        self.panoptic_ann_info = {a['image_id']: a for a in panoptic_ann['annotations']}

        self.local_annotations = [
            ln for ln in annotations
            if 1 in ln['noun_vector']
        ]

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __len__(self):
        return len(self.local_annotations)

    def __getitem__(self, indexes):
        h5file = h5py.File(self.language_path, 'r')
        annotation = self.local_annotations[indexes]

        id = annotation['lg_feature']
        
        if id != 7473:
            return 1, 1, 1, 1, 1, id
        gt_noun = torch.from_numpy(h5file[str(id)][()])
        image_id = annotation['image_id']

        pure_image = Image.open(osp.join(self.image_path, annotation['file_name']))
        image = self.transform(pure_image)
        pure_image = np.array(pure_image)

        panoptic_segm = io.imread(osp.join(self.panoptic_path, "{:012d}.png".format(int(image_id))))
        panoptic_segm = (
                panoptic_segm[:, :, 0]
                + panoptic_segm[:, :, 1] * 256
                + panoptic_segm[:, :, 2] * 256 ** 2
            )

        segms = torch.zeros((self.num_box, self.h, self.w))
        noun_index = torch.IntTensor(np.zeros((self.num_box)))
        # centers = torch.zeros((self.num_box, 3))

        for i in annotation['segments']:
            
            ids = i['seg']

            instance = torch.zeros(annotation['image_h'], annotation['image_w'])
            instance[panoptic_segm == int(ids)] = 1

            instance.unsqueeze_(0)
            instance.unsqueeze_(0)
            instance = torch.nn.functional.interpolate(instance, (self.h, self.w), mode='bilinear')
            instance.squeeze_(0)
            instance.squeeze_(0)

            instance[instance >= 0.5] = 1
            instance[instance < 0.5] = 0
            instance.int()

            pos = int(i['noun'])
            segms[pos - 1] = segms[pos - 1] + instance
            noun_index[pos - 1] = pos

        segms[segms > 0] = 1
        segms.requires_grad = False
                # input_ids.requires_grad = False

        return pure_image, image, segms, noun_index, gt_noun, id




    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
        