# -*- coding: utf-8 -*-
"""
@author: Li Xianyang
"""
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class Pedestrian:
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images_list = list(
            sorted(
                os.listdir(os.path.join(
                    root, 'PNGImages'
                ))
            )
        )
        self.masks_list = list(
            sorted(
                os.listdir(os.path.join(
                    root, 'PedMasks'
                ))
            )
        )

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'PNGImages', self.images_list[index])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks_list[index])
        img = Image.open(img_path).convert('RGB')
        #print('img.shape',img.size)
        mask = Image.open(mask_path)
        # mask.show()
        #print('mask.shape',mask.size)

        mask = np.array(mask)
        #print('npmask.shape',mask.shape)

        # #print('mask',mask)
        obj_ids = np.unique(mask)[1:]
        #print('obj_ids',obj_ids)
        num_obj = len(obj_ids)
        masks = mask == obj_ids[:, None, None]  ######????
        #print('type(masks)',type(masks))
        #print('masks.shape', masks.shape)
        boxes = []
        for i in range(num_obj):
            #print('masks[i]',masks[i])
            pos = np.where(masks[i])
            #print('type(pos)',type(pos))
            #print('pos',pos)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append(([xmin, ymin, xmax, ymax]))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_obj), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:,1]) * (boxes[:, 2] - boxes[:,0])
        iscrowd = torch.zeros((num_obj), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target


dataset = Pedestrian('./PennFudanPed')
img,target=dataset[0]
# img.show()

print('img.size()',img.size)
boxes=target['boxes']
labels=target['labels']
masks=target['masks']
image_id=target['image_id']
area=target['area']
iscrowd=target['iscrowd']
print('boxes',boxes)
print('labels',labels)
print('masks.size()',masks.size())
