# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:35:25 2020

@author: 84685
"""

from __future__ import division

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import cv2
import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from matplotlib import pyplot as plt

# args
data_config = "config/custom.data"
multiscale_training = True
n_cpu = 8
batch_size = 16
epochs = 2
img_size = 416

gridmask_active = True
gridmask_d1 = 96
gridmask_d2 = 224
gridmask_rotate = 360
gridmask_offset = False
gridmask_ratio = 0.6
gridmask_mode = 1
gridmask_prob = 0.8
gridmask_st_epochs = 1
apply_mosaic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data configuration
data_config = parse_data_config(data_config)
train_path = data_config["train"]
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

print("train_path: ", train_path)

# Get dataloader
dataset = ListDataset(train_path, augment=True, multiscale=multiscale_training, 
                      use_mosaic=apply_mosaic)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
    collate_fn=dataset.collate_fn
)

gridMask = GridMask(gridmask_d1, 
                    gridmask_d2, 
                    gridmask_rotate, 
                    gridmask_offset, 
                    gridmask_ratio, 
                    gridmask_mode, 
                    gridmask_prob)


# show image data and labels
for epoch in range(epochs):
    for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
        if gridmask_active:
            gridMask.train()
            gridMask.set_prob(epoch, gridmask_st_epochs)
            imgs = gridMask(imgs.cuda())

        for idx in range(imgs.shape[0]):
            img = imgs[idx]
            img = img.permute(1, 2, 0).cpu().numpy()[..., ::-1]
            img = (img * 255).astype(np.uint8)
            img = img.copy()
            img_h, img_w, _ = img.shape

            for target in targets:
                if target[0].item() == idx:
                    pt1 = ((target[2].item() - target[4].item() / 2) * img_w, 
                           (target[3].item() - target[5].item() / 2) * img_h)
                    pt2 = ((target[2].item() + target[4].item() / 2) * img_w, 
                           (target[3].item() + target[5].item() / 2) * img_h)
                    pt1 = (int(pt1[0]), int(pt1[1]))
                    pt2 = (int(pt2[0]), int(pt2[1]))
                    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
            cv2.imshow("preview", img)
            key = cv2.waitKey()
            if key == 27:
                cv2.destroyAllWindows()