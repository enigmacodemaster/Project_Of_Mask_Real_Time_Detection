import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

# GridMask
class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        # d = self.d
        # self.l = int(d*self.ratio+0.5)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask
        return x.view(n, c, h, w)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, use_mosaic=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.use_mosaic = use_mosaic

    def __getitem__(self, index):
        if self.use_mosaic:
            img_path, img, targets = self.load_mosaic(index)

        else:
            # ---------
            #  Image
            # ---------

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            # Extract image as PyTorch tensor
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

            # Handle images with less than three channels
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))

            _, h, w = img.shape
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            # Pad to square resolution
            img, pad = pad_to_square(img, 0)
            _, padded_h, padded_w = img.shape

            # ---------
            #  Label
            # ---------

            label_path = self.label_files[index % len(self.img_files)].rstrip()

            targets = None
            if os.path.exists(label_path):
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                # Returns (x, y, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h

                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        assert img is not None, 'Image Not Found ' + img_path
        _, h, w = img.shape
        r = self.img_size / max(h, w)  # resize image to img_size
        if r > 1:  # always resize down, only resize up if training with augmentation
            img = img.resize((w * r, h * r), Image.ANTIALIAS)  # resize image with high-quality
        return img, (h, w), img.shape[1:], img_path  # img, hw_original, hw_resized

    # mosaic
    def load_mosaic(self, index):
        # loads images in a mosaic
        img_paths = ""
        img4 = None
        targets = None
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        # 3 additional image indices
        indices = [index] + [random.randint(0, len(self.img_files) - 1) for _ in range(3)]
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w), img_path = self.load_image(index)
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            img_paths += img_path
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((img.shape[0], s * 2, s * 2), 114 / 255)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            if os.path.exists(label_path):
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2) + padw
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2) + padh
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2) + padw
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2) + padh
                # Returns (x, y, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / (2 * s)
                boxes[:, 2] = ((y1 + y2) / 2) / (2 * s)
                boxes[:, 3] *= w_factor / (2 * s)
                boxes[:, 4] *= h_factor / (2 * s)
            if i == 0:
                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes
            else:
                more_targets = torch.zeros((len(boxes), 6))
                more_targets[:, 1:] = boxes
                targets = torch.cat([targets, more_targets], dim=0)

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        # img4, labels4 = random_affine(img4, labels4,
        #                               degrees=self.hyp['degrees'],
        #                               translate=self.hyp['translate'],
        #                               scale=self.hyp['scale'],
        #                               shear=self.hyp['shear'],
        #                               border=-s // 2)  # border to remove

        # 中心点坐标在图外面的labels
        labels = targets[torch.where((targets[:, 2] > 0)
                                     & (targets[:, 3] > 0)
                                     & (targets[:, 2] < 1)
                                     & (targets[:, 3] < 1))]

        img4 = torch.from_numpy(img4).float()
        return img_paths, img4, labels
