from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, label_smoothing=False, eps=0.05):
    """
    generate masks & t·
    :param pred_boxes: 预测的bbox(0, 13) (b, num_anchor, grid_size, grid_size, 4) -> (b, 3, 13, 13, 4)
    :param pred_cls: 预测的类别概率(0, 1) (b, num_anchor, grid_size, grid_size, n_classes) -> (b, 3, 13, 13, 80)
    :param target: label(0, 1) (n_boxes, 6), 第二个维度有6个值，分别为: box所属图片在本batch中的index， 类别index， xc, yc, w, h
    :param anchors: tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]) (num_anchor, 2) -> (3, 2-)->aw, ah
    :param ignore_thres: hard coded, 0.5
    :return: masks & t·
    """

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # batch size
    nA = pred_boxes.size(1)  # anchor size: 3
    nC = pred_cls.size(-1)  # class size: 80
    nG = pred_boxes.size(2)  # grid size: 13

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)  # (b, 3, 13, 13)    # mostly candidates are noobj
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)  # (b, 3, 13, 13)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)  # (b, 3, 13, 13, 80)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    # 仅依靠w&h 计算target box和anchor box的交并比， (num_anchor, n_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])

    best_ious, best_n = ious.max(0)  # 最大iou, 与target box交并比最大的anchor的index // [n_boxes], [n_boxes]

    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1  # 物体中心点落在的那个cell中的，与target object iou最大的那个3个anchor中的那1个，被设成1
    noobj_mask[b, best_n, gj, gi] = 0  # 其相应的noobj_mask被设成0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # label smoothing
    if label_smoothing:
        tcls = tcls * (1.0 - eps) + (1 - tcls) * eps / nC

    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    # label smoothing
    if label_smoothing:
        tconf = tconf * (1.0 - eps) + (1 - tconf) * eps
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
