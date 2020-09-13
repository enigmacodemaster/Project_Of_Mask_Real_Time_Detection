from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# Depthwise Convolution的实现
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def create_modules(module_defs, act_type=0, mobile_yolo=False):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # module_defs = [{"type":"net", "channels":3, ...},                     # each elemnt is a layer block (dtype=dict)
    #                {"type":"convolutional", "batch_normalize":1, ...},
    #                ...]
    hyperparams = module_defs.pop(0)  # [net]的整体参数
    output_filters = [int(hyperparams["channels"])]  # 3: 最初。因为是rgb 3通道
    module_list = nn.ModuleList()  # 存储每一大层，如conv层: 包括conv-bn-leaky relu等

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            # 根据参数选择是否使用depthwise Convolution
            if mobile_yolo:
                modules.add_module(
                    f"conv_{module_i}",
                    DepthwiseConv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(module_def["stride"]),
                        padding=pad,
                        bias=not bn,
                    ),
                )
            else:
                modules.add_module(
                    f"conv_{module_i}",
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(module_def["stride"]),
                        padding=pad,
                        bias=not bn,
                    ),
                )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                if int(act_type) == 0:
                    print("Adding LeakyReLU")
                    modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
                elif int(act_type) == 1:
                    print("Adding Swish")
                    modules.add_module(f"swish_{module_i}", Swish())
                elif int(act_type) == 2:
                    print("Adding Mish")
                    modules.add_module(f"mish_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])  # channel个数相加，对应concat
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            # # mask: 6,7,8 / 3,4,5 / 0,1,2 <=> 小/中/大 feature map <=> 大/中/小 物体
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            # for mask: 6,7,8
            # [(116, 90), (156, 198), (373, 326)]
            num_classes = int(module_def["classes"])  # 80
            img_size = int(hyperparams["height"])   # 416
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


