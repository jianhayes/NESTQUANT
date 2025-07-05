# *
# @file Different utility functions
# Copyright (c) Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao, 
# Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, Minyi Guo
# All rights reserved.
# This file is part of SQuant repository.
#
# SQuant is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SQuant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SQuant repository.  If not, see <http://www.gnu.org/licenses/>.
# *
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from nestquant.quant_affine import *
import warnings
try:
    from quant_cuda import rounding_loop as SQuant_func
except ImportError:
    warnings.warn("CUDA-based SQuant is not installed! PyTorch-based SQuant will lead to a prolonged quantization process.")
    from nestquant.squant_function import SQuant_func

import copy
from nestquant.pack_bits_tensors import pack_simulated_bits_tensor, unpack_bits_tensor
from nestquant.quant_modules import LinearQuantizer, Conv2dQuantizer, TensorQuantizer
from nestquant.quant_utils import infer_nestquant_args, infer_quant_args

logger = logging.getLogger(__name__)
import argparse
import numpy as np


class NestLinearQuantizer(nn.Module):
    def __init__(self, wbit=8, abit=8, nestbit=4, nest=True, packbits=True):
        super(NestLinearQuantizer, self).__init__()
        self.quant_input = None

        self.register_buffer('wbit', torch.tensor(1))
        self.wbit = torch.tensor(wbit)
        self.register_buffer('abit', torch.tensor(1))
        self.abit = torch.tensor(abit)

        self.nest = nest
        self.packbits = packbits
        self.scale = None
        self.zero_point = None

        self.register_buffer('nestbit', torch.tensor(1))
        self.nestbit = torch.tensor(nestbit)
        self.shiftbit = self.wbit - self.nestbit
        assert self.shiftbit > 0

        self.wshape = None


    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quant_input = linear.quant_input

        self.nest = linear.quant_weight.nest
        self.wbit = linear.quant_weight.bit
        self.nestbit = torch.tensor(linear.quant_weight.nestbit)
        self.shiftbit = self.wbit - self.nestbit

        if self.packbits:
            self.wshape = linear.quant_weight.quant_high_int.shape
            quant_high_int = pack_simulated_bits_tensor(linear.quant_weight.quant_high_int.floor().long(),
                                                        k=self.nestbit, packbits=self.packbits)
            quant_low_int = pack_simulated_bits_tensor(linear.quant_weight.quant_low_int.floor().long(),
                                                       k=self.shiftbit + 1, packbits=self.packbits)
            self.register_buffer('quant_high_int', quant_high_int)
            self.register_buffer('quant_low_int', quant_low_int)
        else:
            self.quant_high_int = nn.Parameter(linear.quant_weight.quant_high_int)
            self.quant_low_int = nn.Parameter(linear.quant_weight.quant_low_int)
            self.quant_high_int.requires_grad = False
            self.quant_low_int.requires_grad = False

        self.scale = nn.Parameter(linear.quant_weight.scale)
        self.zero_point = nn.Parameter(linear.quant_weight.zero_point)
        self.scale_nest = nn.Parameter(linear.quant_weight.scale_nest)
        self.zero_point_nest = nn.Parameter(linear.quant_weight.zero_point_nest)

        self.scale.requires_grad = False
        self.zero_point.requires_grad = False
        self.scale_nest.requires_grad = False
        self.zero_point_nest.requires_grad = False

        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input): 
        input = self.quant_input(input)
        with torch.no_grad():
            if self.packbits:
                quant_high_int = self.quant_high_int.clone()
                quant_high_int = unpack_bits_tensor(quant_high_int, k=self.nestbit, shape=self.wshape)
            else:
                quant_high_int = self.quant_high_int
            if self.nest:
                weight = linear_dequantize((quant_high_int).to(torch.float32), self.scale_nest,
                                           self.zero_point_nest, inplace=False)
            else:
                if self.packbits:
                    quant_low_int = self.quant_low_int.clone()
                    quant_low_int = unpack_bits_tensor(quant_low_int, k=self.shiftbit + 1, shape=self.wshape)
                else:
                    quant_low_int = self.quant_low_int
                n = 2 ** (self.wbit - 1)
                quant_high_int = torch.clamp(quant_high_int * 2 ** self.shiftbit + quant_low_int, -n, n - 1)
                weight = linear_dequantize(quant_high_int.to(torch.float32), self.scale, self.zero_point, inplace=False)
        # logger.info(input.unique().numel(), self.quant_input.name)
        return F.linear(input, weight, self.bias)


class NestConv2dQuantizer(nn.Module):
    def __init__(self, wbit=8, abit=8, nestbit=4, nest=True, packbits=True):
        super(NestConv2dQuantizer, self).__init__()
        self.quant_input = None

        self.register_buffer('wbit', torch.tensor(1))
        self.wbit = torch.tensor(wbit)
        self.register_buffer('abit', torch.tensor(1))
        self.abit = torch.tensor(abit)

        self.nest = nest
        self.packbits = packbits
        self.scale = None
        self.zero_point = None

        self.register_buffer('nestbit', torch.tensor(1))
        self.nestbit = torch.tensor(nestbit)
        self.shiftbit = self.wbit - self.nestbit
        assert self.shiftbit > 0

        self.wshape = None

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.quant_input = conv.quant_input

        self.nest = conv.quant_weight.nest
        self.wbit = conv.quant_weight.bit
        self.nestbit = torch.tensor(conv.quant_weight.nestbit)
        self.shiftbit = self.wbit - self.nestbit

        if self.packbits:
            self.wshape = conv.quant_weight.quant_high_int.shape
            quant_high_int = pack_simulated_bits_tensor(conv.quant_weight.quant_high_int.floor().long(),
                                                        k=self.nestbit, packbits=self.packbits)
            quant_low_int = pack_simulated_bits_tensor(conv.quant_weight.quant_low_int.floor().long(),
                                                       k=self.shiftbit + 1, packbits=self.packbits)
            self.register_buffer('quant_high_int', quant_high_int)
            self.register_buffer('quant_low_int', quant_low_int)
        else:
            self.quant_high_int = nn.Parameter(conv.quant_weight.quant_high_int)
            self.quant_low_int = nn.Parameter(conv.quant_weight.quant_low_int)
            self.quant_high_int.requires_grad = False
            self.quant_low_int.requires_grad = False

        self.scale = nn.Parameter(conv.quant_weight.scale)
        self.zero_point = nn.Parameter(conv.quant_weight.zero_point)
        self.scale_nest = nn.Parameter(conv.quant_weight.scale_nest)
        self.zero_point_nest = nn.Parameter(conv.quant_weight.zero_point_nest)

        self.scale.requires_grad = False
        self.zero_point.requires_grad = False
        self.scale_nest.requires_grad = False
        self.zero_point_nest.requires_grad = False

        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def set_nest(self, Nest):
        self.nest = Nest

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        input = self.quant_input(input)

        with torch.no_grad():
            if self.packbits:
                quant_high_int = self.quant_high_int.clone()
                quant_high_int = unpack_bits_tensor(quant_high_int, k=self.nestbit, shape=self.wshape)
            else:
                quant_high_int = self.quant_high_int
            if self.nest:
                weight = linear_dequantize((quant_high_int).to(torch.float32), self.scale_nest,
                                           self.zero_point_nest, inplace=False)
            else:
                if self.packbits:
                    quant_low_int = self.quant_low_int.clone()
                    quant_low_int = unpack_bits_tensor(quant_low_int, k=self.shiftbit + 1, shape=self.wshape)
                else:
                    quant_low_int = self.quant_low_int
                n = 2 ** (self.wbit - 1)
                quant_high_int = torch.clamp(quant_high_int * 2 ** self.shiftbit + quant_low_int, -n, n-1)
                weight = linear_dequantize(quant_high_int.to(torch.float32), self.scale, self.zero_point, inplace=False)
        # logger.info(input.unique().numel(), self.quant_input.name, "input")
        return self._conv_forward(input, weight)


def nest_quantized_model(qmodel):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize convolutional and linear layers to 8-bit
    if type(qmodel) == Conv2dQuantizer:
        # print(model)
        quant_mod = NestConv2dQuantizer(**infer_nestquant_args)
        quant_mod.set_param(qmodel)
        return quant_mod
    elif type(qmodel) == LinearQuantizer:
        # print(model)
        quant_mod = NestLinearQuantizer(**infer_nestquant_args)
        quant_mod.set_param(qmodel)
        return quant_mod

    # recursively use the quantized module to replace the single-precision module
    elif type(qmodel) == nn.Sequential:
        mods = []
        for n, m in qmodel.named_children():
            mods.append(nest_quantized_model(m))
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(qmodel)
        for attr in dir(qmodel):
            mod = getattr(qmodel, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, nest_quantized_model(mod))
        return q_model


class InferLinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, wbit=8, abit=8, packbits=True):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(InferLinearQuantizer, self).__init__()
        self.quant_input = None

        self.register_buffer('wbit', torch.tensor(1))
        self.wbit = torch.tensor(wbit)
        self.register_buffer('abit', torch.tensor(1))
        self.abit = torch.tensor(abit)

        self.packbits = packbits
        self.scale = None
        self.zero_point = None

        self.wshape = None

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quant_input = linear.quant_input

        self.wbit = linear.quant_weight.bit

        if self.packbits:
            self.wshape = linear.quant_weight.quant_int.shape
            quant_int = pack_simulated_bits_tensor(linear.quant_weight.quant_int.floor().long(),
                                                   k=self.wbit, packbits=self.packbits)
            self.register_buffer('quant_int', quant_int)
        else:
            self.quant_int = nn.Parameter(linear.quant_weight.quant_int)
            self.quant_int.requires_grad = False

        self.scale = nn.Parameter(linear.quant_weight.scale)
        self.zero_point = nn.Parameter(linear.quant_weight.zero_point)

        self.scale.requires_grad = False
        self.zero_point.requires_grad = False

        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input):
        input = self.quant_input(input)
        with torch.no_grad():
            if self.packbits:
                quant_int = self.quant_int.clone()
                quant_int = unpack_bits_tensor(quant_int, k=self.wbit, shape=self.wshape)
            else:
                quant_int = self.quant_int
            weight = linear_dequantize(quant_int.to(torch.float32), self.scale, self.zero_point, inplace=False)
        return F.linear(input, weight, self.bias)


class InferConv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, wbit=8, abit=8, packbits=True):
        super(InferConv2dQuantizer, self).__init__()
        self.quant_input = None

        self.register_buffer('wbit', torch.tensor(1))
        self.wbit = torch.tensor(wbit)
        self.register_buffer('abit', torch.tensor(1))
        self.abit = torch.tensor(abit)

        self.packbits = packbits
        self.scale = None
        self.zero_point = None

        self.wshape = None

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.quant_input = conv.quant_input

        self.wbit = conv.quant_weight.bit

        if self.packbits:
            self.wshape = conv.quant_weight.quant_int.shape
            quant_int = pack_simulated_bits_tensor(conv.quant_weight.quant_int.floor().long(),
                                                   k=self.wbit, packbits=self.packbits)
            self.register_buffer('quant_int', quant_int)
        else:
            self.quant_int = nn.Parameter(conv.quant_weight.quant_int)
            self.quant_int.requires_grad = False

        self.scale = nn.Parameter(conv.quant_weight.scale)
        self.zero_point = nn.Parameter(conv.quant_weight.zero_point)

        self.scale.requires_grad = False
        self.zero_point.requires_grad = False

        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        input = self.quant_input(input)
        with torch.no_grad():
            if self.packbits:
                quant_int = self.quant_int.clone()
                quant_int = unpack_bits_tensor(quant_int, k=self.wbit, shape=self.wshape)
            else:
                quant_int = self.quant_int
            weight = linear_dequantize(quant_int.to(torch.float32), self.scale, self.zero_point, inplace=False)
        return self._conv_forward(input, weight)


def infer_quantized_model(qmodel):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize convolutional and linear layers to 8-bit
    if type(qmodel) == Conv2dQuantizer:
        # print(model)
        quant_mod = InferConv2dQuantizer(**infer_quant_args)
        quant_mod.set_param(qmodel)
        return quant_mod
    elif type(qmodel) == LinearQuantizer:
        # print(model)
        quant_mod = InferLinearQuantizer(**infer_quant_args)
        quant_mod.set_param(qmodel)
        return quant_mod

    # recursively use the quantized module to replace the single-precision module
    elif type(qmodel) == nn.Sequential:
        mods = []
        for n, m in qmodel.named_children():
            mods.append(infer_quantized_model(m))
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(qmodel)
        for attr in dir(qmodel):
            mod = getattr(qmodel, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, infer_quantized_model(mod))
        return q_model

