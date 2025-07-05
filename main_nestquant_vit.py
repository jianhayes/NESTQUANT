""" Post-Training Integer-Nesting Quantization for On-Device DNN
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: xiejianhang@bjtu.edu.cn
"""

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nestquant.dataloader import *
from nestquant.quant_utils import *
from nestquant.quant_model import *
from nestquant.quantized_model import *
from timm.models import poolformer_m48, swin_large_patch4_window7_224, swin_base_patch4_window7_224, \
    vit_large_patch14_224, vit_large_patch16_224, vit_base_patch16_224, deit_base_patch16_224

import numpy as np
import random


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # if you are using multi-GPU DataParallel
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch version >= 1.7
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


@torch.no_grad()
def test(quantized_model_, testloader):
    quantized_model_.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = quantized_model_(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        _, predicted_5 = outputs.topk(5, 1, True, True)
        predicted_5 = predicted_5.t()
        correct_ = predicted_5.eq(targets.view(1, -1).expand_as(predicted_5))
        correct_5 += correct_[:5].reshape(-1).float().sum(0, keepdim=True).item()
        if batch_idx % 10 == 0 or batch_idx == len(testloader) - 1:
            print('test: [batch: %d/%d ] | Loss: %.3f | Acc: %.3f%% (%d/%d)/ %.3f%% (%d/%d)'
                        % (batch_idx, len(testloader), test_loss / (batch_idx + 1), 100. * correct / total, correct,
                           total, 100. * correct_5 / total, correct_5, total))

        ave_loss = test_loss / total

    acc = 100. * correct / total

    print("Final accuracy: %.3f" % acc)


def main():
    parser = argparse.ArgumentParser(description='NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN')
    parser.add_argument('--gpu', type=str, default="0", help='GPU id')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--dataset_path', default='/Datasets/ILSVRC2012/', type=str, help='dataset path')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size num')
    parser.add_argument('--mode', default='squant_', type=str, help='quantizer mode')
    parser.add_argument('--wbit', '-wb', default='8', type=int, help='weight bit width')
    parser.add_argument('--abit', '-ab', default='8', type=int, help='activation bit width')
    parser.add_argument('--model', default='resnet18', type=str, help='model name')
    parser.add_argument('--disable_quant', "-dq", default=False, action='store_true', help='disable quant')
    parser.add_argument('--disable_activation_quant', "-daq", default=False, action='store_true', help='quant_activation')
    parser.add_argument('--percent', '-p', default='100', type=int, help='percent')
    parser.add_argument('--constraint_radius', '-cr', default='1.0', type=float, help='Constraint radius')
    parser.add_argument('--packed_element', '-pe', default='1', type=int, help='Packed Element radius')
    parser.add_argument('--sigma', '-s', default='0', type=float, help='Init activation range with Batchnorm Sigma')
    parser.add_argument('--seed', default=0, type=int, help='fixed random seed')
    parser.add_argument('--savepath', default='./nestquant_vit/qmodel/', type=str, help='save model')
    parser.add_argument('--nestbit', '-nb', default='4', type=int, help='nesting weight bit width')
    parser.add_argument('--nest', action='store_true', help='part-bit inference if true; else full-bit inference')
    parser.add_argument('--adaround', action='store_true', help='Adative Rounding for nesting high weight')
    parser.add_argument('--packbits', action='store_true', help='Using INT64 to packbits weight tensor')
    parser.add_argument('--nest', action='store_true', help='save packbits nestquant quantized model')
    parser.add_argument('--squant', action='store_true', help='save packbits squant quantized model')
    parser.add_argument('--eval_nestquant', action='store_true', help='nestquant model ImageNet-1K val Evaluation')
    parser.add_argument('--eval_squant', action='store_true', help='squant model ImageNet-1K val Evaluation')
    args = parser.parse_args()

    print("random seed is {}".format(args.seed))
    init_seed(args.seed)

    assert args.model in ('poolformer_m48', 'swin_large_patch4_window7_224',
                          'swin_base_patch4_window7_224',
                          'vit_large_patch14_224', 'vit_large_patch16_224',
                          'vit_base_patch16_224', 'deit_base_patch16_224')

    if args.model == 'poolformer_m48':
        model = poolformer_m48(pretrained=True)
    elif args.model == 'swin_large_patch4_window7_224':
        model = swin_large_patch4_window7_224(pretrained=True)
    elif args.model == 'swin_base_patch4_window7_224':
        model = swin_base_patch4_window7_224(pretrained=True)
    elif args.model == 'vit_large_patch14_224':
        model = vit_large_patch14_224(pretrained=True)
    elif args.model == 'vit_large_patch16_224':
        model = vit_large_patch16_224(pretrained=True)
    elif args.model == 'vit_base_patch16_224':
        model = vit_base_patch16_224(pretrained=True)
    elif args.model == 'deit_base_patch16_224':
        model = deit_base_patch16_224(pretrained=True)
    else:
        raise NotImplementedError

    rand_input = torch.rand([args.batch_size, 3, 224, 224], dtype=torch.float, requires_grad=False).cuda()

    ### Set Quantizer
    set_quantizer(args)
    quant_model = quantize_model(model)

    if args.disable_quant:
        disable_quantization(quant_model)
    else:
        enable_quantization(quant_model)

    if args.disable_activation_quant:
        disable_input_quantization(quant_model)

    set_first_last_layer(quant_model)

    quant_model.cuda()

    quant_model.eval()
    quant_model(rand_input)

    quant_model.cpu()
    # print(quant_model)

    if args.nestquant:
        set_infernestquant_quantizer(args)
        quantized_DNN_model = nest_quantized_model(quant_model)

        if not os.path.exists(args.savepath):
            os.makedirs(args.savepath)
        savepath = args.savepath + "{}_nestquant_W{}A{}Nest{}.pth".format(args.model, args.wbit, args.abit, args.nestbit)

        torch.save(quantized_DNN_model.state_dict(), savepath)

        quantized_DNN_model.load_state_dict(torch.load(savepath))

        if args.eval_nestquant:
            quantized_DNN_model.cuda()
            # Load validation data
            print('==> Preparing data..')
            testloader = getTestData(args.dataset,
                                     batch_size=args.batch_size,
                                     path=args.dataset_path,
                                     for_inception=args.model.startswith('inception'))

            print('==> Preparing data.. for EVAL2: NestQuant model ImageNet-1K val Evaluation')

            ## Validation
            test(quantized_DNN_model, testloader)

    if args.squant:
        set_inferquant_quantizer(args)
        quantized_DNN_model_squant = infer_quantized_model(quant_model)

        savepath_squant = args.savepath + "{}_squant_W{}A{}.pth".format(args.model, args.wbit, args.abit)

        torch.save(quantized_DNN_model_squant.state_dict(), savepath_squant)

        quantized_DNN_model_squant.load_state_dict(torch.load(savepath_squant))

        if args.eval_squant:
            quantized_DNN_model_squant.cuda()

            # Load validation data
            print('==> Preparing data..')
            testloader = getTestData(args.dataset,
                                     batch_size=args.batch_size,
                                     path=args.dataset_path,
                                     for_inception=args.model.startswith('inception'))

            print('==> Preparing data.. for SQuant model ImageNet-1K val Evaluation')

            ## Validation
            test(quantized_DNN_model_squant, testloader)


if __name__ == '__main__':
    main()