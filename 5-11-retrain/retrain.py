import argparse

parser = argparse.ArgumentParser(description='Run Training Scripts For Attention Project. ')

parser.add_argument('--use-attn', type=str,
                    help='{True, False} Use attention in training or not')
parser.add_argument('--train-phase', type=str,
                    help='{f,c,b}xk, alternate among fc, attention, and batchnorm. e.g. faaba')
parser.add_argument('--attn-loss', type=str,
                    help='{l1,l2}, attention loss regularization')
parser.add_argument('--base-net', type=str,
                    help='{resnet50/101}, base network')
parser.add_argument('--attn-shape', type=str,
                    help='{out, full}, Out channel attention or both out&in channel attention')

args = parser.parse_args()

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import re

use_attn = eval(args.use_attn)
assert use_attn in [True, False]

attn_loss = None
attn_shape = None
if use_attn:
    attn_loss = args.attn_loss.lower()
    assert attn_loss in ['l2', 'l1']

    attn_shape = args.attn_shape.lower()
    assert attn_shape in ['out', 'full']

base_net = args.base_net.lower()
assert 'resnet' in base_net
pretrained = eval('models.{}(pretrained=True)'.format(base_net))

train_phase = args.train_phase.lower()
if not use_attn:
    assert set(train_phase) == {'f'}, "If not using attention, train_phase can only be fffff"
else:
    assert len(set(train_phase).difference({'f', 'a', 'b'})) == 0

config = {
    'use_attn': use_attn,
    'attn_loss': attn_loss,
    'attn_shape': attn_shape,
    'base_net': base_net,
    'train_phase': train_phase,
}

import os
from datetime import datetime

time = str(datetime.now()).replace(' ', '_').replace('-', '_')
file_name = time + '_' + '_'.join(list(map(str, [train_phase, base_net, attn_loss, attn_shape]))) + '.log'
os.makedirs('logs', exist_ok=True)

import logging

logging.basicConfig(format='%(asctime)s : %(message)s',
                    filename='logs/{}'.format(file_name),
                    level=logging.INFO,
                    filemode='w'
                    )


def print_log(*string):
    print(*string)
    logging.info(str(string))


print_log("Logging File @ logs/{}".format(file_name))
print_log("Config:")
print_log(config)

print_log("Initialize Network ...")

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

import numpy as np

import collections
from itertools import repeat
import math


# In[260]:


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


# In[261]:


class _ConvNd_Attn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, attn_init):
        super(_ConvNd_Attn, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.attn_init = attn_init
        if attn_shape == 'full':
            self.attn_weights = nn.Parameter(torch.FloatTensor(out_channels, in_channels, 1, 1))
        else:
            self.attn_weights = nn.Parameter(torch.FloatTensor(out_channels, 1, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.attn_weights.data.fill_(self.attn_init)


class Conv2d_Attn(_ConvNd_Attn):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, attn_init=1.0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_Attn, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, attn_init=attn_init)

    def forward(self, input):
        attn_paid = self.weight * self.attn_weights
        return F.conv2d(input, attn_paid, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


if use_attn:
    nn.Conv2d = Conv2d_Attn
    resnet_attn = eval('models.{}()'.format(base_net))
    resnet_attn.load_state_dict(pretrained.state_dict(), strict=False)
else:
    resnet_attn = pretrained


# This block turns 'layer1.0.downsample.0.weight' to 'layer1[0].downsample[0].weight'
def get_formatted_keys(network_name):
    param_keys = list(eval(network_name).state_dict().keys())
    formatted_keys = []
    for k in param_keys:
        found = re.findall(r'\.[\d]{1,2}\.', k)
        if len(found):
            for f in found:
                k = k.replace(f, '[{}].'.format(f.strip('.')))
        formatted_keys.append(k)
    return formatted_keys


# This block turn off gradient up for all params except attn_weights
def turn_off_grad_except(network_name, lst=[]):
    formatted_keys = get_formatted_keys(network_name)
    for k in formatted_keys:
        obj = eval(f'{network_name}.' + k)
        for kw in lst:
            if not kw in k:
                obj.requires_grad = False
            else:
                obj.requires_grad = True


batch_size = 32

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize])


def get_loader(dirname, batch_size=32):
    trainset = torchvision.datasets.ImageFolder(root=f'../data/{dirname}', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3)
    return trainloader, len(trainset)


def score_batch(inp, label, top, network):
    _, idx = eval(network)(Variable(inp).cuda()).topk(top)
    lab = Variable(label).cuda()
    lab_expand = lab.unsqueeze(1).expand_as(idx)
    return int((idx == lab_expand).sum())

def compute_correct(out, label, top):
    _, idx = out.topk(top)
    lab_expand = label.unsqueeze(1).expand_as(idx)
    return int((idx == lab_expand).sum())


def score_data(data_dir, network_name):
    trainloader, train_total = get_loader(data_dir, batch_size=64)
    top3_count = 0
    top1_count = 0
    for inp, label in iter(trainloader):
        top1_count += score_batch(inp, label, 1, network_name)
        top3_count += score_batch(inp, label, 3, network_name)
    print_log({
        f'{data_dir}_top1': top1_count / train_total,
        f'{data_dir}_top3': top3_count / train_total
    })


def score(network_name, train=True, val=True, batch_size=32):
    if train:
        score_data('train', network_name)
    if val:
        score_data('val', network_name)


def get_loss_opt():
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_attn.parameters()))
    return cls_criterion, optimizer


resnet_attn = resnet_attn.eval().cuda()

trainset = torchvision.datasets.ImageFolder(root=f'../data/train', transform=transform)

print_every = 30
total_imgs = len(trainset)

total_attn_params = 0
for k in get_formatted_keys('resnet_attn'):
    obj = eval('resnet_attn.' + k)
    if 'attn_weights' in k:
        total_attn_params += np.prod(obj.shape)
print_log("Total number of attention parameters", total_attn_params)

_lambda = 1  # set default


def get_params_objs(name, net):
    res = []
    for k in get_formatted_keys(net):
        obj = eval(f'{net}.' + k)
        if name in k:
            res.append(obj)
    return res


def compute_attn_loss(n_params=total_attn_params, name='resnet_attn'):
    attns = get_params_objs('attn_weights', name)
    if attn_loss == 'l2':
        penality = sum([torch.pow(t - 1, 2).mean() for t in attns])
        return _lambda * (-penality)
    else:
        penalty = sum([torch.norm(t, p=1) for t in attns]) / float(total_attn_params)
        return _lambda * (penalty)


def train_one(use_attn=use_attn):
    trainloader, train_total = get_loader('train')

    running_cls_loss = 0.0
    running_attn_loss = 0.0
    top1_count = 0
    top3_count = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = resnet_attn(inputs)
        cls_loss = cls_criterion(outputs, labels)
        loss = cls_loss

        if use_attn:
            attn_loss_iter = compute_attn_loss(name='resnet_attn')
            loss += attn_loss_iter
            running_attn_loss += attn_loss_iter.data[0]

        loss.backward()
        optimizer.step()

        running_cls_loss += cls_loss.data[0]
        
        top1_count += compute_correct(outputs, labels, 1)
        top3_count += compute_correct(outputs, labels, 3)

        if (i + 1) % print_every == 0:
            print_log(
                '{} iter, {} epoch, cls loss: {}, attn loss: {} '.format(
                    i + 1,
                    i * batch_size / total_imgs,
                    running_cls_loss / print_every,
                    running_attn_loss / print_every))
            running_cls_loss = 0.0
            running_attn_loss = 0.0

    print_log("Begin Scoring")
    print_log({
        f'train_top1': top1_count / train_total,
        f'train_top3': top3_count / train_total
    })
    score('resnet_attn', batch_size=64, train=False)
    print_log("Done Scoring")


print_log("Begin Training")


def parse_layer(char):
    if char == 'f':
        resnet_attn.eval()
        return 'fc'
    elif char == 'b':
        resnet_attn.train()
        return 'bn'
    else:
        resnet_attn.eval()
        return 'attn_weights'


for a, l in enumerate(train_phase):
    layer = parse_layer(l)
    print_log(f'-----Iter {a}, Training {layer}-----')
    turn_off_grad_except('resnet_attn', [layer])
    cls_criterion, optimizer = get_loss_opt()
    train_one()

print_log('All phases Done')

print_log('Saving model')

os.makedirs('saved_models', exist_ok=True)

torch.save(resnet_attn, 'saved_models/' + file_name.replace('log', 'pth'))
