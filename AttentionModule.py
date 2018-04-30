
# coding: utf-8

# In[259]:


import torch
from torch import nn
from torchvision import models
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
        self.attn_weights = nn.Parameter(torch.FloatTensor(out_channels,in_channels,1,1))
        #self.attn_mask = nn.Parameter(torch.FloatTensor(out_channels,1,1,1), requires_grad=False)
            
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
        #self.attn_mask.data.fill_(1)


# In[262]:


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
        attn_paid = self.weight*self.attn_weights
        return F.conv2d(input, attn_paid, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# In[263]:


def attn_module_test_same_weight():
    in_channels = 4
    out_channels = 2
    kernel_size = 1
    
    c_attn = Conv2d_Attn(in_channels, out_channels, kernel_size, bias=False)
    c = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    
    inp = Variable(torch.ones(2,4,2,2))
    c.weight = c_attn.weight
    
    c_attn_result = c_attn(inp)
    c_result = c(inp)
    
    assert c_attn_result.data.equal(c_result.data)


# In[264]:


def attn_module_test_05_weight():
    in_channels = 4
    out_channels = 2
    kernel_size = 1
    
    c_attn = Conv2d_Attn(in_channels, out_channels, kernel_size, bias=False, attn_init=0.5)
    c = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    
    inp = Variable(torch.ones(2,4,2,2))
    c.weight = c_attn.weight
    
    c_attn_result = c_attn(inp)
    c_result = c(inp)
    
    assert c_attn_result.data.equal(c_result.data/2)


# In[265]:


def attn_module_test_diff_weight():
    in_channels = 4
    out_channels = 2
    kernel_size = 1
    
    c_attn = Conv2d_Attn(in_channels, out_channels, kernel_size, bias=False)
    rand_weights = torch.rand(out_channels,1,1,1)
    c_attn.attn_weights.data = c_attn.attn_weights.data*rand_weights
    
    c = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    
    inp = Variable(torch.ones(2,4,2,2))
    c.weight = c_attn.weight
    
    c_attn_result = c_attn(inp)
    c_result = c(inp)
    
    rand_weights_on_data = rand_weights.permute(1,0,2,3)
    assert np.isclose(c_attn_result.data.sum(),
                      ((c_result.data*rand_weights_on_data).sum()))


# In[266]:


def test_attn_module():
    attn_module_test_same_weight()
    attn_module_test_05_weight()
    attn_module_test_diff_weight()


# In[268]:


if __name__ == '__main__':
    test_attn_module()

