
# coding: utf-8

# In[1]:


from AttentionModule import Conv2d_Attn

import torch
from torch import nn
from torchvision import models, datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import re
import numpy as np


# In[1]:


resnet_pretrained = models.resnet50(pretrained=True)
nn.Conv2d = Conv2d_Attn
resnet_attn = models.resnet50()
resnet_attn.load_state_dict(resnet_pretrained.state_dict(), strict=False)


# In[6]:


# This block turns 'layer1.0.downsample.0.weight' to 'layer1[0].downsample[0].weight'
param_keys = list(resnet_attn.state_dict().keys())
formatted_keys = []
for k in param_keys:
    found = re.findall(r'\.[\d]{1,2}\.', k)
    if len(found):
        for f in found:
            k = k.replace(f, '[{}].'.format(f.strip('.')))
    formatted_keys.append(k)


# In[7]:


# This block turn off gradient up for all params except attn_weights
def turn_off_grad_except(lst=[]):
    for k in formatted_keys:
        obj = eval('resnet_attn.'+k)
        for kw in lst:
            if not kw in k:
                obj.requires_grad = False
            else:
                obj.requires_grad = True


# In[8]:


resnet_attn.fc = nn.Linear(resnet_attn.fc.in_features, 144)


# Start training

# In[9]:


batch_size = 32


# In[10]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize])

trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


# In[11]:


total_imgs = len(trainset.imgs)


# In[12]:


resnet_attn = resnet_attn.cuda()


# In[13]:


total_attn_params = 0
for k in formatted_keys:
    obj = eval('resnet_attn.'+k)
    if 'attn_weights' in k:
        total_attn_params += np.prod(obj.shape)
print("Total number of attention parameters", total_attn_params)


# We want the attention parameters to diverge from 1, therefore we penalize element-wise square loss as $\lambda (1 \times \text{# params} - (x - 1)^2)$
# 
# But this is too big a number,
# let's try: 
# $- (x - 1)^2$ for now

# In[14]:


_lambda = 1e-2 #set default


# In[15]:


def get_params_objs(name, net='resnet_attn'):
    res = []
    for k in formatted_keys:
        obj = eval(f'{net}.'+k)
        if name in k:
            res.append(obj)
    return res


# In[16]:


def compute_attn_loss(n_params=26560):
    attns = get_params_objs('attn_weights')
    penality = sum([torch.pow(t - 1,2).mean() for t in attns])
    return _lambda*(- penality)


# In[17]:


print_every = 5


# In[18]:


def train_one_epoch(add_attn=True):
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_attn.parameters()))
    
    running_loss = 0.0
    running_attn_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = resnet_attn(inputs)
        loss = cls_criterion(outputs, labels)
        attn_loss = compute_attn_loss()
        if add_attn:
            loss += attn_loss

        loss.backward()
        optimizer.step()


        running_loss += loss.data[0]
        running_attn_loss += attn_loss.data[0]

        if i % print_every == 0:
            print('[%5d] iter, [%2f] epoch, avg loss: %.3f, attn_loss: %.5f ' %
                  (i + 1, i*batch_size/total_imgs, running_loss/print_every, running_attn_loss/print_every))
            running_loss = 0.0
            running_attn_loss = 0.0


# In[19]:


from tqdm import tqdm
def score(net=resnet_attn, batch_size=batch_size):
    trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    valset = torchvision.datasets.ImageFolder(root='./data/val', transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    train_correct = 0
    val_correct = 0
    
    for inp, label in tqdm(iter(trainloader)):
        _, idx = net(Variable(inp).cuda()).max(1)
        train_correct += int(sum(idx.cpu().data == label))
    
    for inp, label in tqdm(iter(valloader)):
        _, idx = net(Variable(inp).cuda()).max(1)
        val_correct += int(sum(idx.cpu().data == label))
    
    return {
        'train_accu': train_correct/len(trainset),
        'val_accu': val_correct/len(valset)
    }


# Train a fresh fc layer. 
# `turn_off_grad_except([])` turns off grads for all weights but the fc layer

# In[20]:


turn_off_grad_except(['fc'])
resnet_attn.eval() # Turn on batchnorm
train_one_epoch(add_attn=False)


# In[21]:


#score(batch_size=32)


# In[22]:


turn_off_grad_except(['attn_weights'])
resnet_attn.eval() # Turn on batchnorm
_lambda = 1
train_one_epoch(add_attn=True)


# In[23]:


#score(batch_size=32)


# In[17]:


arr = resnet_attn.conv1.attn_weights.data.cpu().numpy().squeeze()


# In[18]:


from datetime import datetime


# In[28]:


def get_time():
    return re.sub(r'[^0-9]', '_', str(datetime.now()))


# In[29]:


arr.dump('ckpt/conv1_attn_{}.npx'.format(get_time()))
print("Dump Complete")
