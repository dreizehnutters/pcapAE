import torch
from torchvision import models
from itertools import chain
from collections import OrderedDict
import copy
import torchvision
import numpy as np
from skimage import io
from skimage import transform
import pandas as pd
import json
import imp
train_mnist_model = imp.load_source('train_mnist_model', '../../train_mnist_model.py')
from train_mnist_model import Net as ConvNet
LRP = imp.load_source('LRP', '../__init__.py')
from LRP import LRP
from train_mnist_model import preprocessing

model = ConvNet()
model.load_state_dict(torch.load('../../mnist_model.ph'))
model = model.eval()
lrp = LRP(model, 'z_epsilon_rule')
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Lambda(preprocessing),

#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=False)
for num, (image, label) in enumerate(train_loader):
    if num > 1:
        break
    output = lrp.forward(image)
    relevance = lrp(output)

