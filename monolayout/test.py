import os
import random
import numpy as np
import pandas as pd
import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.resnet as resnet


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from model import model


image_folder = 'data'
annotation_csv = 'data/annotation.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
labeled_scene_index = np.arange(106, 134)

transform = torchvision.transforms.ToTensor()
unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=unlabeled_scene_index , first_dim='sample', transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=32, shuffle=True, num_workers=0)
# [batch_size, 6(images per sample), 3, H, W]
sample = iter(trainloader).next()
print(sample.shape)
batch_size, imgs, channel, height, width = sample.shape

input = sample.view(batch_size, imgs*channel, height,width)
#Input has [batch=3,channel=6*3, H, W] dimensions;
# input = torch.zeros(3,18,256,306)
# for i in range(3):
#     temp = sample[i,0]
#     for j in range(5):
#         temp = torch.cat([temp,sample[i,j+1]])
#     input[i] = temp
input = input.to(device)
print(input.shape)

encoder = model.Encoder(18,256,306,False)
encoder.to(device)
# encoder = resnet()
y = encoder(input)
decoder = model.Decoder(encoder.resnet_encoder.num_ch_enc)
decoder.to(device)
x = decoder(y.to(device))

print(x.shape)
