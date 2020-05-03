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

import model

image_folder = 'data'
annotation_csv = 'data/annotation.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)

transform = torchvision.transforms.ToTensor()
unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=unlabeled_scene_index , first_dim='sample', transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=3, shuffle=True, num_workers=0)
sample = iter(trainloader).next()

#Input has dimension: [batch=3,channel=6*3, H, W]
input = torch.zeros(3,18,256,306)
for i in range(3):
    temp = sample[i,0]
    for j in range(5):
        temp = torch.cat([temp,sample[i,j+1]])
    input[i] = temp
input = input.to(device)
print(input.shape)

encoder = model.Encoder(18,256,306,False)
encoder.to(device)
decoder = model.Decoder(128)
decoder.to(device)

#Encoder(Resnet) Output:
y = encoder(input)
print("Encoder output dim:")
print(y.shape)
#Decoder Output:
z = decoder(y)
print("Decoder output size:")
print(z.shape)

#Plot "Road Map" from decoder:
z = torch.zeros(1,20,20)
z[0,0,:] = -torch.ones(20)
ego = z.detach().cpu()
fig, ax = plt.subplots()
ax.imshow(ego[0].squeeze(), cmap='binary')
plt.show()

