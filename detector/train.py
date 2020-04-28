import os
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize']=[5,5]
matplotlib.rcParams['figure.dpi']=200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

from model.retinanet import resnet18
from model.discriminator import FCDiscriminator

image_folder = 'data'
annotation_csv='data/annotation.csv'
unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)
input_w =256
input_h = 306
num_classes = 9
save_dir = "save"

lr= 1e-5
momentum = 0.9
weight_decay = 0.0005

num_itr = 5
num_step = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    #transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad((1,0)),
        torchvision.transforms.ToTensor()
    ])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load data
    unlabled_trainset= UnlabeledDataset(image_folder=image_folder,
                                        scene_index=unlabeled_scene_index,
                                        first_dim='sample',
                                        transform=transform)
    trainloader_u = DataLoader(unlabled_trainset, batch_size=3, shuffle=False, num_workers=2, pin_memory=True)
    trainloader_u_iter = iter(trainloader_u)
    # sample = trainloader_u_iter.next()
    # print(sample.shape)
    # plt.imshow(sample[0].numpy().transpose(1,2,0))
    # plt.axis('off')
    # plt.show()


    labeled_trainset = LabeledDataset(image_folder= image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=True)
    trainloader_l = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    trainloader_l_iter = iter(trainloader_l)
    sample, target, road_image, extra = trainloader_l_iter.next()
    print(target[0]['bounding_box'][0])
    # plt.imshow(sample[0].numpy().transpose(1,2,0))
    # plt.axis('off')
    # plt.show()

    # init model
    generator = resnet18(num_classes=num_classes, pretrained=False).to(device)
    generator.train()

    discriminator = FCDiscriminator(num_classes=num_classes).to(device)
    discriminator.train()

    # init optimizer
    optimizer_g = optim.SGD(generator.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_g.zero_grad()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9,0.99))
    optimizer_d.zero_grad()

    pred_label = 0
    true_label = 1

    for itr in range(num_itr):
        detection_loss = 0
        adv_pred_loss = 0
        disc_loss = 0
        semi_loss = 0
        semi_adv_loss = 0

        #adjust learning rate
        for step in range(num_step):
            #train generator with unlabled data
            for param in discriminator.parameters():
                param.requires_grad = False

            #load batch of unlabeled data
            try:
                batch = trainloader_u_iter.next()
            except:
                trainloader_u_iter = iter(trainloader_u)
                batch = trainloader_u_iter.next()
            # print(batch.shape)
            #temp = np.swapaxes(batch, 0,1)
            score, cls, box  = generator(batch.to(device), supervised=False)
            print(box)
            pred_remain = pred.detach()

            D_out = discriminator(F.softmax(pred))
            D_out_sigmoid = F.sigmoid(D_out).data.cup().numpy().squeeze(axis=1)

            ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

            #compute loss and update

            # train generator with labled data
            try:
                batch, target, _, _ = trainloader_l_iter.next()
            except:
                trainloader_l_iter = iter(trainloader_l)
                batch, target, _, _ = trainloader_l_iter.next()
            torch.stack(batch)

            pred = generator(batch)

            D_out = discriminator(F.soft_margin_loss(pred))

            ignore_mask = (target.numpy()==255)

            #compute loss and update

            #train discriminator
            for param in discriminator.parameters():
                param.requires_grad = True

            pred = pred.detach()
            pred = torch.cat((pred, pred_remain), 0)
            ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis = 0)

            D_out = discriminator(F.softmax(pred))

            #compute loss and update

            #train discriminator with true labels


if __name__ == '__main__':
    main()