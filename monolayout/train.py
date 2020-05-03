import os

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn

image_folder = 'data'
annotation_csv = 'data/annotation.csv'
save_dir = 'save'
unlabled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106,134)
num_classes = 9

epoch = 5
lr = 1e-5
lr_step_size = 5
iter_size = 1
discr_train_epoch = 5

alpha = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices =['static', 'dynamic'], help = 'Type of model being trained')

    return parser.parse_args()

def compute_losses(input, output):
    losses = {}

    true_label = torch.squeeze(input.long())
    loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., weight]).cuda())
    output = loss(output, true_label)
    losses['loss'] = output.mean()

    return losses


def train():
    arg = get_args()
    models = {}
    criterion_d = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()
    parameters_to_train = []
    parameters_to_train_D = []

    # init models
    # models['encoder'] =
    # models['decoder'] =
    # models['discriminator'] =

    for key in models.keys():
        models[key].to(device)
        if 'discr' in key:
            parameters_to_train_D += list(models[key].parameters())
        else:
            parameters_to_train += list(models[key].parameters())

    #init optimizer
    model_optimizer = optim.Adam(parameters_to_train, lr)
    model_lr_scheduler = optim.lr_scheduler.StepLR(model_optimizer, lr_step_size, 0.1)
    model_optimizer_D = optim.Adam(parameters_to_train_D, lr)
    model_lr_scheduler_D = optim.lr_scheduler.StepLR(model_optimizer_D, lr_step_size, 0.1)

    #patch
    #vaild
    #fake

    #load data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    unlabled_trainset = UnlabeledDataset(
        image_folder=image_folder,
        scene_index=unlabled_scene_index,
        first_dim='sample',
        transform=transform
    )
    trainloader_u = DataLoader(unlabled_trainset, batch_size=3, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    trainloader_u_iter = iter(trainloader_u)

    labeled_trainset = LabeledDataset(
        image_folder=image_folder,
        annotation_file=annotation_csv,
        scene_index=labeled_scene_index,
        transform=transform,
        extra_info=True
    )
    trainloader_l = DataLoader(labeled_trainset, batch_size=3, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    trainloader_l_iter = iter(trainloader_l)

    #training
    for i in range(epoch):
        # iter size = 1
        model_optimizer.step()
        model_optimizer_D.step()
        loss = {}
        loss['loss'], loss['loss_discr'] = 0.0, 0.0

        for j in range(iter_size):
            # train with labeled data first
            outputs = {}

            # Ld = min((CE(1,D(Il))+CE(0,D(Iu)))
            loss_D = 0.0
            # Lg = min(CE(yl,F(xl)) + alpha * CE(1,D(Iu)))
            loss_G = 0.0

            try:
                batch, labled_target,_,_ = trainloader_l_iter.next()
            except:
                trainloader_l_iter = iter(trainloader_l)
                batch, labled_target,road_image,extra = trainloader_l_iter()

            features = models['encoder'](batch.to(device))
            outputs['topview_l'] = models['decoder'](features)

            #compute generator loss for label data
            if arg.dynamic:
                road_image = convert_object(labled_target)
            #generator loss
            losses = compute_losses(road_image, outputs)
            losses['loss_discr'] = torch.zeros(1)

            model_optimizer.zero_grad()

            real_pred = models['discriminator'](outputs['topview_l'])
            loss_D += criterion_d(real_pred, vaild)
            loss_G += losses['loss']

            #train with unlabled data
            try:
                batch = trainloader_u_iter.next()
            except:
                trainloader_u_iter = iter(trainloader_u)
                batch = trainloader_u_iter()

            features = models['encoder'](batch.to(device))
            outputs['topview_u'] = models['decoder'](features)

            #skip compute generator loss for unlabel data
            fake_pred = models['discriminator'](outputs['topview_u'])
            loss_D += criterion_d(fake_pred, invaild)
            loss_G += alpha * criterion(fake_pred, vaild)

            if i > discr_train_epoch:
                loss_G.backward(retain_graph=True)
                model_optimizer.step()

                model_optimizer_D.zero_grad()
                loss_D.backward()
                model_optimizer_D.step()
            else:
                losses['loss'].backward()
                model_optimizer.step()

            loss['loss'] += losses['loss'].item()
            loss['loss_discr'] += loss_D.item()









if __name__ == '__main__':
    train()