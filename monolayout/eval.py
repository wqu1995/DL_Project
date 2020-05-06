import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision

from model import model
from torch.utils.data import DataLoader
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn

image_folder = 'data'
annotation_csv = 'data/annotation.csv'
save_dir = 'save'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512,512)),
        torchvision.transforms.ToTensor()
    ])

batch_size =8

eval_num = 120
eval_scene_index = np.arange(eval_num,134)
eval_set = LabeledDataset(
        image_folder = image_folder,
        annotation_file = annotation_csv,
        scene_index = eval_scene_index,
        transform = transform,
        extra_info = True
    )
eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)


def compute_losses(pred, target):
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    # print(pred.shape, target.shape)
    loss = nn.BCEWithLogitsLoss()
    output = loss(pred, target.float())
    # print(output.shape, output.item())
    return output

def avg_loss_road(encoder, decoder, loader, device):
    total_loss = 0
    for i in range(len(loader)):
        try:
            samples, target, road_image, _ = loader.__next__()
        except:
            loader = iter(loader)
            samples, target, road_image, extra = loader.__next__()
        samples = torch.stack(samples).view(batch_size, 18, 512, 512).to(device)
        road_image = torch.stack(road_image).view(batch_size, 1, 800, 800).float().to(device)
        test_pred =decoder(encoder(samples))
        total_loss += compute_losses(test_pred, road_image)

    avg_loss = total_loss / len(loader)
    return avg_loss

def main():
    model_path = {'encoder': 'save.encoder_save', 'decoder': 'save.decoder_save'}

    encoder = model.Encoder(18,800,800,False,num_imgs=6).to(device)
    encoder.load_state_dict(torch.load(model_path['encoder']))
    decoder = model.Decoder(encoder.resnet_encoder.num_ch_enc).to(device)
    decoder.load_state_dict(torch.load(model_path['decoder']))
    print("Avg Evaluation Loss is:")
    print(avg_loss_road(encoder,decoder,eval_loader,device))

if __name__ == '__main__':
    main()
