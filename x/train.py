# code adpoted and inspried from https://github.com/hbutsuak95/monolayout and https://github.com/hfslyc/AdvSemiSeg
import os

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn
from model import model

from helper import compute_ts_road_map, compute_ats_bounding_boxes
import matplotlib.pyplot as plt
from boxes.bb_helper import *

image_folder = 'data'
annotation_csv = 'data/annotation.csv'
save_dir = 'save'
unlabled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106,130)
vaildate_scene_index = np.arange(130,134)


height=256
width=256

batch_size =8
step = 50000
lr = 2.5e-4
dlr = 1e-4
iter_size = 1
weight = 5

lambda_semi_adv = 0.001
semi_start_adv = 500

lambda_semi = 0.1
semi_start = 1000

lambda_adv = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

upsample_bb = nn.Upsample(size=(800, 800), mode='nearest')
upsample_road = nn.Upsample(size=(800,800), mode='bilinear', align_corners=True)
sigmoid = nn.Sigmoid()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices =['static', 'dynamic'], help='Type of model being trained', default='dynamic')
    parser.add_argument('--data', type=str, help='path to the data folder', default=image_folder)
    parser.add_argument('--save', type=str, help='path to the save folder', default=save_dir)
    parser.add_argument('--abtch', type=int, help='batch size', default=batch_size)
    parser.add_argument('--step', type=int, help='step size', default=step)
    parser.add_argument('--lr', type=float, help='learning rate for generator', default=lr)
    parser.add_argument('--dlr', type=float, help='learning rate for discriminator', default=dlr)
    parser.add_argument('--adv', type=int, help='starting step for adversary learning', default=semi_start_adv)
    parser.add_argument('--semi', type=int, help='staring step for semi supervised learning', default=semi_start)
    parser.add_argument('--cont', type=bool, help='continue training from previous state', default=False)

    return parser.parse_args()


def vaildate(model, test_loader, itr, type):
    test_loss = 0
    test_ts = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            sample,bounding_box,road_image,_ = data
            total+=1
            # print(bounding_box['bounding_box'][0][0])
            # print(bounding_box['bounding_box'].shape)
            if type == 'dynamic':
                target = coordinates_to_binary_tensor(bounding_box['bounding_box'].squeeze()).view(1,1,800,800).float()
            else:
                target = road_image.view(1,1,800,800).float()
            # print("**************************")
            # print(bounding_box['bounding_box'][0][0])
            # print(target.shape)
            # print(bounding_box[0]['bounding_box'].shape)
            # print(torch.sum(target[0].float()))
            sample = process_imgs(sample)

            features = model['encoder'](sample)
            output = model['decoder'](features)
            if type == 'dynamic':
                loss = compute_losses(upsample_bb(output), target.to(device), type)
            else:
                loss = compute_losses(upsample_road(output), target.to(device), type)

            # print(road_image.shape)
            test_loss+=loss

            if type == 'dynamic':
                pred_map = upsample_bb(sigmoid(output))
                ignore_mask = (pred_map <0.5)
            else:
                pred_map = upsample_road(sigmoid(output))
                ignore_mask = (pred_map<0.3)
            map = torch.ones(pred_map.shape)
            map[ignore_mask] = 0
            map = map.view(1,800,800)
            # print(map.shape)

            if type == 'dynamic':
                coordinates = batch_binary_tensor_to_coordinates(map)
                # print(coordinates.shape)
                ts = compute_ats_bounding_boxes(coordinates[0], bounding_box['bounding_box'][0])
            else:
                ts = compute_ts_road_map(map, road_image)
            # print(ts.item())
            test_ts+= ts

            if i == 7:
                # print(coordinates[0].shape, bounding_box['bounding_box'][0].shape)
                # print(coordinates[0][:10], bounding_box['bounding_box'][0][:10])
                # f, axarr = plt.subplots(1,2)
                # axarr[0,0] = plt.imshow(road_image[0], cmap='binary')
                # axarr[0,1] = plt.imshow(map[0], cmp='binary')
                # plt.show()
                fig, ax = plt.subplots(1,3)
                # print(target[0].shape)
                ax[0].imshow(target[0][0],cmap='binary')
                ax[1].imshow(map[0], cmap='binary')
                ax[2].imshow(np.squeeze(pred_map.cpu().numpy()), cmap = 'binary')
                # plt.savefig(os.path.join(save_dir,str(itr)+'.png'))
                plt.close('all')
    print('iter: {:d}\t vaildate result: loss: {:.4f}\t ts: {:.4f}'.format(itr, test_loss/total, test_ts/total))
    return test_ts/total


def compute_losses(pred, target, type):
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    # print(pred.shape, target.shape)
    if type == 'static':
        loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(0.8))
    else:
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(38.))
    output = loss(pred, target.float())
    # print(output.shape, output.item())
    return output

def process_imgs(batch):
    batch[:,3:6] = batch[:,3:6].flip(3)
    temp = torch.stack([torchvision.utils.make_grid(s, nrow=3, padding=0) for s in batch])
    return temp.to(device)

def train():
    arg = get_args()
    models = {}
    criterion_d = nn.BCEWithLogitsLoss()
    parameters_to_train = []
    parameters_to_train_D = []

    # init models
    models['encoder'] = model.Encoder(18, height*2, width*3, False, num_imgs=1)
    models['decoder'] = model.Decoder(models['encoder'].resnet_encoder.num_ch_enc)
    models['discriminator'] = model.Discriminator()

    if arg.cont:
        bounding_state = torch.load(os.path.join(arg.save, 'gen.pth'))
        models['encoder'].load_state_dict(bounding_state['encoder'])
        models['decoder'].load_state_dict(bounding_state['decoder'])
        models['discriminator'].load_state_dict(torch.load(os.path.join(arg.save, 'dis.pth')))

    for key in models.keys():
        models[key].to(device)
        if 'discr' in key:
            parameters_to_train_D += list(models[key].parameters())
        else:
            parameters_to_train += list(models[key].parameters())

    #init optimizer
    model_optimizer = optim.Adam(parameters_to_train, arg.lr)
    model_optimizer_D = optim.Adam(parameters_to_train_D, arg.dlr)
    model_optimizer.zero_grad()
    model_optimizer_D.zero_grad()

    patch = (1, 800//2**4, 800//2**4)
    vaild = Variable(torch.ones((batch_size, *patch)), requires_grad=False).to(device)
    fake = Variable(torch.zeros((batch_size, *patch)), requires_grad=False).to(device)
    #print(vaild.shape, fake.shape)

    #load data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor()
    ])

    if not os.path.exists(arg.save):
        os.makedirs(arg.save)

    unlabled_trainset = UnlabeledDataset(
        image_folder=image_folder,
        scene_index=unlabled_scene_index,
        first_dim='sample',
        transform=transform
    )
    trainloader_u = DataLoader(unlabled_trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    trainloader_u_iter = iter(trainloader_u)
    # sample = trainloader_u_iter.next()
    # sample[:,3:6] = torch.flip(sample[:,3:6], 3)


    # b,i,c,h,w = sample.shape
    # sample = sample.view(b, i*c, 800, 800)
    # pic = torchvision.utils.make_grid(sample[2][3:6], padding=0)
    # plt.imshow(pic.numpy().transpose(1,2,0))
    # plt.axis('off')
    # plt.show()

    labeled_trainset = LabeledDataset(
        image_folder=image_folder,
        annotation_file=annotation_csv,
        scene_index=labeled_scene_index,
        transform=transform,
        extra_info=True
    )
    trainloader_l = DataLoader(labeled_trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    trainloader_l_iter = iter(trainloader_l)

    labeled_vaildate = LabeledDataset(
        image_folder=image_folder,
        annotation_file=annotation_csv,
        scene_index=vaildate_scene_index,
        transform=transform,
        extra_info=True
    )
    vaildateloader = DataLoader(labeled_vaildate, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    best_ts = 0


    for i in range(arg.step):
        loss_d = 0
        loss_g = 0

        model_optimizer.zero_grad()
        model_optimizer_D.zero_grad()

        for sub_i in range(iter_size):
            #train with unlabel data
            # do not update discriminator with unlabel data
            for param in models['discriminator'].parameters():
                param.requires_grad = False

            if(lambda_semi > 0 or lambda_semi_adv >0) and i > semi_start_adv:
                try:
                    samples = trainloader_u_iter.next()
                except:
                    trainloader_u_iter = iter(trainloader_u)
                    samples = trainloader_u_iter.next()
                samples = process_imgs(samples)

                features = models['encoder'](samples)
                if arg.type =='dynamic':
                    output = upsample_bb(models['decoder'](features))
                else:
                    output = upsample_road(models['decoder'](features))

                fake_pred = models['discriminator'](output)

                loss_semi_adv = lambda_semi_adv* criterion_d(fake_pred, vaild)

                if lambda_semi <= 0 or i < semi_start_adv:
                    loss_semi_adv.backward()
                    loss_semi = 0
                else:
                    if arg.type == 'dynamic':
                        fake_pred_sig = upsample_bb(sigmoid(fake_pred))
                        ignore_mask = (fake_pred_sig < 0.5)
                    else:
                        fake_pred_sig = upsample_road(sigmoid(fake_pred))
                        ignore_mask = (fake_pred_sig<0.3)
                    semi_gt = torch.ones(fake_pred_sig.shape)
                    semi_gt[ignore_mask] = 0

                    loss_semi_ce = lambda_semi* compute_losses(output, semi_gt.to(device), arg.type)
                    loss_semi = loss_semi_ce+ loss_semi_adv
                    loss_semi.backward()
            else:
                loss_semi = 0

            #train with label data
            for param in models['discriminator'].parameters():
                param.requires_grad = True
            try:
                samples, bounding_box, road_image, _ = trainloader_l_iter.next()
            except:
                trainloader_l_iter = iter(trainloader_l)
                samples, bounding_box, road_image, _ = trainloader_l_iter.next()

            samples = process_imgs(torch.stack(samples))
            # print(bounding_box[0]['bounding_box'][0])
            # samples = torch.stack(samples).view(batch_size, 18, 512, 512).to(device)

            if arg.type == 'dynamic':
                target = batch_coordinates_to_binary_tensor(bounding_box).view(batch_size,1,800,800).float().to(device)
            else:
                target = torch.stack(road_image).view(batch_size,1,800,800).float().to(device)

            features = models['encoder'](samples)
            if arg.type == 'dynamic':
                output = upsample_bb(models['decoder'](features))
            else:
                output = upsample_road(models['decoder'](features))
            # print(output.shape)
            #compute L_ce
            loss_ce = compute_losses(output, target, arg.type)

            fake_pred = models['discriminator'](output)
            real_pred = models['discriminator'](target)

            loss_adv = criterion_d(fake_pred, vaild)
            loss_d = criterion_d(fake_pred, fake) + criterion_d(real_pred, vaild)
            loss_g = lambda_adv*loss_adv+loss_ce


            #update
            # model_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            model_optimizer.step()
            # model_optimizer_D.zero_grad()
            loss_d.backward()
            model_optimizer_D.step()

        if i %100 == 0:
            print('itr: {:d}\t training reslt: generator loss: {:.4f}\t disc loss: {:.4f}'.format(i, loss_g.item(), loss_d.item()))
            ts = vaildate(models, vaildateloader, i, arg.type)

            if ts >= best_ts:
                torch.save({
                    'encoder': models['encoder'].state_dict(),
                    'decoder': models['decoder'].state_dict()
                }, os.path.join(save_dir, 'gen.pth'))
                torch.save(models['discriminator'].state_dict(), os.path.join(save_dir, 'dis.pth'))

if __name__ == '__main__':
    train()