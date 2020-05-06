import os

import numpy as np
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
import matplotlib.pyplot as plt
import matplotlib

image_folder = 'data'
annotation_csv = 'data/annotation.csv'
save_dir = 'save'
unlabled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106,134)
num_classes = 9

width = 306
height = 256

r_width = 800
r_height = 800

batch_size = 8
epoch = 2000
lr = 1e-4
lr_step_size = 5
iter_size = 1
discr_train_epoch = 0
weight = 5

lambda_semi_adv = 0.001
semi_start_adv = 0

lambda_semi = 0.1
semi_start = 100
mask_t = 0.2

alpha = 0.01

lambda_adv = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices =['static', 'dynamic'], help='Type of model being trained', default='static')
    parser.add_argument("eval_num", type=int, default=(106 + 134) / 2,
                        help="Index separating training data and evaluation data(between 106,134).")

    return parser.parse_args()

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).to(device)

    return D_label

def compute_losses(pred, target):
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    # print(pred.shape, target.shape)
    loss = nn.BCEWithLogitsLoss()
    output = loss(pred, target.float())
    # print(output.shape, output.item())
    return output
    # losses = {}
    #
    # true_label = torch.squeeze(input.float())
    # pred = torch.squeeze(output['topview_l'])
    #
    # # loss = nn.CrossEntropyLoss2d(weight=torch.Tensor([1., weight]).cuda())
    # loss = nn.BCEWithLogitsLoss()
    # #print(true_label.shape, pred.shape)
    # output = loss(pred, true_label)
    # losses['loss'] = output.mean()

    # return losses

def train():
    arg = get_args()
    models = {}
    criterion_d = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()
    parameters_to_train = []
    parameters_to_train_D = []

    # init models
    models['encoder'] = model.Encoder(18, r_height, r_width, False, num_imgs=6)
    models['decoder'] = model.Decoder(models['encoder'].resnet_encoder.num_ch_enc)
    models['discriminator'] = model.Discriminator()

    for key in models.keys():
        models[key].to(device)
        if 'discr' in key:
            parameters_to_train_D += list(models[key].parameters())
        else:
            parameters_to_train += list(models[key].parameters())

    #init optimizer
    model_optimizer = optim.Adam(parameters_to_train, 2.5e-4)
    model_lr_scheduler = optim.lr_scheduler.StepLR(model_optimizer, lr_step_size, 0.1)
    model_optimizer_D = optim.Adam(parameters_to_train_D, 1e-4)
    model_lr_scheduler_D = optim.lr_scheduler.StepLR(model_optimizer_D, lr_step_size, 0.1)
    model_optimizer.zero_grad()
    model_optimizer_D.zero_grad()

    patch = (1, 800//2**4, 800//2**4)
    vaild = Variable(torch.ones((batch_size, *patch)), requires_grad=False).to(device)
    fake = Variable(torch.zeros((batch_size, *patch)), requires_grad=False).to(device)
    #print(vaild.shape, fake.shape)

    #load data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512,512)),
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
    trainloader_u = DataLoader(unlabled_trainset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    trainloader_u_iter = iter(trainloader_u)
    # sample = trainloader_u_iter.next()
    # b,i,c,h,w = sample.shape
    # sample = sample.view(b, i*c, 800, 800)
    # pic = torchvision.utils.make_grid(sample[2][3:6], padding=0)
    # plt.imshow(pic.numpy().transpose(1,2,0))
    # plt.axis('off')
    # plt.show()

    #Dividing labeled data to training part and validation part:
    eval_num = arg.eval_num
    labeled_scene_index = np.arange(106,eval_num)
    eval_scene_index = np.arange(eval_num,134)

    #Labeled Data
    labeled_trainset = LabeledDataset(
        image_folder=image_folder,
        annotation_file=annotation_csv,
        scene_index=labeled_scene_index,
        transform=transform,
        extra_info=True
    )
    trainloader_l = DataLoader(labeled_trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)
    trainloader_l_iter = iter(trainloader_l)

    #Evaluation/ Validation Data
    eval_set = LabeledDataset(
        image_folder = image_folder,
        annotation_file = annotation_csv,
        scene_index = eval_scene_index,
        transform = transform,
        extra_info = True
    )
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)

    #labels for true/ false
    gt_label =1
    pred_label = 0

    upsample = nn.Upsample(size = (800, 800), mode='bilinear', align_corners=True)
    sigmoid = nn.Sigmoid()

    for i in range(epoch):
        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        loss_ce = 0
        loss_adv = 0
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
                    batch = trainloader_u_iter.next()
                except:
                    trainloader_u_iter = iter(trainloader_u)
                    batch = trainloader_u_iter.next()
                batch.view(batch_size, 18, 512, 512).to(device)

                features = models['encoder'](samples)
                #output = upsample(models['decoder'](features)) #(Upsample moved to decoder)
                output = models['decoder'](features)

                fake_pred = models['discriminator'](output)

                loss_semi_adv = lambda_semi_adv* criterion_d(fake_pred, vaild)

                if lambda_semi <= 0 or i < semi_start_adv:
                    loss_semi_adv.backward()
                    loss_semi = 0
                else:
                    #Confidence Map:
                    fake_pred_sig = upsample(sigmoid(fake_pred))

                    ignore_mask = (fake_pred_sig < 0.2)
                    semi_gt = torch.ones(fake_pred_sig.shape)
                    semi_gt[ignore_mask] = 0

                    loss_semi_ce = lambda_semi * compute_losses(output, semi_gt.to(device))
                    loss_semi = loss_semi_ce + loss_semi_adv
                    loss_semi.backward()
            else:
                loss_semi = 0

            #train with label data
            for param in models['discriminator'].parameters():
                param.requires_grad = True
            try:
                samples, labled_target, road_image, _ = trainloader_l_iter.next()
            except:
                trainloader_l_iter = iter(trainloader_l)
                samples, labled_target, road_image, extra = trainloader_l_iter.next()

            samples = torch.stack(samples).view(batch_size, 18, 512, 512).to(device)
            road_image = torch.stack(road_image).view(batch_size,1,800,800).float().to(device)

            features = models['encoder'](samples)
            # output = upsample(models['decoder'](features)) #(Upsample moved into decoder)
            output = models['decoder'](features)
            # print(output.shape)

            #compute L_ce
            loss_ce = compute_losses(output, road_image)

            fake_pred = models['discriminator'](output)
            real_pred = models['discriminator'](road_image)

            loss_adv = criterion_d(fake_pred, vaild)
            loss_d = criterion_d(fake_pred, fake) + criterion_d(real_pred, vaild)
            loss_g = lambda_adv*loss_adv+loss_ce


            #update
            model_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            model_optimizer.step()
            model_optimizer_D.zero_grad()
            loss_d.backward()
            model_optimizer_D.step()

        #Print out losses every 100 epoch:
        if i %100 == 0:
            print(loss_g.item(), loss_d.item())

    #Saving param to local folder:
    torch.save(models['encoder'].state_dict(), 'encoder_save')
    torch.save(models['decoder'].state_dict(), 'decoder_save')
    path = {}
    path['encoder'] = 'encoder_save'
    path['decoder'] = 'decoder_save'
    # eval_loss = main_loss(eval_loader, path, device)



    # try:
    #     batch = trainloader_u_iter.next()
    # except:
    #     trainloader_u_iter = iter(trainloader_u)
    #     batch = trainloader_u_iter.next()
    # features = models['encoder'](batch.view(batch_size, 18, 512, 512).to(device))
    # print(features.shape)
    # pred = models['decoder'](features)
    #
    # print(pred[0,0,0,:10])
    # x = (sigmoid(upsample(pred)))
    # print(x.shape)
    # print(x[0][0].sum())
    # d = models['discriminator'](x)
    # print(d.shape)

    # for i in range(epoch):
    #     loss_seg_value = 0
    #     loss_adv_pred_value = 0
    #     loss_D_value = 0
    #     loss_semi_value = 0
    #     loss_semi_adv_value = 0
    #
    #     model_optimizer.zero_grad()
    #     model_optimizer_D.zero_grad()
    #
    #     for sub_i in range(iter_size):
    #         #train generator
    #         if (lambda_semi > 0 or lambda_semi_adv > 0 ) and i >= semi_start_adv:
    #             print('in here')
    #             for param in models['discriminator'].parameters():
    #                 param.requires_grad=False
    #
    #             try:
    #                 batch = trainloader_u_iter.next()
    #             except:
    #                 trainloader_u_iter = iter(trainloader_u)
    #                 batch = trainloader_u_iter()
    #             features = models['encoder'](batch.view(batch_size, 18, r_height, r_width).to(device))
    #             pred = models['decoder'](features)
    #             pred_remain = pred.detach()
    #             # print(pred.shape)
    #
    #             D_out = models['discriminator'](F.softmax(pred, dim=1))
    #             # print(D_out.shape)
    #             D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)
    #             # print('doutsig', D_out_sigmoid)
    #             ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)
    #
    #             loss_semi_adv = lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
    #             loss_semi_adv = loss_semi_adv/iter_size
    #
    #             loss_semi_adv_value+= loss_semi_adv.data.cpu().numpy()[0]/lambda_semi_adv
    #
    #             if lambda_semi <= 0 or i < semi_start:
    #                 loss_semi_adv.backward()
    #                 loss_semi_value = 0
    #             else:
    #                 semi_ignore_mask = (D_out_sigmoid < mask_t)
    #                 semi_gt = pred.data.cpu().numpy
    #             # print(loss_semi_adv)

    #training
    # for i in range(epoch):
    #     # iter size = 1
    #     model_optimizer.step()
    #     model_optimizer_D.step()
    #     loss = {}
    #     loss['loss'], loss['loss_discr'] = 0.0, 0.0
    #
    #     for j in range(iter_size):
    #         # train with labeled data first
    #         outputs = {}
    #
    #         # Ld = min((CE(1,D(Il))+CE(0,D(Iu)))
    #         loss_D = 0.0
    #         # Lg = min(CE(yl,F(xl)) + alpha * CE(1,D(Iu)))
    #         loss_G = 0.0
    #
    #         try:
    #             samples, labled_target,road_image,_ = trainloader_l_iter.next()
    #         except:
    #             trainloader_l_iter = iter(trainloader_l)
    #             samples, labled_target,road_image,extra = trainloader_l_iter()
    #         #print(torch.stack(road_image).shape)
    #         samples = torch.stack(samples).view(batch_size, 18, r_height,r_width).to(device)
    #
    #         features = models['encoder'](samples)
    #         print(features.shape)
    #         outputs['topview_l'] = models['decoder'](features)
    #         #print(outputs['topview_l'].shape)
    #
    #         #compute generator loss for label data
    #         if arg.type =='dynamic':
    #             road_image = []
    #         #generator loss
    #         losses = compute_losses(torch.stack(road_image).to(device), outputs)
    #         losses['loss_discr'] = torch.zeros(1)
    #
    #
    #         real_pred = models['discriminator'](outputs['topview_l'])
    #         print('disc', real_pred.shape)
    #         loss_D += criterion_d(real_pred, vaild)
    #         loss_G += losses['loss']
    #         #print('done with label')
    #
    #         #train with unlabled data
    #         try:
    #             batch = trainloader_u_iter.next()
    #         except:
    #             trainloader_u_iter = iter(trainloader_u)
    #             batch = trainloader_u_iter()
    #
    #         #print(batch.shape)
    #
    #         features = models['encoder'](batch.view(batch_size, 18, r_height, r_width).to(device))
    #         outputs['topview_u'] = models['decoder'](features)
    #
    #         #skip compute generator loss for unlabel data
    #         fake_pred = models['discriminator'](outputs['topview_u'])
    #         loss_D += criterion_d(fake_pred, fake)
    #         loss_G += alpha * criterion(fake_pred, vaild)
    #
    #         if i > discr_train_epoch:
    #             model_optimizer.zero_grad()
    #             loss_G.backward(retain_graph=True)
    #             model_optimizer.step()
    #
    #             model_optimizer_D.zero_grad()
    #             loss_D.backward()
    #             model_optimizer_D.step()
    #         else:
    #             losses['loss'].backward()
    #             model_optimizer.step()
    #
    #         loss['loss'] += losses['loss'].item()
    #         loss['loss_discr'] += loss_D.item()
    #
    #     print('loss: {:.4f}, disc loss:{:.4f}'.format(loss['loss'], loss['loss_discr']))

if __name__ == '__main__':
    train()