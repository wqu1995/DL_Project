"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from model.model import *
import os

save_dir = 'save'

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    team_number = 1
    round_number = 1
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(self, model_file='put_your_model_file(or files)_name_here'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #

        print('hi')
        models = {}
        models['s_encoder'] = Encoder(18, 800, 800, False, num_imgs=6)
        models['s_decoder'] = Decoder(models['s_encoder'].resnet_encoder.num_ch_enc)

        models['d_encoder'] = Encoder(18, 800, 800, False, num_imgs=6)
        models['d_decoder'] = Decoder(models['d_encoder'].resnet_encoder.num_ch_enc)

        # models['s_encoder'].load_state_dict(torch.load(os.path.join(save_dir, '0_e.pth')))
        # models['s_decoder'].load_state_dict(torch.load(os.path.join(save_dir, '0_d.pth')))
        #
        #
        # models['d_encoder'].load_state_dict(torch.load(os.path.join(save_dir, '0_e.pth')))
        # models['d_decoder'].load_state_dict(torch.load(os.path.join(save_dir, '0_d.pth')))


    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return torch.rand(1, 800, 800) > 0.5
