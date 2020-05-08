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

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Alset'
    team_number = 36
    round_number = 3
    team_member = ['Wenjun Qu', 'Xulai Jiang', 'Neil Menghani']
    contact_email = 'nlm326@nyu.edu'

    def __init__(self, model_file='save':
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = {'encoder':'save/encoder_save','decoder_road':'save/decoder_road','decoder_box':'save/decoder_box'}
        self.encoder = model.Encoder(18,800,800,False,num_imgs=6).to(self.device)
        self.encoder.load_state_dict(torch.load(model_path['encoder']))
        self.decoder = model.Decoder(encoder.resnet_encoder.num_ch_enc).to(self.device)
        self.decoder.load_state_dict(torch.load(model_path['decoder_road']))

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return torch.rand(1, 800, 800) > 0.5
