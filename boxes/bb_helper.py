import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import collate_fn, convert_map_to_road_map

from PIL import Image

def draw_black_box(ax, corners):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color='black', linewidth=0.5)
    
def draw_bounding_boxes(coordinates):
    ### Plot empty background (or negative roadmap) ###
    fig, ax = plt.subplots()
    plt.axis('off')
    background = torch.zeros((800,800)) > 0
    # background = road_image[0] != True
    ax.imshow(background, cmap='binary');

    ### Draw Boxes ###
    for i, bb in enumerate(coordinates):
        draw_black_box(ax, bb)

    ### Ensure DPI is Correct and Save Image ###
    DPI = 200
    matplotlib.rcParams['figure.dpi'] = DPI
    fig.set_size_inches(1007.0/float(DPI),1007.0/float(DPI)) 
    plt.savefig("temp.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)

    ### Read Image and Convert to Tensor ###
    bb_image = Image.open("temp.png")
    bb_image = torchvision.transforms.functional.to_tensor(bb_image)
    binary_bb_image = convert_map_to_road_map(bb_image)
#     print(new_image.shape)
#     fig, ax = plt.subplots()
#     ax.imshow(new_image, cmap='binary');
    return binary_bb_image