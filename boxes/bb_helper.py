import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import collate_fn, convert_map_to_road_map

import cv2

def draw_black_box(ax, corners):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color='black', linewidth=0.5)
    
def coordinates_to_binary_tensor(coordinates):
    """input: (n, 2, 4)"""
    """output: (800, 800)"""
    
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
    fig.set_size_inches(800.0/float(DPI),800.0/float(DPI))
    fig.canvas.draw()
    plt.close(fig)

    # Now we can save it to a numpy array.
    bb_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bb_image = bb_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    bb_image_gray = cv2.cvtColor(bb_image,cv2.COLOR_BGR2GRAY)
    bb_image_binary = torch.tensor(bb_image_gray) < 255

    return bb_image_binary

def batch_coordinates_to_binary_tensor(batch_coordinates):
    """input: batchsize-tuple of (n,2,4)-tensors"""
    """output: (batchsize, 800, 800)"""
    output_tensors = []
    for coordinates in batch_coordinates:
        bb_image_binary = coordinates_to_binary_tensor(coordinates)
        output_tensors.append(bb_image_binary)
    output_tensors_stacked = torch.stack(output_tensors)
    return output_tensors_stacked

def binary_tensor_to_image(bb_image_binary):
    # TODO
    return

