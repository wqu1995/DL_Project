import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import collate_fn, convert_map_to_road_map

import copy
import cv2
    
def car_angle(bl, br):
    abs_angle = np.arccos((bl[1] - br[1]) / torch.norm(bl - br, 2)) * (180 / np.pi)
    if bl[0] <= br[0]:
        return -abs_angle
    else:
        return abs_angle

def draw_rectangle(ax, corners, color):
    fl, fr, bl, br = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    xy = (br[0] * 10 + 400, -br[1] * 10 + 400)
    width = torch.norm(br - fr, 2) * 10
    height = -torch.norm(br - bl, 2) * 10
    rect = mpatches.Rectangle(xy, width, height, angle=car_angle(bl, br), color=color)
    ax.add_patch(rect)
    
def coordinates_to_binary_tensor(coordinates):
    """input: (n, 2, 4)"""
    """output: (800, 800)"""
    img = np.ones((800,800)) * 255
    for box in coordinates:
        bb = copy.deepcopy(box)
        bb[0,:] = bb[0,:]*10 + 400
        bb[1,:] = -bb[1,:]*10 + 400
        bb = torch.stack([bb[:, 0], bb[:, 1], bb[:, 3], bb[:, 2]])
        bb = np.array(bb, dtype=np.int32)
        cv2.fillPoly(img, [bb], 0)
    bb_image_binary = torch.tensor(img) < 255
    return bb_image_binary

def batch_coordinates_to_binary_tensor(batch_coordinates):
    """input: batchsize-tuple of (n,2,4)-tensors"""
    """output: (batchsize, 800, 800)"""
    output_tensors = []
    for coordinates in batch_coordinates:
        bb_image_binary = coordinates_to_binary_tensor(coordinates['bounding_box'])
        output_tensors.append(bb_image_binary)
    output_tensors_stacked = torch.stack(output_tensors)
    return output_tensors_stacked

def binary_tensor_to_coordinates(bb_binary):
    """input: (800, 800)"""
    """output: (n, 2, 4)"""
    
    ### Find Contours ###
    bb_image_gray = np.array((bb_binary != True).int()*255).astype(np.uint8)
    edged = cv2.Canny(bb_image_gray, 30, 200)
    try:
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    ### Find Corners ###
    coords_list = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        boxPoints = cv2.boxPoints(rect)
        if np.linalg.norm(boxPoints[0] - boxPoints[1]) < np.linalg.norm(boxPoints[0] - boxPoints[3]):
            coord_indices = (0, 1, 3, 2)
        else:
            coord_indices = (1, 2, 0, 3)
        fl_idx, fr_idx, bl_idx, br_idx = coord_indices
        boxPoints = np.array([boxPoints[fl_idx], boxPoints[fr_idx], boxPoints[bl_idx], boxPoints[br_idx]])
        corners = torch.tensor(boxPoints).T
        corners[0] = (corners[0] - 400) / 10
        corners[1] = - (corners[1] - 400) / 10
        coords_list.append(corners)
    if len(coords_list) > 0:
        coords_output = torch.stack(coords_list)
    else:
        coords_output = torch.zeros((1,2,4))
    return coords_output

def batch_binary_tensor_to_coordinates(batch_bb_binary):
    """input: (batchsize, 800, 800)"""
    """output: batchsize-tuple of (n,2,4)-tensors"""
    batch_coords_output = []
    for bb_binary in batch_bb_binary:
        coords_output = binary_tensor_to_coordinates(bb_binary)
        batch_coords_output.append(coords_output)
    return batch_coords_output