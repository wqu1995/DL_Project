# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset import ValDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm