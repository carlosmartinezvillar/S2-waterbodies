import os
import time
import sys
import numpy as np
import torch
# import torch.nn.functional as F #if pad collate is needed
# import multiprocessing
import torchvision as tv
from pillow import Image

import utils

################################################################################
# FILE PATHS
################################################################################
DATA_DIR = "../dat/"


################################################################################
# CLASSES
################################################################################
class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self):

	def __len__(self):

	def __getitem__(self):
		b = Image.open(f'{DATA_DIR}/chips/{self.rgbn_arr[i]}_B02.tif')
		