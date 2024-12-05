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
# CLASSES
################################################################################
class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3):
		self.root = chip_dir
		self.ids  = 

		if n_bands == 3:
			band_suffixes = ['B02','B02','B03']
		elif n_bands == 4:
			band_suffixes = ['B02','B02','B03','B08']
		else:
			print("")

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		# one way
		b = Image.open(f'{self.root}/{self.ids[i]}_B02.tif')
		g = Image.open(f'{self.root}/{self.ids[i]}_B03.tif')
		r = Image.open(f'{self.root}/{self.ids[i]}_B04.tif')
		n = Image.open(f'{self.root}/{self.ids[i]}_B08.tif')	

		t = np.array(Image.open(f'{self.root}/{self.ids[i]}_LBL.tif').convert('L'))
		t = (t >> 7).astype(np.float32)

		return