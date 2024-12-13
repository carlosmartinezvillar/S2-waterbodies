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
		self.dir = chip_dir
		self.ids = []

		if n_bands == 3:
			band_suffixes = ['B02','B02','B03']
		elif n_bands == 4:
			band_suffixes = ['B02','B02','B03','B08']
		else:
			raise ValueError("Incorrect number of bands in dataloader.")

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		# one way
		b = Image.open(f'{self.dir}/{self.ids[i]}_B02.tif')
		g = Image.open(f'{self.dir}/{self.ids[i]}_B03.tif')
		r = Image.open(f'{self.dir}/{self.ids[i]}_B04.tif')
		if self.bands == 4:
			n = Image.open(f'{self.dir}/{self.ids[i]}_B08.tif')
			arr = r+g+b+n
		else:
			arr = r+g+b	
		t = np.array(Image.open(f'{self.root}/{self.ids[i]}_LBL.tif').convert('L'))
		t = (t >> 7).astype(np.float32)

		return

'''
testing, validation split:
Test dataset.
1. Choose 2/6 UTM zones at random, without replacement.
2. Choose a tile with (a number of raster below the median) at random for each of the two UTM zones.
Validation dataset
3. Off remaining set, choose 2/6 UTM zones at random, without replacement.
4. Choose a tile at random for each of the two UTM zones. 
'''

if __name__ == '__main__':
	pass