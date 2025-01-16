import os
import time
import sys
import numpy as np
import torch
import torchvision as tv
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import utils
import glob

################################################################################
# CLASSES
################################################################################
class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3):
		self.dir        = chip_dir
		self.rgbn_files = sorted(glob.glob("*_B0X.tif",root_dir=self.dir))
		self.ids        = [i[0:-8] for i in files]
		self.n_bands    = n_bands
		if (self.n_bands!=3) and (self.n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")

		self.transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True)])

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		if self.n_bands == 4:
			img = self.transform(Image.open(f'{self.dir}/{self.ids[idx]}_B0X.tif'))
		else:
			r,g,b,_ = Image(open(f'{self.dir}/{self.ids[idx]}_B0X.tif')).split()
			img = torch.cat(self.transform([r,g,b]),axis=0)
			
		lbl = self.transform(Image.open(f'{self.dir}/{self.ids[idx]}_LBL.tif'))
		return img,lbl
'''
testing, validation split:
Test dataset.
1. Choose 2/6 UTM zones at random, without replacement.
2. Choose a tile with (a number of raster below the median) at random for each of the two UTM zones.
Validation dataset
3. Off remaining set, choose 2/6 UTM zones at random, without replacement.
4. Choose a tile at random for each of the two UTM zones. 
'''

def test_validation_split():
	pass


if __name__ == '__main__':

	# CHECK LABELS AND STATS.TXT MAKE SOME SENSE...
	band_files  = sorted(glob.glob("*_B0X.tif",root_dir='../chips'))
	label_files = sorted(glob.glob("*_LBL.tif",root_dir='../chips'))

	chips   = np.unique(chips).sort() #23932
	rasters = [i[0:-19] for i in files] #23932
	tiles   = [i[-25:-19] for i in files] #23932
	zones   = [i[-24:-21] for i in files] #23932
 
	unique_chips   = np.unique(chips,return_counts=True)
	unique_rasters = np.unique(rasters)
	unique_tiles   = np.unique(tiles)
	unique_zones   = np.unique(zones)

	with open('../lake_chips/chips/stats.txt','r') as fp:
		lines = fp.readlines()
	stat_files = [i.rstrip().split('\t')[0] for i in lines]
	stat_pixels = [int(i.rstrip().split('\t')[1]) for i in lines]

	stat_chips   = [i[0:-8] for i in stat_files]
	stat_rasters = [i[0:-19] for i in stat_files]
	stat_tiles   = [i[-25:-19] for i in stat_files]
	stat_zones   = [i[-24:-21] for i in stat_files]
	pass
 