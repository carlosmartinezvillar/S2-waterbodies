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
		self.ids        = [i[0:-8] for i in self.rgbn_files]
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
			r,g,b,_ = Image.open(f'{self.dir}/{self.ids[idx]}_B0X.tif').split()
			img = torch.cat(self.transform([r,g,b]),axis=0)
			
		lbl = self.transform(Image.open(f'{self.dir}/{self.ids[idx]}_LBL.tif')).squeeze(0).to(torch.long)
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

def test_validation_split(chip_dir):
	# CHECK LABELS AND STATS.TXT MAKE SOME SENSE...
	band_files  = sorted(glob.glob("*_B0X.tif",root_dir=chip_dir))
	label_files = sorted(glob.glob("*_LBL.tif",root_dir=chip_dir))

	all_rasters = [i[0:-19] for i in band_files] #19456
	all_tiles = [i[-25:-19] for i in band_files] #19456
	
	unique_r,r_inv,r_cnt = np.unique(all_rasters,return_inverse=True,return_counts=True) #726
	tiles   = [i[-6:] for i in unique_r] #726
	unique_t,t_inv,t_cnt = np.unique(tiles,return_inverse=True,return_counts=True) #21

	# CHOOSE 4 TILES AT RANDOM
	tile_choices     = np.random.choice(unique_t,4,replace=False)
	validation_tiles = tile_choices[0:2]
	test_tiles       = tile_choices[2:]

	validation_mask  = np.isin(all_tiles,validation_tiles)
	validation_idx   = np.where(validation_mask)[0]
	test_mask        = np.isin(all_tiles,test_tiles)
	test_idx         = np.where(test_mask)[0]
	training_mask    = ~(validation_mask+test_mask)
	training_idx     = np.where(training_mask)[0]
	
	return training_idx,validation_idx,test_idx


if __name__ == '__main__':

	# CHECK LABELS AND STATS.TXT MAKE SOME SENSE...
	band_files  = sorted(glob.glob("*_B0X.tif",root_dir='../../chips'))
	label_files = sorted(glob.glob("*_LBL.tif",root_dir='../../chips'))

	all_rasters = [i[0:-19] for i in band_files] #19456
	unique_r,r_inv,r_cnt = np.unique(all_rasters,return_inverse=True,return_counts=True) #726

	all_tiles = [i[-25:-19] for i in band_files] #19456
	tiles   = [i[-6:] for i in unique_r] #726
	unique_t,t_inv,t_cnt = np.unique(tiles,return_inverse=True,return_counts=True) #21

	# CHOOSE 4 TILES AT RANDOM
	# tile_choices     = np.random.choice(unique_t,4,replace=False)
	# validation_tiles = tile_choices[0:2]
	# validation_mask  = np.isin(all_tiles,validation_tiles)
	# validation_idx   = np.where(validation_mask)[0]
	# test_tiles       = tile_choices[2:]
	# test_mask        = np.isin(all_tiles,test_tiles)
	# test_idx         = np.where(test_mask)[0]
	# training_mask    = ~(validation_mask+test_mask)
	# training_idx     = np.where(training_mask)[0]
	# pass

	# tile_choices = []
	# unique_z,z_inv,z_cnt = np.unique(zones,return_inverse=True,return_counts=True) #6
	zone_choice = np.random.choice(unique_z,2,replace=False)
	# for choice in zone_choice:
	# 	idx = np.where(unique_z==choice)
	# 	tiles_in_zone = unique_t[(z_inv==idx).flatten()]
	# 	tile_choices.append(np.random.choice(tiles_in_zone,1))

	# tile_0 = np.nonzero(tile_choice[0] == all_tiles)[0]
	# tile_1 = np.nonzero(tile_choice[1] == all_tiles)[0]
	# test_indices = np.concatenate(tile_0,tile_1)
	# pass

	# all_zones = [i[-24:-21] for i in band_files] #19456
	# zones   = [i[1:4] for i in unique_t] #21

	#STATS.TXT
	# with open('../lake_chips/chips/stats.txt','r') as fp:
	# 	lines = fp.readlines()
	# stat_files  = [i.rstrip().split('\t')[0] for i in lines]
	# stat_pixels = [int(i.rstrip().split('\t')[1]) for i in lines]

	# stat_chips   = [i[0:-8] for i in stat_files]
	# stat_rasters = [i[0:-19] for i in stat_files]
	# stat_tiles   = [i[-25:-19] for i in stat_files]
	# stat_zones   = [i[-24:-21] for i in stat_files]
	# pass
 