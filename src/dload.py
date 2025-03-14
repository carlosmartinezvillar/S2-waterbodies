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


def get_split_indices():
	# if os.path.isfile('../cfg/training.txt') and os.path.isfile('../cfg/validation.txt'):
	assert os.path.isfile('../cfg/training.txt'), "dload.py: NO FILE FOUND FOR TRAINING INDICES."
	assert os.path.isfile('../cfg/validation.txt'),"dload.py: NO FILE FOUND FOR VALIDATION INDICES."
	assert os.path.isfile('../cfg/testing.txt'), "dload.py: NO FILE FOUND FOR TESTING INDICES."

	# LOAD INDICES FROM PREVIOULY SAVED FILES
	with open('../cfg/training.txt','r') as fp_tr:
		lines = fp_tr.readlines()
		training_idxs = [int(_.rstrip('\n')) for _ in lines]

	with open('../cfg/validation.txt','r') as fp_va:
		lines = fp_va.readlines()
		validation_idxs = [int(_.rstrip('\n')) for _ in lines]

	with open('../cfg/testing.txt','r') as fp_te:
		lines = fp_te.readlines()
		test_idxs = [int(_.rstrip('\n')) for _ in lines]

	return training_idxs,validation_idxs,test_idxs


def preprocess_potsdam():
	CHP_SIZE = 512
	pass

def preprocess_loveda():
	pass

def preprocess_isaid():
	pass

if __name__ == '__main__':
	print('-> dload.py')