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
	def __init__(self,chip_dir,n_bands=3,n_labels=2):
		self.dir        = chip_dir
		self.vnir_files = sorted(glob.glob("*_B0X.tif",root_dir=self.dir))
		self.ids        = [i[0:-8] for i in self.vnir_files]

		if (n_bands!=3) and (n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")

		if n_bands == 3:
			self.input_func = self.rgb_get
		if n_bands == 4:
			self.input_func = self.vnir_get

		self.transform = v2.Compose([
			v2.ToImage(),
			# v2.ToDtype(torch.float32, scale=True)
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5)
		])

	def rgb_get(self,idx):
		r,g,b,_ = Image.open(f'{self.dir}/{self.ids[idx]}_B0X.tif').split()
		return Image.merge(mode='RGB',bands=[r,g,b])

	def vnir_get(self,idx):
		return Image.open(f'{self.dir}/{self.ids[idx]}_B0X.tif')
		
	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		img = self.input_func(idx)
		lbl = Image.open(f'{self.dir}/{self.ids[idx]}_LBL.tif')

		img,lbl = self.transform(img,lbl)

		img = img.to(torch.float32).div(255)
		lbl = lbl.squeeze(0).div(255,rounding_mode='floor').to(torch.int64)
		return img,lbl


class PotsdamDataset(torch.utils.data.Dataset):
	def __init__(self,root_dir):
		pass

	def __len__(self):
		pass

	def __getitem__(self,i):
		pass
		return


class LoveDADataset(torch.utils.data.Dataset):
	def __init__(self,root_dir):
		pass

	def __len__(self):
		pass

	def __getitem__(self,i):
		pass
		return


class iSaidDataset(torch.utils.data.Dataset):
	def __init__(self,root_dir):
		pass

	def __len__(self):
		pass

	def __getitem__(self,i):
		pass
		return


def sentinel_split_indices():
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
	### TEST DATALOADERS --- TODO
	ds = SentinelDataset('../../chips_sorted/validation')
	print(ds[0])
	pass
