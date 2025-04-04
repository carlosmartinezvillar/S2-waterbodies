import os
import time
import sys
import numpy as np
import torch
import torchvision as tv
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import glob

################################################################################
# CLASSES
################################################################################
class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3,n_labels=2,transform=None):
		self.dir        = chip_dir
		self.vnir_files = sorted(glob.glob(f"{chip_dir}/*_B0X.tif"))
		self.ids        = [i[0:-8] for i in self.vnir_files]

		if (n_bands!=3) and (n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")

		if n_bands == 3:
			self.input_func = self.rgb_get
		if n_bands == 4:
			self.input_func = self.vnir_get

		if (n_labels)!=3 and (n_labels!=2):
			raise ValueError("Incorrect number of target labels.")

		if n_labels == 2:
			lbl_div = 255
		if n_labels == 3:
			lbl_div = 127

		self.input_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32,scale=True)		
		])

		self.label_transform = v2.Compose([
			v2.ToImage(),
			v2.Lambda(lambda x: torch.squeeze(x,0)),
			v2.Lambda(lambda x: torch.div(x,lbl_div,rounding_mode='floor')),
			v2.ToDtype(torch.int64)
		])

		self.joint_transform = transform #outside instance for validation/testing

	def rgb_get(self,idx):
		r,g,b,_ = Image.open(f'{self.ids[idx]}_B0X.tif').split()		
		return Image.merge(mode='RGB',bands=[r,g,b])

	def vnir_get(self,idx):
		return Image.open(f'{self.ids[idx]}_B0X.tif')

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		img = self.input_transform(self.input_func(idx))
		lbl = self.label_transform(Image.open(f'{self.ids[idx]}_LBL.tif'))
		if self.joint_transform:
			img,lbl = self.joint_transform(img,lbl)
		return img,lbl


class PotsdamDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir):
		self.chip_dir = chip_dir
		self.paths    = sorted(glob.glob(f"{self.chip_dir}/*.tif"))

		self.joint_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32,scale=True)
		])
		pass

	def __len__(self,idx):
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
	transform = v2.Compose([
		v2.RandomHorizontalFlip(p=0.5),
		v2.RandomVerticalFlip(p=0.5)
	])

	ds = SentinelDataset('../../chips_sorted/validation',n_bands=3,n_labels=2,transform=None)
	print("-"*40)
	print(f"CHECKING {ds.__class__.__name__}")
	print("-"*40)	
	x,t = ds[0]
	print(f'{x.shape} {x.dtype}\n{t.shape} {t.dtype}')
	start = time.time()
	for x,t in ds:
		pass
	delta = time.time() - start
	print(f"TOTAL TIME: {delta:.5f} | SAMPLES: {len(ds)}")
	pass
