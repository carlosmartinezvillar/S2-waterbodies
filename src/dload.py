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
import argparse

import utils
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

		# def help_label_transform(x): # --forkingpickler not working--
			# return torch.div(torch.squeeze(x,0),lbl_div,rounding_mode='floor')

		self.label_transform = v2.Compose([
			v2.ToImage(),
			v2.Lambda(lambda x: torch.squeeze(x,0)),
			v2.Lambda(lambda x: torch.div(x,lbl_div,rounding_mode='floor')),
			# v2.Lambda(help_label_transform), # --forkingpickler not working--
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
		return img,lbl,self.ids[idx]


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


################################################################################
# EXTERNAL DATASETS PREPROCESSING
################################################################################
def preprocess_potsdam():
	CHP_SIZE = 512
	pass


def preprocess_loveda():
	pass


def preprocess_isaid():
	pass


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	__spec__ = None #multiprocessing w/ interactive/debugger

	print('-> dload.py')

	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir',required=True,help='dataset directory.')
	args = parser.parse_args()
	DATA_DIR  = args.data_dir
	# DATA_DIR = '../../chips_sorted_256'
	print(f'DATA_DIR set to {DATA_DIR}')

	# SET SEED
	utils.set_seed(476)

	transform = v2.Compose([
		v2.RandomHorizontalFlip(p=0.5),
		v2.RandomVerticalFlip(p=0.5)
	])

	tr_ds = SentinelDataset(f"{DATA_DIR}/training",
		n_bands=3,
		n_labels=2,
		transform=transform)

	va_ds = SentinelDataset(f"{DATA_DIR}/validation",
		n_bands=3,
		n_labels=2,
		transform=None)

	dataloaders = {
		'training': torch.utils.data.DataLoader(
			tr_ds,
			batch_size=8,
			drop_last=False,
			shuffle=True,
			num_workers=3),
		'validation': torch.utils.data.DataLoader(
			va_ds,
			batch_size=8,
			drop_last=False,
			shuffle=False,
			num_workers=0)
	}


	print("\nTESTING BATCH ITERATION")
	print("-"*40)
	for epoch in range(5):
		print(f"\nepoch nr. {epoch}")
		print("-"*20)
		for i,(X,T,img_path) in enumerate(dataloaders['training']):
			if i==0:
				print("First batch")
				for p in img_path:print(p)
			pass


	# print("-"*40)
	# print(f"CHECKING {tr_ds.__class__.__name__}")
	# x,t = tr_ds[0]
	# print(f'SHAPE: {x.shape} {x.dtype}\n{t.shape} {t.dtype}')
	# print("-"*40)	
	# start = time.time()
	# for x,t in ds:
	# 	pass
	# delta = time.time() - start
	# print(f"TOTAL TIME: {delta:.5f} | SAMPLES: {len(ds)}")
	pass
