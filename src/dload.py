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
class LabelDivTransform(torch.nn.Module):
	'''
	Labels in the S2-DW dataset are stored as 255 and 0 for two classes and 255,
	127, and 0 for 3 classes. This is for ease of visualization/inspection.
	Labels are converted to values in [1,0] or [2,1,0] by floor division. In 
	either case water is 1 and land is 0.
	'''
	def __init__(self,lbl_div):
		super().__init__()
		self.lbl_div = lbl_div

	def forward(self,lbl):
		'''
		Parameters
		----------
		lbl : torch.Tensor
		    The label array. It's ingested with dimension [1,H,W,L], i.e.: height, 
		    width, and L number of 2 or 3 classes.
		Returns
		-------
		torch.Tensor
		    Converted label with values 0 and 1 for the binary case; and 0,1, and 2
		    for three class arrays.	
		'''		
		lbl = torch.squeeze(lbl,0)
		lbl = torch.div(lbl,self.lbl_div,rounding_mode='floor')
		return lbl


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
			# v2.Lambda(lambda x: torch.squeeze(x,0)),
			# v2.Lambda(lambda x: torch.div(x,lbl_div,rounding_mode='floor')),
			LabelDivTransform(lbl_div=lbl_div), #Removed lambda for ddp func pickling
			v2.ToDtype(torch.int64)
		])

		self.additional_transform = transform #applied only to train, not valdtn/test

	def rgb_get(self,idx):
		r,g,b,_ = Image.open(f'{self.ids[idx]}_B0X.tif').split()		
		return Image.merge(mode='RGB',bands=[r,g,b])

	def vnir_get(self,idx):
		return Image.open(f'{self.ids[idx]}_B0X.tif')

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,idx):
		image = self.input_transform(self.input_func(idx))
		label = self.label_transform(Image.open(f'{self.ids[idx]}_LBL.tif'))
		if self.additional_transform:
			image,label = self.additional_transform(image,label)
		# return image,label,self.ids[idx]
		return image,label


class PotsdamDataset(torch.utils.data.Dataset):
	'''
	512x512
	0:void,1:impervious,2:building,3:low veg,4:tree,5:car,6:clutter/backg #6CLASS
	0:void+clutter,1:impervious,2:building,3:low veg,4:tree,5:car #5CLASS
	IGNORE 0 in LOSS ! 
	'''
	def __init__(self,chip_dir,train,split,validate,transform=None):
		self.root_dir = chip_dir

		if train == True:
			#For both
			self.img_dir = f"{self.root_dir}/img_dir/train/"
			self.ann_dir = f"{self.root_dir}/ann_dir/train/"

			#LEAVE OUT TILES
			if split == True: 
				vtiles = ['2_10','7_9']
				if validate == True:
					self.img_paths = []
					self.ann_paths = []
					for tile in vtiles:
						self.img_paths.append(sorted(glob.glob(f"{self.img_dir}{tile}*.png")))
						self.ann_paths.append(sorted(glob.glob(f"{self.ann_dir}{tile}*.png")))
				else:
					all_ids = glob.glob("*.png",root_dir=self.img_dir) #ann and img match	
					filtered = [f for f in all_ids if not os.path.basename(f).startswith(tuple(vtiles))]
					self.img_paths = [f"{self.img_dir}{f}" for f in filtered]
					self.ann_paths = [f"{self.ann_dir}{f}" for f in filtered]

			#ENTIRE TRAIN FOLDER
			else: 
				self.img_paths = sorted(glob.glob(f"{self.img_dir}*.png"))
				self.ann_paths = sorted(glob.glob(f"{self.ann_dir}*.png"))

		else:
			self.img_dir = f"{self.root_dir}/img_dir/val/"
			self.ann_dir = f"{self.root_dir}/ann_dir/val/"
			self.img_paths = sorted(glob.glob(f"{self.img_dir}*.png"))
			self.ann_paths = sorted(glob.glob(f"{self.ann_dir}*.png"))

		self.img_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32,scale=True)
		])

		self.ann_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.int64)
		])

		self.additional_transform = transform


	def __len__(self):
		return len(self.ann_paths)


	def __getitem__(self,idx):
		image = self.img_transform(Image.open(f"{self.img_paths[idx]}"))
		label = self.ann_transform(Image.open(f"{self.ann_paths[idx]}"))
		if self.additional_transform:
			image,label = self.additional_transform(image,label)
		return image,label


class LoveDADataset(torch.utils.data.Dataset):
	# 1024x1024
	# No-Data     0): 255, 255, 255 (White) — IGNORE IN LOSS!
	# Background  1): 255, 0, 0 (Red)
	# Building    2): 255, 255, 0 (Yellow)
	# Road        3): 128, 0, 128 (Purple)
	# Water       4): 0, 0, 255 (Blue)
	# Barren      5): 255, 165, 0 (Orange)
	# Forest      6): 0, 128, 0 (Green)
	# Agriculture 7): 0, 255, 0 (Lime)
	def __init__(self,chip_dir,train,transform=None):
		self.root_dir = chip_dir

		if train == True: # TEST SET IS ONLINE
			self.img_dir = f"{self.root_dir}/img_dir/train/"
			self.ann_dir = f"{self.root_dir}/ann_dir/train/"
			self.img_paths = sorted(glob.glob(f"{self.img_dir}*.png"))
			self.ann_paths = sorted(glob.glob(f"{self.ann_dir}*.png"))
		else:
			self.img_dir = f"{self.root_dir}/img_dir/val/"
			self.ann_dir = f"{self.root_dir}/ann_dir/val/"
			self.img_paths = sorted(glob.glob(f"{self.img_dir}*.png"))
			self.ann_paths = sorted(glob.glob(f"{self.ann_dir}*.png"))

		self.img_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32,scale=True)
		])

		self.ann_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.int64)
		])

		self.additional_transform = transform


	def __len__(self):
		return len(self.ann_paths)


	def __getitem__(self,idx):
		image = self.img_transform(Image.open(f"{self.img_paths[idx]}"))
		label = self.ann_transform(Image.open(f"{self.ann_paths[idx]}"))
		if self.additional_transform:
			image,label = self.additional_transform(image,label)
		return image,label


class iSAIDDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,train,split,validate,transform=None,):
		'''
		896x896 with 512 overlap
		Background          0): 0, 0, 0  <--- IGNORE IN LOSS!
		Ship                1): 0, 0, 63
		Store Tank          2): 0, 63, 0
		Baseball Diamond    3): 0, 63, 127
		Tennis Court        4): 0, 63, 191
		Basketball Court    5): 0, 63, 255
		Ground Track Field  6): 0, 127, 0
		Bridge              7): 0, 127, 63
		Large Vehicle       8): 0, 127, 127
		Small Vehicle       9): 0, 127, 191
		Helicopter         10): 0, 127, 255
		Swimming Pool      11): 0, 191, 0
		Roundabout         12): 0, 191, 63
		Soccer Ball Field  13): 0, 191, 127
		Plane              14): 0, 191, 191
		Harbor             15): 0, 191, 255
		'''
		self.root_dir = chip_dir

		if train == True:
			#For both
			self.img_dir = f"{self.root_dir}/img_dir/train/"
			self.ann_dir = f"{self.root_dir}/ann_dir/train/"

			#LEAVE OUT TILES
			if split == True: 
				vtiles = ['P0003','P0011','P0002'] #Urban,coastal/harbor,aviation/scale
				if validate == True:
					self.img_paths = []
					self.ann_paths = []
					for tile in vtiles:
						self.img_paths.append(sorted(glob.glob(f"{self.img_dir}{tile}_*.png")))
						self.ann_paths.append(sorted(glob.glob(f"{self.ann_dir}{tile}_*.png")))
				else:
					all_ids = glob.glob("*.png",root_dir=self.img_dir) #ann and img match	
					filtered = [f for f in all_ids if not os.path.basename(f).startswith(tuple(vtiles))]
					self.img_paths = [f"{self.img_dir}{f}" for f in filtered]
					self.ann_paths = [f"{self.ann_dir}{f}" for f in filtered]

			#ENTIRE TRAIN FOLDER
			else: 
				self.img_paths = sorted(glob.glob(f"{self.img_dir}*.png"))
				self.ann_paths = sorted(glob.glob(f"{self.ann_dir}*.png"))			


		else:
			self.img_dir = f"{self.root_dir}/img_dir/val/"
			self.ann_dir = f"{self.root_dir}/ann_dir/val/"
			self.img_paths = sorted(glob.glob(f"{self.img_dir}*.png"))
			self.ann_paths = sorted(glob.glob(f"{self.ann_dir}*.png"))

	def __len__(self):
		return len(self.ann_paths)

	def __getitem__(self,idx):
		image = self.img_transform(Image.open(f"{self.img_paths[idx]}"))
		label = self.ann_transform(Image.open(f"{self.ann_paths[idx]}"))
		if self.additional_transform:
			image,label = self.additional_transform(image,label)
		return image,label

################################################################################
# EXTERNAL DATASETS PREPROCESSING -- If further pre-process needed
################################################################################
def preprocess_potsdam():
	CHP_SIZE = 512
	pass


def preprocess_loveda():
	CHP_SIZE = 1024
	pass


def preprocess_isaid():
	CHP_SIZE = 896
	pass


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':

	__spec__ = None #multiprocessing w/ interactive/debugger

	from tqdm import tqdm

	print('-> dload.py')

	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir',required=True,help='dataset directory.')
	args = parser.parse_args()
	DATA_DIR  = args.data_dir
	print(f'DATA_DIR set to {DATA_DIR}')

	# SET SEED
	utils.set_seed(476)

	transform = v2.Compose([
		v2.RandomHorizontalFlip(p=0.5),
		v2.RandomVerticalFlip(p=0.5)
	])

	############################################################
	# S2-DW
	############################################################
	# s2dw_tr_ds = SentinelDataset(f"{DATA_DIR}/training",
	# 	n_bands=3,
	# 	n_labels=2,
	# 	transform=transform)

	# s2dw_va_ds = SentinelDataset(f"{DATA_DIR}/validation",
	# 	n_bands=3,
	# 	n_labels=2,
	# 	transform=None)
	# dataloaders = {
	# 	'training': torch.utils.data.DataLoader(
	# 		s2dw_tr_ds,
	# 		batch_size=8,
	# 		drop_last=False,
	# 		shuffle=True,
	# 		num_workers=3),
	# 	'validation': torch.utils.data.DataLoader(
	# 		s2dw_va_ds,
	# 		batch_size=8,
	# 		drop_last=False,
	# 		shuffle=False,
	# 		num_workers=0)
	# }
	# print("\nS2-DW -- TESTING BATCH ITERATION")

	############################################################
	# POTSDAM
	############################################################
	# potsdam_tr_ds = PotsdamDataset(f"{DATA_DIR}",train=True,split=True,validate=False,transform=transform)
	# potsdam_va_ds = PotsdamDataset(f"{DATA_DIR}",train=True,split=True,validate=True)
	# dataloaders = {
	# 	'training': torch.utils.data.DataLoader(
	# 		potsdam_tr_ds,
	# 		batch_size=8,
	# 		drop_last=False,
	# 		shuffle=True,
	# 		num_workers=3),
	# 	'validation': torch.utils.data.DataLoader(
	# 		potsdam_va_ds,
	# 		batch_size=8,
	# 		drop_last=False,
	# 		shuffle=False,
	# 		num_workers=0)
	# }
	# print("\nPOTSDAM -- TESTING BATCH ITERATION")


	############################################################
	# LOVEDA
	############################################################
	# loveda_tr_ds = LoveDADataset(f"{DATA_DIR}",train=True,transform=transform)
	# loveda_va_ds = LoveDADataset(f"{DATA_DIR}",train=False)
	# dataloaders = {
	# 	'training': torch.utils.data.DataLoader(
	# 		loveda_tr_ds,
	# 		batch_size=8,
	# 		drop_last=False,
	# 		shuffle=True,
	# 		num_workers=3),
	# 	'validation': torch.utils.data.DataLoader(
	# 		loveda_va_ds,
	# 		batch_size=8,
	# 		drop_last=False,
	# 		shuffle=False,
	# 		num_workers=0)
	# }
	# print("\nLOVEDA -- TESTING BATCH ITERATION")


	############################################################
	# iSAID
	############################################################
	isaid_tr_ds = iSAIDDataset(f"{DATA_DIR}",train=True,split=True,validate=False,transform=transform)
	isaid_va_ds = iSAIDDataset(f"{DATA_DIR}",train=True,split=True,validate=True)
	dataloaders = {
		'training': torch.utils.data.DataLoader(
			isaid_tr_ds,
			batch_size=8,
			drop_last=False,
			shuffle=True,
			num_workers=3),
		'validation': torch.utils.data.DataLoader(
			isaid_va_ds,
			batch_size=8,
			drop_last=False,
			shuffle=False,
			num_workers=0)
	}
	print("\niSAID -- TESTING BATCH ITERATION")

	################
	# ITERATE
	################
	for epoch in range(2):
		print(f"\nepoch nr. {epoch}")
		print("-"*20)
		t = tqdm(total=len(dataloaders['training']),ncols=80,ascii=True)
		for i, (X,T) in enumerate(dataloaders['training']):
			pass
			t.update(1)
		t.close()

	pass
