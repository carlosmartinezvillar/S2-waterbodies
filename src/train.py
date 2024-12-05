import os
import numpy as np
import torch
import torchvision
import random
import time
from tqdm import tqdm
import argparse

import utils
import model
import dload

####################################################################################################
# SET GLOBAL VARS FROM ENV ET CETERA.
####################################################################################################
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #temp
ROOT = '../'
LOG_DIR = ROOT + 'log/'
MDL_DIR = ROOT + 'mod/'
HP = {}

__spec__ = None # DEBUG with tqdm/temp.

####################################################################################################
# FILE PATHS
####################################################################################################
ROOT    = "../"
LOG_DIR = ROOT + "log/"
MDL_DIR = ROOT + "mod/"


####################################################################################################
# FUNCTIONS
####################################################################################################
# def train_mixed_precision(model,loss_fn,optimizer,dataloader,device=0):
# 	model.train()
# 	sum_loss = 0
# 	for batch_idx,(X,T) in enumerate(dataloader):
# 		X,T = X.cuda(device,non_blocking=True), T.cuda(device,non_blocking=True)
# 		optimizer.zero_grad()
# 		with torch.cuda.amp.autocast(enabled=True,dtype=toch.float16):
# 			Y = model(X)
# 			loss = loss_fn(Y,T)
# 		scaler.scale(loss).backward()
# 		scaler.unscale_(optimizer) #unscale for loss calc at fp32
# 		sum_loss += loss.item()
# 		scaler.step(optimizer)
# 		scaler.update()


def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler=None,n_epochs=50):

	N_tr = len(dataloaders['training'].dataset)
	N_va = len(dataloaders['validation'].dataset)
	best_acc   = 0.0
	best_epoch = 0
	total_start_time = time.time()

	for epoch in range(n_epochs):
		M_tr = utils.ConfusionMatrix(n_classes=2)
		M_va = utils.ConfusionMatrix(n_classes=2)
		epoch_start_time = time.time()
		print(f'\nEpoch {epoch}/{n_epochs-1}')
		print('-'*80)

		############################################################
		# TRAINING
		############################################################		
		t = tqdm(total=len(dataloaders['training']),ncols=80)

		loss_sum    = 0.0
		samples_ran = 0
		model.train()

		for X,T in dataloaders['training']

			X = X.to(cuda_device)
			T = T.to(cuda_device)

			with torch.set_grad_enabled(True):
				outputs = model(X)
				Y       = torch.max()

			loss_sum = loss.item() * X.size(0)
			samples_ran += X.size(0)
			M_tr.update(Y.detach().cpu().numpy())

			t.set_postfix(loss='{:05.4f}'.format(loss_sum/samples_ran))
			t.update(1)

		t.close()
		loss_tr = loss_sum / N_tr
		print(f'[T] Loss: {loss_tr:.4f}')

		if scheduler is not None:
			scheduler.step()

		############################################################
		# VALIDATION
		############################################################
		t = tqdm(total=len(dataloaders['validation']),ncols=80)

		loss_sum    = 0.0
		samples_ran = 0
		model.eval()

		for X,T in dataloaders['validation']:

			X = X.to(cuda_device)
			T = T.to(cuda_device)



			loss_sum = loss.item() * X.size(0)
			samples_ran += X.size(0)
			M_va.update(Y.detach().cpu().numpy(),T.detach.cpu().numpy())

			t.set_postfix(loss='{:05.4f}'.format(loss_sum/samples_ran))
			t.update(1)

		t.close()

		loss_va = loss_sum / N_va

if __name__ == "__main__":
	pass