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

__spec__ = None # DEBUG with tqdm -- temp.

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',default='../dat',
	help='Dataset directory.')
# parser.add_argument('--model-dir',default='../mdl',
	# help='Directory storing the trained models/weights.')
# parser.add('--log-dir',default='../log',
	# help="Training log directory.")
parser.add_argument('--params',default='../exp/parameters.json',
	help='Path to file listing the hyperparameters to run.')
parser.add_argument('--id',default=0,
	help='A unique number identifying each model.')


args = parser.parse_args()

DATA_DIR = args.data_dir
LOG_DIR  = f'../log'
MDL_DIR  = f'{DATA_DIR}/mdl'

####################################################################################################
# FUNCTIONS
####################################################################################################
def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler=None,n_epochs=50):

	N_tr = len(dataloaders['training'].dataset)
	N_va = len(dataloaders['validation'].dataset)
	best_acc   = 0.0
	best_epoch = 0
	epoch_logger = utils.Logger(f'{LOG_DIR}/train_log_{model.model_id:03}.tsv',
		["tloss","t_acc","vloss","v_acc","v_tpr","v_ppv","v_iou"])

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

		for X,T in dataloaders['training']:

			X = X.to(cuda_device,non_blocking=True)
			T = T.to(cuda_device,non_blocking=True)

			# forward-pass
			with torch.set_grad_enabled(True):
				output = model(X)
				loss   = loss_fn(output,T)

			# backprop
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# metrics
			loss_sum = loss.item() * X.size(0)
			samples_ran += X.size(0)			
			Y = output.detach().cpu().numpy().max(axis=1)
			T = T.detach().cpu().numpy()
			M_tr.update(Y,T)

			# update bar
			t.set_postfix(loss='{:05.4f}'.format(loss_sum/samples_ran))
			t.update(1)
		
		t.close()
		# log training
		loss_tr = loss_sum / N_tr
		print(f'[T] loss: {loss_tr:.4f} | acc: {M_tr.acc():.4f}')

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

			X = X.to(cuda_device,non_blocking=True)
			T = T.to(cuda_device,non_blocking=True)

			with torch.set_grad_enabled(False):
				output = model(X)
				loss   = loss_fn(output,T)
				_,Y    = torch.max(output,1)

			#metrics 
			loss_sum = loss.item() * X.size(0)
			samples_ran += X.size(0)
			M_va.update(Y.cpu().numpy(),T.cpu().numpy())

			t.set_postfix(loss='{:05.4f}'.format(loss_sum/samples_ran))
			t.update(1)

		t.close()
		# log validation
		loss_va = loss_sum / N_va
		print(f'[V] loss: {loss_va:.4f} | acc: {M_va.acc():.4f}')


		# LOG EPOCH
		###########
		epoch_time = time.time() - epoch_start_time
		print(f'Epoch time: {epoch_time:.2f}')
		epoch_log = [loss_tr,M_tr.acc(),loss_va,M_va.acc(),M_va.tpr(),M_va.ppv(),M_va.iou()]
		epoch_logger.log(epoch_log)

		# SAVE MODEL
		epoch_metric = M_va.iou()
		if best_acc < epoch_metric:
			best_acc = epoch_metric
			best_epoch = epoch
			utils.save_checkpoint(model,optimizer,epoch,loss_tr,loss_va,best=True)

		print(f'\nBest validation IoU: {best_acc:.4f}')
		total_time = time.time() - total_start_time
		print(f'\nTotal time: {total_time:.2f}')


if __name__ == "__main__":

	HP = {
		'ID':"000",
		'LEARNING_RATE': 0.001,
		'SCHEDULER':"step",
		'OPTIM': "adam",
		'LOSS': "ce",				
		'BATCH': 16,
		'INIT': "random",
		'MODEL': "unet1_1"
	}

	# PARSE HP DICT HERE TO SET OPTIMIZER, SCHEDULER, LOSS_FN, MODEL, ETC. --------->TODO
	model_str = HP['MODEL'][0:4]
	if model_str == 'unet':
		exec(f"net = model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']})")
	elif model_str == 'attn':
		pass
	else:
		raise ValueError("INCORRECT MODEL NAME STRING.")

	net = net.to(cuda_device) #checked above

	#LOSS
	if HP['LOSS'] == "ce":
		loss_fn = torch.nn.CrossEntropyLoss()
	elif HP['LOSS'] == "ew":
		loss_fn = None
	elif HP['LOSS'] == "cw":
		loss_fn = None
	else:
		raise ValueError("INCORRECT STRING FOR LOSS HYPERPARAMETER.")

	#OPTIMIZER
	if HP['OPTIM'] == "adam":
		optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
	elif HP['OPTIM'] == "lamb":
		optimizer = None
	else:
		raise ValueError("INCORRECT STRING FOR OPTIMIZER HYPERPARAMETER.")

	#LEARNING RATE SCHEDULER
	if HP['SCHEDULER'] == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
	elif HP['SCHEDULER'] == "linear":
		scheduler = None
	elif HP['SCHEDULER'] == "exp":
		scheduler = None
	else:
		scheduler = None	

	#DATALOADERS
	#SPLIT DATASET INTO TRAININIG VALIDATION
	# ------------------------------------------------------------------------------->TODO
	tr_idx,va_idx,te_idx = dload.test_validation_split(DATA_DIR)
	dataset              = dload.SentinelDataset(DATA_DIR)
	tr_dataset = torch.utils.data.Subset(dataset,tr_idx)
	va_dataset = torch.utils.data.Subset(dataset,va_idx)
	# te_dataset = torch.utils.data.Subset(dataset,te_idx)

	dataloaders = {
		'training': torch.utils.data.DataLoader(tr_dataset,batch_size=HP['BATCH'],
			drop_last=False,shuffle=True,num_workers=2),
		'validation': torch.utils.data.DataLoader(va_dataset,batch_size=HP['BATCH'],
			drop_last=False,shuffle=True,num_workers=2)
	}

	train_and_validate(net,dataloaders,optimizer,loss_fn,scheduler,n_epochs=5)



### THE END...
################################################################################
################################################################################
################################################################################
################################################################################
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
