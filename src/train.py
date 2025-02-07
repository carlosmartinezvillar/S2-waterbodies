import os
import numpy as np
import torch
import torchvision
import random
import time
from tqdm import tqdm
import argparse
import json

import utils
import model
import dload

####################################################################################################
# SET GLOBAL VARS FROM ENV ET CETERA ET CETERA
####################################################################################################
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #temp

__spec__ = None # DEBUG with tqdm -- temp.

parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--data-dir',required=True,
	help='Dataset directory.')
required.add_argument('--model-dir',required=True,
	help='Directory to store the trained models/weights.')
required.add_argument('--log-dir',required=True,default='../log',
	help="Directory to store training logs.")
required.add_argument('--params',required=True,default='../hpo/parameters.json',
	help='Path to file listing hyperparameters.')
required.add_argument('--row',required=True,type=int,default=0,
	help='Row number in parameter file listed in --params')
optional.add_argument('--seed',required=False,action='store_true',
	help='Fix the random seed of imported modules for reproducibility.')

args = parser.parse_args()

DATA_DIR  = args.data_dir
LOG_DIR   = args.log_dir
MODEL_DIR = args.model_dir

####################################################################################################
# TRAIN FUNCTION
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

		if scheduler is not None:
			scheduler.step()

		# log training
		loss_tr = loss_sum / N_tr
		print(f'[T] loss: {loss_tr:.4f} | acc: {M_tr.acc():.4f}')

		
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

		############################################################
		# LOG EPOCH
		############################################################
		epoch_time = time.time() - epoch_start_time
		print(f'Epoch time: {epoch_time:.2f}')
		epoch_log = [loss_tr,M_tr.acc(),loss_va,M_va.acc(),M_va.tpr(),M_va.ppv(),M_va.iou()]
		epoch_logger.log(epoch_log)

		# SAVE MODEL
		epoch_iou = M_va.iou()
		if best_iou < epoch_iou:
			best_iou = epoch_iou
			best_epoch = epoch
			utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,loss_tr,loss_va,best=True)

		print(f'\nBest validation IoU: {best_acc:.4f}')
		total_time = time.time() - total_start_time
		print(f'\nTotal time: {total_time:.2f}')


if __name__ == "__main__":

	# HP = {
	# 	'ID':0,
	# 	'LEARNING_RATE': 0.001,
	# 	'SCHEDULER':"step",
	# 	'OPTIM': "adam",
	# 	'LOSS': "ce",				
	# 	'BATCH': 16,
	# 	'INIT': "random",
	# 	'MODEL': "unet1_1"
	# }

	# LOAD AND PARSE HP DICT
	assert os.path.isfile(args.params), "train.py: INCORRECT JSON FILE PATH"
	with open(args.params,'r') as fp:
		HP_LIST = json.load(fp)
	assert len(HP_LIST) > 0, "train.py: EMPTY JSON FILE"
	assert 0 <= args.row < len(HP_LIST), "train.py: ROW arg OUT OF RANGE"
	HP = HP_LIST[args.row]

	# MODEL
	model_str = HP['MODEL'][0:4]
	assert model_str in ["attn","unet"], "train.py: INCORRECT MODEL STRING."
	if model_str == 'unet':
		exec(f"net = model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']})")
	if model_str == 'attn':
		pass

	# MODEL -- TO GPU
	net = net.to(cuda_device) #checked above

	# LOSS
	assert HP['LOSS'] in ["ce","ew","cw"], "train.py: INCORRECT STRING FOR LOSS IN DICT."
	if HP['LOSS'] == "ce":
		loss_fn = torch.nn.CrossEntropyLoss()
	if HP['LOSS'] == "ew":
		loss_fn = None
	if HP['LOSS'] == "cw":
		loss_fn = None


	# OPTIMIZER
	assert HP["OPTIM"] in ["adam","lamb"], "train.py: INCORRECT STRING FOR OPTIMIZER IN DICT."
	if HP['OPTIM'] == "adam":
		optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == "lamb":
		optimizer = None


	# LEARNING RATE SCHEDULER
	if HP['SCHEDULER'] == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
	elif HP['SCHEDULER'] == "linear":
		scheduler = None
	elif HP['SCHEDULER'] == "exp":
		scheduler = None
	else:
		scheduler = None	

	# SET ALL SEEDS
	if args.seed is True:
		utils.set_seed(476)	

	#DATALOADERS -- SPLIT DATASET INTO TRAINING VALIDATION --- GOTTA FIX THE SPLIT BEFORE-HAND:TODO
	tr_idx,va_idx,te_idx = dload.test_validation_split(DATA_DIR)
	dataset              = dload.SentinelDataset(DATA_DIR)
	tr_dataset = torch.utils.data.Subset(dataset,tr_idx)
	va_dataset = torch.utils.data.Subset(dataset,va_idx)
	# te_dataset = torch.utils.data.Subset(dataset,te_idx)

	dataloaders = {
		'training': torch.utils.data.DataLoader(tr_dataset,batch_size=HP['BATCH'],
			drop_last=False,shuffle=True,num_workers=2),
		'validation': torch.utils.data.DataLoader(va_dataset,batch_size=HP['BATCH'],
			drop_last=False,shuffle=False,num_workers=2)
	}


	# RUN
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
