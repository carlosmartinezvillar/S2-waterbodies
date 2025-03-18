import os
import numpy as np
import torch
import torchvision
import random
import time
from tqdm import tqdm
import argparse
import json
from functools import wraps

import utils
import model
import dload

####################################################################################################
# SET GLOBAL VARS FROM ENV ET CETERA ET CETERA
####################################################################################################
__spec__ = None # DEBUG with tqdm -- temp.

parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--data-dir',required=True,
	help='Dataset directory.')
required.add_argument('--net-dir',required=True,
	help='Directory to store the trained models/weights.')
required.add_argument('--log-dir',required=True,default='../log',
	help="Directory to store training logs.")
required.add_argument('--params',required=True,default='../hpo/parameters.json',
	help='Path to file listing hyperparameters.')
required.add_argument('--row',required=True,type=int,default=0,
	help='Row number in the given file for hyperparameters.')
# optional.add_argument('--seed',required=False,action='store_true',
	# help='Fix the random seed of imported modules for reproducibility.') --> moved to HParams
optional.add_argument('--gpu',required=False,type=int,default=0)
optional.add_argument('--multi-gpu',required=False,type=bool,action='store_true',default=False)
args = parser.parse_args()

DATA_DIR  = args.data_dir
LOG_DIR   = args.log_dir
MODEL_DIR = args.net_dir
CUDA_DEV  = None

def total_time_decorator(orig_func):
	@wraps(orig_func)
	def wrapper(*args, **kwargs):
		total_time_start = time.time()
		orig_func(*args,**kwargs)
		total_time = time.time() - total_time_start
		print(f'TOTAL TRAINING TIME: {total_time:.2f}s')

	return wrapper

####################################################################################################
# TRAININING+VALIDATION
####################################################################################################
@total_time_decorator
def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler=None,n_epochs=50):

	N_tr = len(dataloaders['training'].dataset)
	N_va = len(dataloaders['validation'].dataset)
	best_iou   = 0.0
	best_epoch = 0
	epoch_header = ["tloss","t_acc","vloss","v_acc","v_tpr","v_ppv","v_iou"]
	epoch_logger = utils.Logger(f'{LOG_DIR}/epoch_log_{model.model_id:03}.tsv',epoch_header)
	tr_batch_loss = []
	va_batch_loss = []

	for epoch in range(n_epochs):
		M_tr = utils.ConfusionMatrix(n_classes=2)
		M_va = utils.ConfusionMatrix(n_classes=2)
		epoch_start_time = time.time()
		print(f'\nEpoch {epoch}/{n_epochs-1}')
		print('-'*80)

		############################################################
		# TRAINING
		############################################################		
		t = tqdm(total=len(dataloaders['training']),ncols=80,ascii=True)

		tr_loss_sum = 0.0
		samples_ran = 0
		model.train()

		for X,T in dataloaders['training']:

			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			# forward-pass
			with torch.set_grad_enabled(True):
				output = model(X)
				loss   = loss_fn(output,T)

			# backprop
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# metrics
			tr_loss_sum += loss.item() * X.size(0)
			samples_ran += X.size(0)			
			Y = output.detach().cpu().numpy().max(axis=1)
			T = T.detach().cpu().numpy()
			M_tr.update(Y,T)

			# update bar
			t.set_postfix(loss='{:05.4f}'.format(tr_loss_sum/samples_ran))
			t.update(1)
		
		t.close()

		if scheduler is not None:
			scheduler.step()

		# log training
		loss_tr = tr_loss_sum / N_tr
		print(f'[T] loss: {loss_tr:.5f} | acc: {M_tr.acc():.5f} | iou: {M_tr.iou():.5f}')

		
		############################################################
		# VALIDATION
		############################################################
		t = tqdm(total=len(dataloaders['validation']),ncols=80,ascii=True)

		va_loss_sum = 0.0
		samples_ran = 0
		model.eval()

		for X,T in dataloaders['validation']:

			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			with torch.set_grad_enabled(False):
				output = model(X)
				loss   = loss_fn(output,T)
				_,Y    = torch.max(output,1)

			#metrics 
			va_loss_sum += loss.item() * X.size(0)
			samples_ran += X.size(0)
			M_va.update(Y.cpu().numpy(),T.cpu().numpy())

			t.set_postfix(loss='{:05.4f}'.format(va_loss_sum/samples_ran))
			t.update(1)

		t.close()
		# log validation
		loss_va = va_loss_sum / N_va
		print(f'[V] loss: {loss_va:.5f} | acc: {M_va.acc():.5f} | iou: {M_va.iou():.5f}')

		############################################################
		# LOG EPOCH
		############################################################
		epoch_time = time.time() - epoch_start_time
		print(f'\nEpoch time: {epoch_time:.2f}s')
		epoch_result = [loss_tr,M_tr.acc(),loss_va,M_va.acc(),M_va.tpr(),M_va.ppv(),M_va.iou()]
		epoch_logger.log(epoch_result)

		# SAVE MODEL
		epoch_iou = M_va.iou()
		if best_iou < epoch_iou:
			best_iou = epoch_iou
			best_epoch = epoch
			utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,loss_tr,loss_va,best=True)

		with open()

		print(f'Best validation IoU: {best_iou:.4f}')


if __name__ == "__main__":
	#---------- LOAD AND PARSE HP DICT ----------
	#some checks
	assert os.path.isfile(args.params), "train.py: INCORRECT JSON FILE PATH"
	with open(args.params,'r') as fp:
		HP_LIST = json.load(fp)
	assert len(HP_LIST) > 0, "train.py: GOT EMPTY JSON FILE."
	assert 0 <= args.row < len(HP_LIST), "train.py: OUT OF RANGE ROW ARGUMENT." #0-indexed

	#load dictionary
	HP = HP_LIST[args.row]

	#---------- GPU (IF SET) ----------
	# assert torch.cuda.is_available(), "train.py: torch.cuda.is_available() returned False"
	# assert torch.cuda.device_count() > 0, "train.py: number of CUDA devices is zero."
	# assert args.gpu < torch.cuda.device_count(), "train.py: GPU INDEX OUT OF RANGE."
	
	if torch.cuda.is_available():
		CUDA_DEV = torch.device(f"cuda:{args.gpu}")
	else:
		CUDA_DEV = torch.device("cpu")

	#---------- MULTI-GPU (IF SET) ---------- <---------- TODO!


	#---------- MODEL ----------
	model_str = HP['MODEL'][0:4]
	assert model_str in ["attn","unet"], "train.py: INCORRECT MODEL STRING."
	if model_str == 'unet':
		exec(f"net = model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']})")
	if model_str == 'attn':
		pass
	# ---> TO GPU
	net = net.to(CUDA_DEV) #checked above

	#---------- LOSS ----------
	assert HP['LOSS'] in ["ce","ew","cw"], "train.py: INCORRECT STRING FOR LOSS IN DICT."
	if HP['LOSS'] == "ce":
		loss_fn = torch.nn.CrossEntropyLoss()
	if HP['LOSS'] == "ew":
		loss_fn = None
	if HP['LOSS'] == "cw":
		loss_fn = None


	#---------- OPTIMIZER ----------
	assert HP["OPTIM"] in ["adam","lamb"], "train.py: INCORRECT STRING FOR OPTIMIZER IN DICT."
	if HP['OPTIM'] == "adam":
		optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == "lamb":
		optimizer = None


	#---------- LEARNING RATE SCHEDULER ----------
	if HP['SCHEDULER'] == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
	elif HP['SCHEDULER'] == "linear":
		scheduler = None
	elif HP['SCHEDULER'] == "exp":
		scheduler = None
	else:
		scheduler = None	

	#---------- SET ALL SEEDS ----------
	assert HP['SEED'] in (0,1), "train.py: INCORRECT SEED IN JSON PARAMETER DICT."
	if HP['SEED'] == True:
		utils.set_seed(476)	

	#---------- DATALOADERS ----------
	tr_idx,va_idx,te_idx = dload.sentinel_split_indices()
	dataset              = dload.SentinelDataset(DATA_DIR)
	tr_dataset = torch.utils.data.Subset(dataset,tr_idx)
	va_dataset = torch.utils.data.Subset(dataset,va_idx)
	dataloaders = {
		'training': torch.utils.data.DataLoader(tr_dataset,batch_size=HP['BATCH'],
			drop_last=False,shuffle=True,num_workers=2),
		'validation': torch.utils.data.DataLoader(va_dataset,batch_size=HP['BATCH'],
			drop_last=False,shuffle=False,num_workers=2)
	}

	#---------- RUN ----------
	train_and_validate(net,dataloaders,optimizer,loss_fn,scheduler,n_epochs=50)

