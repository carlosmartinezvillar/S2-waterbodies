import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
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
	help='Input dataset directory.')
required.add_argument('--net-dir',required=True,
	help='Final trained models/weights directory.')
required.add_argument('--log-dir',required=True,default='../log',
	help='Final training logs directory.')
required.add_argument('-p','--params',required=True,default='../hpo/parameters.json',
	help='Path to hyperparameters file.')
required.add_argument('--row',required=True,type=int,default=0,
	help='Row number in hyperparameter file.')
optional.add_argument('--gpu',required=False,type=int,default=0,
	help='GPU to train in. Useful for training locally.')
optional.add_argument('--multi-gpu',required=False,action='store_true',default=False,
	help='Use multiple GPUs to train.')
optional.add_argument('--full',required=False,action='store_true',default=False,
	help='Train on both training and validation sets (training final model).')
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
def train_full_set(model,dataloaders,optimizer,loss_fn,scheduler=None,n_epochs=100):
	'''
	Train the model with the full set (train+validation combined).
	'''
	N_tr = len(dataloaders['training'].dataset) #nr of samples
	N_va = len(dataloaders['validation'].dataset)
	B_tr = len(dataloaders['training']) #nr of batches
	B_va = len(dataloaders['validation'])
	best_iou   = 0.0
	best_epoch = 0
	loss_sum = 0.0


	#ENABLE GRAD
	model.train()	
	torch.set.set_grad_enabled(True)


	for epoch in range(n_epochs):
		batch_loss = []
		M = utils.ConfusionMatrix(n_classes=2)
		t = tqdm(total=B_tr+B_va,ncols=80,ascii=True)
		epoch_start_time = time.time()
		print(f'\nEpoch {epoch}/{n_epochs-1}')
		print('-'*80)		

		samples  = 0

		for X,T in dataloaders['training']:
			X.to(CUDA_DEV,non_blocking=True)
			T.to(CUDA_DEV,non_blocking=True)

			output = model(X)
			loss   = loss_fn(output,T)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_sum += loss.item() * X.size(0)
			samples  += X.size(0)
			Y = output.detach().cpu().numpy().argmax(axis=1)
			T = T.detach().cpu().numpy()
			M.update(Y,T)
			batch_loss.append(loss.item())

			t.set_postfix(loss='{:05.5f}'.format(loss_sum/samples))
			t.update(1)

		for X,T in dataloaders['validation']:
			X.to(CUDA_DEV,non_blocking=True)
			T.to(CUDA_DEV,non_blocking=True)

			output = model(X)
			loss   = loss_fn(output,T)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_sum += loss.item() * X.size(0)
			samples  += X.size(0)
			Y = output.detach().cpu().numpy().argmax(axis=1)
			T = T.detach().cpu().numpy()
			M.update(Y,T)
			batch_loss.append(loss.item())


			t.set_postfix(loss='{:05.5f}'.format(loss_sum/samples))
			t.update(1)

		t.close()

		if scheduler is not None:
			scheduler.step()

		loss_total = loss_sum / (N_tr+N_va)
		print(f'[T] loss: {loss_total:.5f} | acc: {M.acc():.5f} | iou: {M.iou():.5f}')

		epoch_time = time.time() - epoch_start_time
		print(f'\nEpoch time: {epoch_time:.2f}s')
		# epoch_result = [loss_tr,M_tr.acc(),loss_va,M_va.acc(),M_va.tpr(),M_va.ppv(),M_va.iou()]
		# epoch_logger.log(epoch_result)

		# SAVE MODEL
		epoch_iou = M.iou()
		if best_iou < epoch_iou:
			best_iou = epoch_iou
			best_epoch = epoch
			# utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,loss_tr,loss_va,best=True)

		print(f'Best validation IoU: {best_iou:.4f}')

@total_time_decorator
def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler=None,n_epochs=50):

	N_tr = len(dataloaders['training'].dataset)
	N_va = len(dataloaders['validation'].dataset)
	best_iou   = 0.0
	best_epoch = 0
	epoch_header = ["tloss","t_acc","vloss","v_acc","v_tpr","v_ppv","v_iou"]
	epoch_logger = utils.Logger(f'{LOG_DIR}/epoch_log_{model.model_id:03}.tsv',epoch_header)

	for epoch in range(n_epochs):
		tr_batch_loss = []
		va_batch_loss = []
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
			#to device
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

			# METRICS
			tr_loss_sum += loss.item() * X.size(0)
			samples_ran += X.size(0)			
			Y = output.detach().cpu().numpy().argmax(axis=1)
			T = T.detach().cpu().numpy()
			M_tr.update(Y,T)
			tr_batch_loss.append(loss.item())

			# update bar
			t.set_postfix(loss='{:05.5f}'.format(tr_loss_sum/samples_ran))
			t.update(1)
		t.close()

		if scheduler is not None:
			scheduler.step()

		# LOG TRAINING
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
			#to device
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			with torch.set_grad_enabled(False):
				output = model(X)
				loss   = loss_fn(output,T)
				_,Y    = torch.max(output,1)

			# METRICS
			va_loss_sum += loss.item() * X.size(0)
			samples_ran += X.size(0)
			M_va.update(Y.cpu().numpy(),T.cpu().numpy())
			va_batch_loss.append(loss.item())

			t.set_postfix(loss='{:05.5f}'.format(va_loss_sum/samples_ran))
			t.update(1)
		t.close()

		# LOG VALIDATION
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

		print(f'Best validation IoU: {best_iou:.4f}')

		#LOG BATCH LOSSes
		if epoch == 0:
			batch_fp_mode = 'w+'
		else:
			batch_fp_mode = 'a'
		with open(f'{LOG_DIR}/train_batch_log_{model.model_id:03}.tsv',batch_fp_mode) as batch_fp:
			batch_fp.writelines([f'{_:.5f}\n' for _ in tr_batch_loss])
		with open(f'{LOG_DIR}/valid_batch_log_{model.model_id:03}.tsv',batch_fp_mode) as batch_fp:
			batch_fp.writelines([f'{_:.5f}\n' for _ in va_batch_loss])


if __name__ == "__main__":

	#---------- GPU (IF SET) --------------------
	# assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
	if torch.cuda.is_available():
		assert args.gpu < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."
		CUDA_DEV = torch.device(f"cuda:{args.gpu}")
	else:
		CUDA_DEV = torch.device("cpu")

	#---------- MULTI-GPU (IF SET) -------------- <---------- TODO!


	#---------- LOAD AND PARSE HP DICT ----------
	assert os.path.isfile(args.params), "INCORRECT JSON FILE PATH"
	with open(args.params,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines()]
	assert len(HP_LIST) > 0, "GOT EMPTY JSON FILE."
	assert 0 <= args.row < len(HP_LIST), "OUT OF RANGE ROW ARGUMENT." #0-indexed
	HP = HP_LIST[args.row] # load dictionary

	#---------- INPUT BANDS ---------------------
	assert HP['BANDS'] in ['rgb','vnir'],"INCORRECT BANDS IN JSON HP FILE."
	if HP['BANDS'] == 'rgb':
		input_bands = 3
	if HP['BANDS'] == 'vnir':
		input_bands = 4

	#---------- OUTPUT CHANNELS -----------------

	#---------- MODEL ---------------------------
	model_str = HP['MODEL'][0:4]
	assert model_str in ["attn","unet"], "INCORRECT MODEL STRING."
	if model_str == 'unet':
		net = eval(f"model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']},in_channels={input_bands})")
	if model_str == 'attn':
		pass
	# ---> TO GPU
	net = net.to(CUDA_DEV) #checked above

	#---------- LOSS ----------------------------
	assert HP['LOSS'] in ["ce","ew","cw"], "INCORRECT STRING FOR LOSS IN DICT."
	if HP['LOSS'] == "ce":
		loss_fn = torch.nn.CrossEntropyLoss()
	if HP['LOSS'] == "ew":
		loss_fn = None
	if HP['LOSS'] == "cw": #<<< --- Needs some work...
		loss_fn = None

	#---------- OPTIMIZER -----------------------
	assert HP["OPTIM"] in ["adam","lamb"], "INCORRECT STRING FOR OPTIMIZER IN DICT."
	if HP['OPTIM'] == "adam":
		optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == "sgd":
		optimizer = torch.optim.SGD(net.parameters(),lr=HP['LEARNING_RATE'])

	#---------- LEARNING RATE SCHEDULER ----------
	if HP['SCHEDULER'] == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.3)
	elif HP['SCHEDULER'] == "exp":
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
	else:
		scheduler = None	

	#---------- SET ALL SEEDS --------------------
	assert HP['SEED'] in (0,1), "INCORRECT SEED IN JSON PARAMETER DICT."
	if HP['SEED'] == True:
		utils.set_seed(476)	

	#---------- DATALOADERS ----------------------
	transform = v2.Compose([
		v2.RandomHorizontalFlip(p=0.5),
		v2.RandomVerticalFlip(p=0.5)
	])

	tr_ds = dload.SentinelDataset(f"{DATA_DIR}/training",
		n_bands=input_bands,
		n_labels=2,
		transform=transform)

	va_ds = dload.SentinelDataset(f"{DATA_DIR}/validation",
		n_bands=input_bands,
		n_labels=2,
		transform=None)

	dataloaders = {
		'training': torch.utils.data.DataLoader(
			tr_ds,
			batch_size=HP['BATCH'],
			drop_last=False,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
			prefetch_factor=8),
		'validation': torch.utils.data.DataLoader(
			va_ds,
			batch_size=HP['BATCH'],
			drop_last=False,
			shuffle=False,
			num_workers=4,
			pin_memory=True,
			prefetch_factor=8)
	}

	#---------- RUN -----------------------------
	train_and_validate(net,dataloaders,optimizer,loss_fn,scheduler,HP['EPOCHS'])


