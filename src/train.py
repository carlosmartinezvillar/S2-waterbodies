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


def train_and_validate_ddp(model,dataloaders,optimizer,loss_fn,scaler,scheduler=None,n_epochs=50,n_class=2):
	N_tr = len(dataloaders['training'].dataset)
	N_va = len(dataloaders['validation'].dataset)

	log_header   = ["tloss","t_acc","vloss","v_acc","v_tpr","v_ppv","v_iou"]
	log_path     = f'{LOG_DIR}/epoch_log_{model.model_id:03}.tsv'	

	# dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


@total_time_decorator
def train_and_validate(model,dataloaders,optimizer,loss_fn,scaler,scheduler=None,n_epochs=50,n_class=2):

	N_tr = len(dataloaders['training'].dataset)
	N_va = len(dataloaders['validation'].dataset)
	log_header   = ["tloss","t_acc","vloss","v_acc","v_tpr","v_ppv","v_iou"]
	log_path     = f'{LOG_DIR}/epoch_log_{model.model_id:03}.tsv'
	epoch_logger = utils.Logger(log_path,log_header)
	best_iou   = 0.0
	best_epoch = 0

	for epoch in range(n_epochs):
		# M_tr = utils.ConfusionMatrix(n_classes=2)
		# M_va = utils.ConfusionMatrix(n_classes=2)
		confusion_matrix_tr = torch.zeros((n_classes,n_classes))
		confusion_matrix_va = torch.zeros((n_classes,n_classes))
		epoch_start_time = time.time()
		print(f'\nEpoch {epoch}/{n_epochs-1}')
		print('-'*80)

		############################################################
		# TRAINING
		############################################################		
		t = tqdm(total=len(dataloaders['training']),ncols=80,ascii=True)
		loss_sum_tr = 0.0
		samples_ran = 0
		model.train()

		for X,T in dataloaders['training']:
			#TO DEVICE
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			# FORWARD
			with torch.autocast(device_type="cuda", dtype=torch.float16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T)

			# BACKPROP
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			# METRICS
			loss_sum_tr += loss.item() * X.size(0)
			# samples_ran += X.size(0)

			Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
			T = T.detach()
			# M_tr.update(Y,T)
			# t.set_postfix(loss='{:05.5f}'.format(loss_sum_tr/samples_ran))
			t.update(1)

		t.close()

		#SCHEDULER UPDATE
		if scheduler is not None:
			scheduler.step()

		# LOG TRAINING
		loss_tr = loss_sum_tr / N_tr
		print(f'[T] LOSS: {loss_tr:.5f} | ACC: {M_tr.acc():.5f} | IoU: {M_tr.iou():.5f}')

		
		############################################################
		# VALIDATION
		############################################################
		t = tqdm(total=len(dataloaders['validation']),ncols=80,ascii=True)
		loss_sum_va = 0.0
		samples_ran = 0
		model.eval()

		with torch.no_grad():
			for X,T in dataloaders['validation']:
				#to device
				X = X.to(CUDA_DEV,non_blocking=True)
				T = T.to(CUDA_DEV,non_blocking=True)

				# FORWARD
				with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
					output = model(X)
					loss   = loss_fn(output,T)
				_,Y    = torch.max(output,1) #soft-prediction, hard-prediction

				# METRICS
				loss_sum_va += loss.item() * X.size(0) #sync
				# samples_ran += X.size(0)
				M_va.update(Y.cpu().numpy(),T.cpu().numpy()) #sync
				va_batch_loss.append(loss.item()) #sync

				# t.set_postfix(loss='{:05.5f}'.format(loss_sum_va/samples_ran))
				t.update(1)
		
		t.close()

		# LOG VALIDATION
		loss_va = loss_sum_va / N_va
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

		print(f'Best validation IoU: {best_iou:.4f} -- Epoch {best_epoch}')


if __name__ == "__main__":

	#---------- GPU (IF SET) --------------------
	# assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
	if torch.cuda.is_available():
		assert args.gpu < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."# <--- CHANGE THIS
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
	assert HP['BANDS'] in [3,4],"INCORRECT NR. of BANDS IN JSON HYPERPARAMETER FILE."
	input_bands = HP['BANDS']

	#---------- OUTPUT CHANNELS -----------------
	assert HP['CLASS'] in [2,3], "INCORRECT # OF CLASSES SET IN JSON HYPERPARAMETER FILE."
	# n_classes = HP['CLASS'] 

	#---------- MODEL ---------------------------
	model_str = HP['MODEL'][0:4]
	assert model_str in ["attn","unet"], "INCORRECT MODEL STRING."
	if model_str == 'unet':
		net = eval(f"model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']},in_channels={input_bands})")
	if model_str == 'attn':
		pass
	# ---> TO GPU
	net = net.to(CUDA_DEV) #checked above
	net = torch.compile(net)

	#---------- LOSS ----------------------------
	assert HP['LOSS'] in ["ce","ew","cw"], "INCORRECT STRING FOR LOSS IN DICT."
	if HP['LOSS'] == "ce":
		loss_fn = torch.nn.CrossEntropyLoss()
	if HP['LOSS'] == "ew":
		loss_fn = None
	if HP['LOSS'] == "cw": #<<< --- Needs some work...
		loss_fn = None

	#---------- OPTIMIZER -----------------------
	assert HP["OPTIM"] in ["adam","lamb","adamw"], "INCORRECT STRING FOR OPTIMIZER IN DICT."
	if HP['OPTIM'] == "adam":
		optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == "sgd":
		optimizer = torch.optim.SGD(net.parameters(),lr=HP['LEARNING_RATE'])
	if HP['OPTIM'] == 'adamw':
		optimizer = torch.optim.AdamW(net.parameters(),lr=HP['LEARNING_RATE'])

	#---------- LEARNING RATE SCHEDULER ----------
	if HP['SCHEDULER'] == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.3)
	elif HP['SCHEDULER'] == "exp":
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
	else:
		scheduler = None	

	#----------- AUTOMATIC MIXED PRECISION -------
	scaler = torch.amp.GradScaler("cuda",enabled=True)

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


