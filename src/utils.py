# import rasterio
# from PIL import Image, ImageDraw
import os
import json
import numpy as np
import torch
import itertools
import random
import argparse

####################################################################################################
# CLASSES
####################################################################################################

class Logger():
	def __init__(self,path,head):
		'''
		path: str
			The file path to the text file where we log.

		head: [str]
			The column names to be included.
		'''
		self.path = path
		self.arr  = None

		with open(self.path,'w') as fp:
			fp.write('\t'.join([_ for _ in head])+'\n')

	def log(self,stats):
		'''
		stats: [float] or np.ndarray
			List with floats corresponding to one line of the log including 
			accuracy, recall, etc.
		'''
		line = '\t'.join([f'{_:.5f}' for _ in stats])
		with open(self.path,'a') as fp:
			fp.write(line + '\n')
		return line

class ConfusionMatrix():
	"""
	Y and T are taken to be two arrays 2 or 3-dimesnional arrays of shape [B,C,C] or [C,C], where B
	is the batch size and C the number of classes. The orientation of the confusion matrix for
	the 2 class problem is set as:

		           predicted
		             T   F
		          +----+----+
		       T  | TP | FN |
		actual    +----+----+
		       F  | FP | TN |
		          +----+----+

	For the case of 3 classes, a single confusion matrix is set analogously so that it includes all
	three classes in one matrix. As usual, the diagonal corresponds to true positives of a class      
	but FNs, FPs, and TNs vary. For example, the class 0 confusion matrix is 

		             predicted
		            0    1    2
		          +----+----+----+
		       0  | TP | FN | FN |
		          +----+----+----+
		actual 1  | FP | TN | TN |
		          +----+----+----+
		       2  | FP | TN | TN |
		          +----+----+----+
	"""
	def __init__(self, n_classes=2):
		self.n_classes = n_classes
		self.M         = np.zeros((n_classes,n_classes))
		self.TP        = 0
		self.FP        = 0
		self.FN        = 0
		self.TN        = 0
		self.y_batches = None
		self.t_batches = None
		self.epsilon   = 0.0000000001

	def update(self,Y,T):
		#for 2 classes, array of indices and mask are the same.
		if self.n_classes == 2:
			self.M[0,0] += ((T==1) & (Y==1)).sum() #TP
			self.M[0,1] += ((T==1) & (Y==0)).sum() #FN
			self.M[1,0] += ((T==0) & (Y==1)).sum() #FP		
			self.M[1,1] += ((T==0) & (Y==0)).sum() #TN
			self.TP = self.M[0,0]
			self.FN = self.M[0,1] 
			self.FP = self.M[1,0] 
			self.TN = self.M[1,1] 

		if self.n_classes == 3:
			# intersections and such -- coulda been modulo loop
			self.M[0,0] += ((T==0) & (Y==0)).sum()
			self.M[0,1] += ((T==0) & (Y==1)).sum() 
			self.M[0,2] += ((T==0) & (Y==2)).sum() 
			self.M[1,0] += ((T==1) & (Y==0)).sum()
			self.M[1,1] += ((T==1) & (Y==1)).sum()
			self.M[1,2] += ((T==1) & (Y==2)).sum()
			self.M[2,0] += ((T==2) & (Y==0)).sum()
			self.M[2,0] += ((T==2) & (Y==1)).sum()
			self.M[2,2] += ((T==2) & (Y==2)).sum()
			# 1x3 arrays with counts for each class
			self.TP = self.M.diagonal()
			self.FP = self.M.sum(axis=0) - self.TP
			self.FN = self.M.sum(axis=1) - self.TP
			self.TN = self.M.sum() - self.TP - self.FP - self.FN


			# <------------------------------------ TODO: TEST THIS
		if self.n_classes > 3:
			for k in range(n_classes*n_classes):
				i = k // n_classes
				j = k % n_classes

				self.M[i,j] += ((T==i) & (Y==j)).sum() #this is reversed from above!

				self.TP = self.M.diagonal()
				self.FP = self.M.sum(axis=0) - self.TP
				self.FN = self.M.sum(axis=1) - self.TP
				self.TN = self.M.sum() - self.TP - self.FP - self.FN

	def ppv(self):
		# Precision -- predictive positive rate
		return self.TP/(self.TP + self.FP + self.epsilon)

	def tpr(self):
		# Recall, sensitivity -- true positive rate
		return self.TP/(self.TP + self.FN + self.epsilon)

	def acc(self):
		# Accuracy -- hit+correct rejections
		return (self.TP+self.TN)/(self.TP+self.FN+self.FP+self.TN+self.epsilon)

	def iou(self,reverse=False):
		'''
		Intersection over union, jaccard index,
		critical success index, whatever you wanna call it...
		'''
		if self.n_classes==2 and reverse:
			# land iou--maybe useful for 2-class
			return self.TN/(self.TN+self.FN+self.FP+self.epsilon)
		return self.TP / (self.TP+self.FN+self.FP+self.epsilon)

	def clear(self):
		self.M = np.zeros((2,2))
		self.TP = self.FP = self.FN = self.TN = 0

	def __call__(self):
		print(self.M)
		print(type(self.M))

####################################################################################################
# FUNCTIONS
####################################################################################################

def save_checkpoint(path,model,optim,scaler,epoch,t_loss,v_loss,best=False):
	'''
	Saves model+optim+scaler state as .pth.tar 
	'''
	# save_path = f'{MODEL_DIR}/state_{epoch:03d}.pt'
	if best == True:
		save_path = f'{path}/best_{model.model_id:03}.pth.tar'
	else:
		save_path = f'{path}/model_{model.model_id:03}_e{epoch:02}.pth.tar'
	checkpoint = {'epoch': epoch,
					't_loss': t_loss,
					'v_loss': v_loss,
					'model_state_dict': model.state_dict(),
					'optim_state_dict': optim.state_dict(),
					'scaler_state_dict': scaler.state_dict()}
	torch.save(checkpoint,save_path)


def save_ddp_checkpoint(path,model,optim,scaler,epoch,t_loss,v_loss,best=False):
	'''
	Saves model+optim+scaler state. References module within DDP wrapper.
	'''
	if best == True:
		save_path = f'{path}/best_{model.module.model_id:03}.pth.tar'
	else:
		save_path = f'{path}/model_{model.module.model_id:03}_e{epoch:02}.pth.tar'
	checkpoint = {'epoch': epoch,
					't_loss': t_loss,
					'v_loss': v_loss,
					'model_state_dict': model.module.state_dict(),
					'optim_state_dict': optim.state_dict(),
					'scaler_state_dict': scaler.state_dict()}
	torch.save(checkpoint,save_path)


def load_checkpoint(path,model,optim):
	checkpoint = torch.load(path,weights_only=False)
	model.load_state_dict(checkpoint['model_state_dict'])
	optim.load_state_dict(checkpoint['optim_state_dict'])
	epoch  = checkpoint['epoch']
	v_loss = checkpoint['v_loss']
	t_loss = checkpoint['t_loss']
	return epoch,t_loss, v_loss


def randomize_hyperparameters(n=1): #-----------------------------------> TODO
	HP = {}
	#same as a grid search but randomize the choice of parameters
	return HP


def sequence_hyperparameters(out_file_path,id_start,trial):
	'''
	Create a list of dict elements each containing a model's hyperparameters.
	The list is stored in out_path in .json format and created using a 
	cross-product (all-by-all) of the parameters provided.
	'''
	# Each parameter -- Trial 1
	if trial == 1:
		seeds = [1]	
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005,0.00075,0.001]	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001]
		loss  = ["ce"]	
		batch = [16,32]
		inits = ["random"]
		bands = [3]
		label = [2]
		model = ["unet2_1","unet2_2","unet2_4",
			"unet3_1",
			"unet5_1","unet5_2","unet5_4",
			"unet6_1"]

	# Each parameter -- Trial 2 No residuals
	if trial == 2:
		seeds = [1]
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005,0.00075,0.001]	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001]
		loss  = ["ce"]	
		batch = [16,32]
		inits = ["random"]
		bands = [3]
		label = [2]
		model = ["unet1_3","unet4_3","unet1_2"] #try no residuals for best in trial 1

	# Each parameter -- Trial 3 RGB+NIR
	if trial == 3:
		seeds = [1]	
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005] #?	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001] #?
		loss  = ["ce"]	
		batch = [16,32] #?
		inits = ["random"]
		bands = [4]
		label = [2]
		model = ["unet2_1","unet2_2","unet2_4"] #best 5? from trial 1

	# Each parameter -- Trial 4 Best unseeded 100 epochs
	if trial == 4:
		seeds = [1]	
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005] #?	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001] #?
		loss  = ["ce"]	
		batch = [16,32] #?
		inits = ["random"]
		bands = [4]
		label = [2]
		model = ["unet2_1","unet2_2","unet2_4"] #best 5? from trial 1

	# Cross-product
	hp = list(itertools.product(seeds,epoch,lrate,sched,optim,decay,loss,batch,inits,bands,label,model))
	HP_NEW = []

	for i in range(len(hp)):
		row_dict = {}
		row_dict["ID"]            = i+id_start
		row_dict["SEED"]          = hp[i][0]
		row_dict["EPOCHS"]        = hp[i][1]		
		row_dict["LEARNING_RATE"] = hp[i][2]
		row_dict["SCHEDULER"]     = hp[i][3]
		row_dict["OPTIM"]         = hp[i][4]
		row_dict["DECAY"]         = hp[i][5]
		row_dict["LOSS"]          = hp[i][6]
		row_dict["BATCH"]         = hp[i][7]
		row_dict["INIT"]          = hp[i][8]
		row_dict["BANDS"]         = hp[i][9]
		row_dict["OUTPUTS"]       = hp[i][10]
		row_dict["MODEL"]         = hp[i][11]
		HP_NEW.append(row_dict)

	with open(out_file_path,'w') as fp:
		for line in HP_NEW:
			# fp.write(str(line))
			# fp.write('\n')
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


def set_seed(seed,cuda=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
	    torch.cuda.manual_seed(seed)  # If using CUDA
	    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
	    torch.backends.cudnn.deterministic = True
	    torch.backends.cudnn.benchmark = False #Am I losing speed here?
    os.environ['PYTHONHASHSEED'] = str(seed)

####################################################################################################
if __name__ == "__main__":

	# ARGV
	parser = argparse.ArgumentParser()
	parser.add_argument("--hpo",required=False,default=None,
		help="Create a JSON file with a combination of hyperparameters.")
	args = parser.parse_args()

	# python3 utils.py --hpo ../hpo/trial1.json
	if args.hpo is not None:
		out_file_path = args.hpo
		assert not os.path.isfile(out_file_path), f"Overwriting existing file {out_file_path}"
		# sequence_hyperparameters(out_file_path,id_start=101,trial=1)		
		sequence_hyperparameters(out_file_path,id_start=421,trial=2)
