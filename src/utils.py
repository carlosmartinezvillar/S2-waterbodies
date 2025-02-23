# import gdal
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import rasterio
# from PIL import Image, ImageDraw
import os
import json
import numpy as np
import torch
import itertools
import random

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
		# self.head = '\t'.join([_ for _ in head])
		self.arr  = None

		with open(self.path,'w+') as fp:
			fp.write('\t'.join([_ for _ in head])+'\n')

	def log(self,stats):
		'''
		stats: [float] or np.ndarray
			List with floats corresponding to accuracy, recall, etc.
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
			#another way
			# tp_mask = (Y==1) & (T==1)
			# fp_mask = tp_mask ^ (Y==1)
			# fn_mask = tp_mask ^ (T==1)
			# tn_mask = ~(tp_mask | fp_mask | fn_mask)
			#yet another way...
			# tnm = ~((Y==1) | (T==1))
			# tn_mask = (Y==0) & (T==0)
			# tnm = ~(Y==1) & ~(T==1)
			# self.TP += tp_mask.sum()
			# self.FP += fp_mask.sum()
			# self.FN += fn_mask.sum()
			# self.TN += tn_mask.sum()

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
		# Intersection over union, jaccard index, critical success index, whatever...
		if self.n_classes==2 and reverse:
			# land iou--maybe useful for 2-class
			return self.TN/(self.TN+self.FN+self.FP+self.epsilon)
		return self.TP / (self.TP+self.FN+self.FP+self.epsilon)

	def clear(self):
		self.M = np.zeros((2,2))
		self.TP = self.FP = self.FN = self.TN = 0

	def __call__(self):
		print(self.table)
		print(type(self.table))

####################################################################################################
def IoU(Y,T,n_classes=2):
	'''
	2-class	
	'''
	if n_classes == 2:
		intersection = ((Y==1) & (T==1)).sum()
		union        = ((Y==1) | (T==1)).sum()
		return intersection/union
	if n_classes == 3:
		pass


def save_checkpoint(path,model,optim,epoch,t_loss,v_loss,best=False):
	'''
	Saves model+optim as .pth.tar 
	'''
	# save_path = f'{MODEL_DIR}/state_{epoch:03d}.pt'
	if best == True:
		save_path = f'{path}/best_{model.model_id:03}.pth.tar'
	else:
		save_path = f'{path}/model_{model.model_id:03}_e{epoch:02}.pth.tar'
	checkpoint = {
			'epoch': epoch,
			't_loss': t_loss,
			'v_loss': v_loss,
			'model_state_dict': model.state_dict(),
			'optim_state_dict': optim.state_dict()
		}
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


def sequence_hyperparameters(id_start,out_path):
	'''
	Create a list of dict elements each containing a model's hyperparameters.
	The list is stored in out_path in .json format and created using the 
	'cross-product' (all-by-all) of the parameters provided.
	'''
	HP = []

	# Each parameter
	lrate = [0.1,0.01,0.001,0.0001]	

	sched = ["step","linear","exp"]

	optim = ["adam","lamb"]

	loss  = ["ce"]
	# loss  = ["ce","ew","cw"] #Cross-entropy,edge-weighted,class-weighted

	batch = [16]
	# batch = [16,32,64]

	init  = ["random"]
	# init  = ["resnet","random"] #Resnet weights adjusted or sqrt(2/n_l) (He et al.)

	model = ["unet1_1"]
	# model   = ["unet1_1","unet1_2","unet1_3","unet1_4","unet2_1","unet2_2","unet2_3","unet2_4",
	# 	"unet3_1","unet4_1","unet4_2","unet4_3","unet4_4","unet5_1","unet5_2","unet5_3",
	# 	"unet5_4","unet6_1"]


	# Cross-product
	hp0 = list(itertools.product(lrate,sched,optim,loss,batch,init,model))

	for i in range(len(hp0)):
		row_dict = {}
		row_dict['ID'] = i+id_start
		row_dict['LEARNING_RATE'] = hp0[i][0]
		row_dict['SCHEDULER'] = hp0[i][1]
		row_dict['OPTIM']  = hp0[i][2]
		row_dict['LOSS']   = hp0[i][3]
		row_dict['BATCH']  = hp0[i][4]
		row_dict['INIT']   = hp0[i][5]
		row_dict['MODEL']  = hp0[i][6]
		
		HP.append(d)

	return HP


def set_seed(seed,cuda=True):
    torch.manual_seed(seed)
    if cuda is True:
	    torch.cuda.manual_seed(seed)  # If using CUDA
    # torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #Am I losing speed here?

####################################################################################################
if __name__ == "__main__":

	#TEST CONFUSION MATRIX
	print("TESTING CONFUSION MATRIX")

	#CHECK CONFUSION MATRIX
	#check 2-way classification
	y0 = np.array([
		[0,0,0,0,0],
		[0,0,1,1,1],
		[0,0,1,1,1],
		[0,0,0,0,0],
		[0,0,0,0,0]])

	t0 = np.array([
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,1,1,0,0],
		[0,1,1,0,0],
		[0,0,0,0,0]])
	cm2 = ConfusionMatrix(n_classes=2)

	# for i in range(1000):
	cm2.update(y0,t0)
	cm2

	#another check for 3-way classification
	#from array [B,C,x,y] after argmax axis=1, [B,x,y]
	y0 = np.array([[
		[0,0,0,2,2],
		[0,1,2,2,2],
		[0,0,1,2,0],
		[0,0,0,1,0],
		[0,0,0,0,0]]])
		
	t0 = np.array([[
		[0,1,2,2,2],
		[0,1,2,2,2],
		[0,0,1,2,2],
		[0,0,0,1,1],
		[0,0,0,0,0]]])

	cm3 = ConfusionMatrix(n_classes=3)
	# for i in range(1000):
	cm3.update(y0,t0)
	cm3