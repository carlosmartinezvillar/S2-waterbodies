import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from glob import glob
import json

plt.rcParams['font.family'] = 'Courier'
plt.rcParams['font.size'] = 10

def plot_training_log(log_path,out_dir):
	'''
	Plot two things. Loss for validation and training, and validation set metrics.
	'''
	assert os.path.isfile(log_path), f"evals.py: NO FILE FOUND IN PATH \"{log_path}\""
	#assert out exists
	if out_dir[-1] == '/':
		out_dir = out_dir.rstrip('/')

	model_nr = log_path.split('_')[-1].rstrip('.tsv')

	with open(log_path,'r') as fp:
		header = fp.readline().rstrip('\n').split('\t')
		lines  = [_.rstrip('\n').split('\t') for _ in list(fp)]
		array  = np.array(lines).astype(float)

	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.plot(array[:,0],label='Training',linestyle='--',linewidth=0.7)
	ax.plot(array[:,2],label='Validation',linestyle='-',linewidth=0.7)
	ax.set_ylabel('Loss')
	ax.set_xlabel('Epoch')
	ax.set_title("Training and validation loss")
	plt.legend()
	plt.savefig(f'{out_dir}/loss_{model_nr}.png')
	plt.close()

	fig = plt.figure()
	ax  = fig.add_subplot(111)
	params = {'linewidth':0.7}
	ax.plot(array[:,3],label='Accuracy',linestyle=':',**params)
	ax.plot(array[:,4],label='Recall',linestyle='--',**params)
	ax.plot(array[:,5],label='Precision',linestyle='-.',**params)
	ax.plot(array[:,6],label='IoU',linestyle='-',**params)
	# ax.set_ylim((0.0,1.0))
	ax.set_ylabel('Score')
	ax.set_xlabel('Epoch')
	ax.set_title("Validation metrics")
	plt.legend()
	plt.savefig(f'{out_dir}/metrics_{model_nr}.png')
	plt.close()


def plot_all_training_log(log_dir,out_dir):
	'''
	sort dir and iterate through all log files
	'''
	files = glob("epoch_log_*.tsv",root_dir=log_dir)
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	for file in files:
		plot_training_log(f"{log_dir}/{file}",out_dir)


def find_best_epoch(log_path,metric='v_iou'):
	'''
	Iterate thru epochs in a model's log to find the best performer for
	this particular model (best epoch).

	Possible metrics: 'v_iou','v_ppv','v_tpr','v_acc'
	'''
	with open(log_path,'r') as fp:
		header = fp.readline().rstrip('\n').split('\t')
		lines  = [_.rstrip('\n').split('\t') for _ in list(fp)]
		array  = np.array(lines).astype(float)

	# best_epoch     = array[:,header.index(metric)].argmax()
	best_epoch_val = array[:,header.index(metric)].max()
	return best_epoch_val


def find_best_performer(log_dir,hp_file,metric='v_iou'):
	'''
	Iterate thru logs and find the best model by IoU.
	'''
	assert metric in ['v_iou','v_ppv','v_tpr'], "Incorrect metric in evals.find_best_performer()"

	files = glob("epoch_log_*.tsv",root_dir=log_dir)
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	model_ids = [_.split('_')[-1].rstrip('.tsv') for _ in files]

	model_val = []
	for file in files:
		log_path = f"{log_dir}/{file}"
		model_val.append(find_best_epoch(log_path,metric))

	assert os.path.isfile(f"../hpo/{hp_file}"), f"No {hp_file} found."
	with open(f"../hpo/{hp_file}",'r') as fp:
		HP_LIST = json.load(fp)
	hp_models = [row['MODEL'] for row in HP_LIST]
	hp_ids    = [row['ID'] for row in HP_LIST]

	matched_names = [hp_models[hp_ids.index(int(i))] for i in model_ids]

	idx = np.argmax(model_val)
	print(f"BEST PERFOMER: {matched_names[idx]} | {metric}: {model_val[idx]}")

	sorted_idx = np.argsort(matched_names)
	x = np.array(matched_names)[sorted_idx]
	y = np.array(model_val)[sorted_idx]

	plt.figure(figsize=(10,8))
	plt.plot(x,y,linestyle='-',color='C0')
	plt.title('Performance by Model')
	plt.xlabel('model')
	plt.ylabel(metric)
	plt.xticks(matched_names,rotation=45)
	plt.grid(True)
	plt.savefig(f"{log_dir}/model_metric.png")
	plt.close()

def match_parameter_performance():
	'''
	A function that matches model performance and the parameters that yielded
	that performance. Takes (i) a dict (row) in the JSON list of hyperparams.
	And (ii) a row in the training log (the best performer across epochs).
	'''
	pass

def plot_batch_log(log_path,out_dir):

	model_nr = log_path.split('_')[-1].rstrip('.tsv')

	train_path = log_path
	valid_path = train_path.replace('train','valid')

	with open(train_path,'r') as fp:
		train_lines  = [float(_.rstrip('\n')) for _ in list(fp)]

	with open(valid_path,'r') as fp:
		valid_lines  = [float(_.rstrip('\n')) for _ in list(fp)]

	# N_train = len(train_lines)
	# N_valid = len(valid_lines)
	# batches_per_train = N_train/50
	# batches_per_valid = N_valid/50

	# empty_valid = np.empty(N_train,dtype=float)
	# empty_valid[:] = np.nan

	fig = plt.figure(figsize=(14,7))
	fig.suptitle("Batch Loss")
	ax1  = fig.add_subplot(121)
	ax1.plot(train_lines,label='Training',linestyle='-',linewidth=0.5,color='C0')
	ax1.set_title("Training")
	ax2  = fig.add_subplot(122)
	ax2.plot(valid_lines,label='Validation',linestyle='-',linewidth=0.5,color='C1')
	ax2.set_title("Validation")
	ax2.sharey(ax1)
	plt.tight_layout()
	plt.savefig(f'{out_dir}/batch_{model_nr}.png')
	plt.close()	

if __name__ == '__main__':
	# find_best_performer('../../lake_logs','params.json')
	# plot_all_training_log('../../lake_logs','../../lake_logs')
	plot_batch_log('../../lake_logs/train_batch_log_005.tsv','../../lake_logs')
