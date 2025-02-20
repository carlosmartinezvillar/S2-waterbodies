import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

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


def find_best_performer(log_dir,metric='iou'):
	'''
	Iterate thru logs and find the best model by IoU.
	'''
	assert metric in ['iou','ppv','tpr']
	pass #do the stuff here
	if metric == 'iou': #pick different column
		pass
	if metric == 'ppv':
		pass
	if metric == 'tpr':
		pass


def plot_all_models():
	'''
	sort dir and iterate through all log files
	'''
	pass

if __name__ == '__main__':
	plot_training_log('../../training_logs_temp/train_log_000.tsv','../fig')
