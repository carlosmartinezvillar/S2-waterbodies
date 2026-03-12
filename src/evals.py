'''
What does this script need to do??
1. Plot metrics per epoch for a model
2. Plot best mIoU for all models -- need to define range of models as arg to function
3. Find best epoch for a model
4. Plot training loss for a model
5. Plot a grid plot for two given hyperparameters colored by metric value -- MISSING
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from glob import glob
import json

plt.rcParams['font.family'] = 'Courier'
plt.rcParams['font.size'] = 20
FIG_DIR = "../fig"

def plot_training_log(log_path,out_dir):
	'''
	Plot two things. i) Loss for validation and training, ii) and validation metrics.
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
	
	# check - no file
	if not os.path.isfile(log_path):
		return 0

	# open and check if error loading
	try:
		with open(log_path,'r') as fp:
			header = fp.readline().rstrip('\n').split('\t')
			lines  = [_.rstrip('\n').split('\t') for _ in list(fp)]
	except Exception as e:
		print(f"Skipping {log_path}.\nGot an error loading file: {e}")


	# check - Did not run for at least 1 epoch
	if len(lines) < 2:
		return 0

	array  = np.array(lines).astype(float)
	# best_epoch     = array[:,header.index(metric)].argmax()
	best_epoch_val = array[:,header.index(metric)].max()
	return best_epoch_val


def find_best_performer(log_dir,out_dir,hp_file,metric='v_iou'): #<--- fix this to do it for each metric
	'''
	Iterate thru logs and find the best model by IoU.
	'''
	#check metric labels
	assert metric in ['v_iou','v_ppv','v_tpr','v_acc'], "Wrong metric in find_best_performer()"
	if metric == 'v_iou':
		metric_label = 'Validation mIoU'
	if metric == 'v_ppv':
		metric_label = 'Validation PPV (Precision)'
	if metric == 'v_tpr':
		metric_label = 'Validation TPR (Recall)'
	if metric == 'v_acc':
		metric_label = 'Validation Accuracy'

	#epoch log per model
	files = sorted(glob("epoch_log_*.tsv",root_dir=log_dir))

	#adjust path
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	# open each log and get best epoch by 'metric'
	#get ids
	model_max = [find_best_epoch(f"{log_dir}/{f}",metric) for f in files]
	model_ids = [f.split('_')[-1].rstrip('.tsv') for f in files]

	# check hyperparameter file and load
	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != '\n']

	# set log id as key, row as value 
	unet_indexed = {}
	vits_indexed = {}
	for row in HP_LIST:
		if row['MODEL'][0:4] == 'unet':
			unet_indexed[row['ID']] = {k:row[k] for k in row if k!='ID'}
		if row['MODEL'][0:3] == 'vit':
			vits_indexed[row['ID']] = {k:row[k] for k in row if k!='ID'}


	# SUBSET TO UNIQUE MODEL NAMES -- keeps highest
	best  = {}
	for i,m in enumerate(model_ids):
		name = unet_indexed[int(m)]['MODEL']
		if name in best:
			if model_max[i] == 0:
				continue
			#check if new entry is better
			if model_max[i] > best[name]['max']:
				best[name]['max']   = model_max[i]
				best[name]['id']    = m
				best[name]['max_index'] = i
			best[name]['values'].append(model_max[i])
		else:
			best[name] = {'id':m,'max':model_max[i],'max_index':i,'values':[model_max[i]]}


	# Sort
	sorted_indices = np.argsort(model_max)
	sorted_ids     = np.array(model_ids)[sorted_indices]
	sorted_max     = np.array(model_max)[sorted_indices]
	sorted_names   = [unet_indexed[int(i)]['MODEL'] for i in sorted_ids] # match names in HParam to log id

	#color coded by model name
	unique_names = sorted(list(set(sorted_names)))
	cmap         = plt.colormaps.get_cmap('tab20c')
	color_map    = {model_name: cmap(i) for i,model_name in enumerate(unique_names)}
	bar_colors   = [color_map[model_name] for model_name in sorted_names]

	# BAR PLOT ALL COLOR CODED
	plt.figure(figsize=(30,15))
	plt.bar(range(len(sorted_ids)),sorted_max,color=bar_colors,linewidth=0.75)
	plt.title('Performance by Model')
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.xticks(ticks=range(len(sorted_ids)),labels=sorted_names,rotation=90)
	plt.savefig(f"{out_dir}/unet_iou_all.png")
	plt.close()


	# BAR PLOT UNIQUE MODEL NAME
	best_max   = [best[name]['max'] for name in best]
	best_names = [name for name in best]
	best_sorted_idx = np.argsort(best_max)
	best_sorted_max = np.array(best_max)[best_sorted_idx]
	best_sorted_names = np.array(best_names)[best_sorted_idx]
	plt.figure(figsize=(30,15))
	plt.bar(best_sorted_names,best_sorted_max,color='C2',linewidth=0.75)	
	plt.title('Performance by Model')
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.xticks(rotation=90)
	plt.savefig(f"{out_dir}/unet_iou_unique.png")
	plt.close()	

	# BOXPLOT
	boxplot_names  = [name for name in best] #dict only guarantees insertion order?
	boxplot_values = [best[name]['values'] for name in best] #grouped
	boxplot_sorted_names  = np.array(boxplot_names)[best_sorted_idx]
	boxplot_sorted_values = [boxplot_values[i] for i in best_sorted_idx]
	plt.figure(figsize=(30,15))
	plt.boxplot(boxplot_sorted_values,tick_labels=boxplot_sorted_names)
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.title('Performance by Model')
	plt.xticks(rotation=90)
	plt.savefig(f"{out_dir}/unet_boxplot.png")
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

	fig = plt.figure(figsize=(14,7))
	fig.suptitle("Batch Loss")
	ax1  = fig.add_subplot(121)
	ax1.plot(train_lines,label='Training',linestyle='-',linewidth=0.3,color='C0')
	ax1.set_title("Training")
	ax2  = fig.add_subplot(122)
	ax2.plot(valid_lines,label='Validation',linestyle='-',linewidth=0.3,color='C1')
	ax2.set_title("Validation")
	ax2.sharey(ax1)
	plt.tight_layout()
	plt.savefig(f'{out_dir}/batch_{model_nr}.png')
	plt.close()


def plot_all_batch_log(log_dir,out_dir):

	files = glob("train_batch_log_*.tsv",root_dir=log_dir)
	if log_dir[-1] == '/':
		log_dir=log_dir.rstrip('/')

	for file in files:
		plot_batch_log(f"{log_dir}/{file}",out_dir)


if __name__ == '__main__':

	params = '../hpo/params.json'

	parser = argparse.ArgumentParser(description="Plot and summarize train logs.")
	parser.add_argument('--logs',default=None,help="Dir to read the logs from")
	parser.add_argument('--epoch',default=False,action='store_true')
	parser.add_argument('--best',default=False,action='store_true')
	args = parser.parse_args()

	assert os.path.isdir(args.logs),f"No path found for {args.logs}"
	assert os.path.isdir(FIG_DIR),f"No output directory found in {FIG_DIR}"

	if args.best is True:
		find_best_performer(args.logs,FIG_DIR,params)


	# plot_all_training_log('../../lake_logs','../../lake_logs')
	# plot_batch_log('../../lake_logs/train_batch_log_005.tsv','../../lake_logs')
