'''
What does this script need to do??
1. Plot metrics per epoch for a model
2. Plot best_unet mIoU for all models -- need to define range of models as arg to function
3. Find best_unet epoch for a model
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
	Iterate thru epochs in a model's log to find the best metric
	this particular model (best_unet epoch).

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
	best_epoch_idx = array[:,header.index(metric)].argmax()
	best_epoch_val = array[:,header.index(metric)].max()
	return best_epoch_val,best_epoch_idx


def find_best_performer(log_dir,hp_file,metric='v_iou'): #<--- fix to all metric
	'''
	Iterate thru logs and find the best_unet model by IoU.
	'''
	# check metric labels
	assert metric in ['v_iou','v_ppv','v_tpr','v_acc'], "Wrong metric string."

	if metric == 'v_iou':
		metric_label = 'Validation mIoU'
	if metric == 'v_ppv':
		metric_label = 'Validation PPV (Precision)'
	if metric == 'v_tpr':
		metric_label = 'Validation TPR (Recall)'
	if metric == 'v_acc':
		metric_label = 'Validation Accuracy'


	# epoch log per model
	files = sorted(glob("epoch_log_*.tsv",root_dir=log_dir))

	# adjust path
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	# open each log and get highest 'metric' for a model across epochs
	model_max = [find_best_epoch(f"{log_dir}/{f}",metric) for f in files] #max metric in log
	model_ids = [f.split('_')[-1].rstrip('.tsv') for f in files]

	# check hyperparameter file and load
	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != '\n']

	# set log id-row as key-value 
	unet_indexed = {}
	vits_indexed = {}
	for row in HP_LIST:
		if row['MODEL'][0:4] == 'unet':
			unet_indexed[row['ID']] = {k:row[k] for k in row if k!='ID'}
		if row['MODEL'][0:3] == 'vit':
			vits_indexed[row['ID']] = {k:row[k] for k in row if k!='ID'}

	# separate array into vits and cnns
	unet_max = []
	unet_ids = []
	vits_max = []
	vits_ids = []
	for i,m in zip(model_ids,model_max):
		if i in unet_indexed:
			unet_ids.append(i)
			unet_max.append(m)
		if i in vits_indexed:
			vits_ids.append(i)
			vits_max.append(m)

	#######
	# CNNs
	#######
	# SUBSET CNNs TO UNIQUE MODEL NAMES -- keeps highest
	best_unet  = {}
	for i,m_id in enumerate(unet_ids):
		name = unet_indexed[int(m_id)]['MODEL']
		if name in best_unet:
			if unet_max[i] == 0:
				continue
			#check if new entry is better
			if model_max[i] > best_unet[name]['max']:
				best_unet[name]['max']       = unet_max[i]
				best_unet[name]['id']        = m_id
				best_unet[name]['max_index'] = i
			best_unet[name]['values'].append(unet_max[i])
		else:
			best_unet[name] = {'id':m,'max':unet_max[i],'max_index':i,'values':[unet_max[i]]}

	# Sort
	sorted_unet_indices = np.argsort(unet_max)
	sorted_unet_ids     = np.array(unet_ids)[sorted_unet_indices]
	sorted_unet_max     = np.array(unet_max)[sorted_unet_indices]
	sorted_unet_names   = [unet_indexed[int(i)]['MODEL'] for i in sorted_unet_ids] # match HParams

	# color coded by model name
	unique_unet_names = sorted(list(set(sorted_unet_names)))
	cmap_unet         = plt.colormaps.get_cmap('tab20c')
	color_map_unet    = {model_name: cmap(i) for i,model_name in enumerate(unique_unet_names)}
	bar_colors_unet   = [color_map[model_name] for model_name in sorted_unet_names]

	# BAR PLOT ALL CNNs COLOR CODED
	plt.figure(figsize=(30,15))
	plt.bar(range(len(sorted_unet_ids)),sorted_unet_max,color=bar_colors_unet,linewidth=0.75)
	plt.title('Performance by Model')
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.xticks(ticks=range(len(sorted_unet_ids)),labels=sorted_unet_names,rotation=90)
	plt.savefig("../fig/unet_iou_all.png")
	plt.close()

	# BAR PLOT CNNs UNIQUE MODEL NAME
	best_unet_max   = [best_unet[name]['max'] for name in best_unet]
	best_unet_names = [name for name in best_unet]
	best_unet_sorted_idx   = np.argsort(best_unet_max)
	best_unet_sorted_max   = np.array(best_unet_max)[best_unet_sorted_idx]
	best_unet_sorted_names = np.array(best_unet_names)[best_unet_sorted_idx]
	plt.figure(figsize=(30,15))
	plt.bar(best_unet_sorted_names,best_unet_sorted_max,color='C2',linewidth=0.75)	
	plt.title('Performance by Model')
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.xticks(rotation=90)
	plt.savefig("../fig/unet_iou_unique.png")
	plt.close()	

	# BOXPLOT CNNs UNIQUE MODEL NAME
	boxplot_names  = [name for name in best_unet] #dict only guarantees insertion order?
	boxplot_values = [best_unet[name]['values'] for name in best_unet] #grouped
	boxplot_sorted_names  = np.array(boxplot_names)[best_unet_sorted_idx]
	boxplot_sorted_values = [boxplot_values[i] for i in best_unet_sorted_idx]
	plt.figure(figsize=(30,15))
	plt.boxplot(boxplot_sorted_values,tick_labels=boxplot_sorted_names)
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.title('Performance by Model')
	plt.xticks(rotation=90)
	plt.savefig("../fig/unet_boxplot.png")
	plt.close()

	######
	# ViTs
	######

def match_parameter_performance(log_dir,hp_file):
	'''
	A function that matches model performance and the parameters that yielded
	that performance. Takes (i) a dict (row) in the JSON list of hyperparams.
	And (ii) a row in the training log (the best performer across epochs).
	'''
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != '\n']

	# DICT OF MODEL NAMES CONTAINING A LIST OF HP ROWS (EACH MODEL RUN)
	models = {}
	for row in HP_LIST:
		if row['MODEL'] in models:
			models[row['MODEL']].append({k:row[k] for k in row if k!='MODEL'})
		else:
			models[row['MODEL']] = [{k:row[k] for k in row if k!='MODEL'}]


	# SINGLE MODEL -- GRAB ONE MODEL AND FLATTEN RUNS
	model_str = 'unet2_1'
	trial_id,trial_lrate,trial_miou,trial_batch = [],[],[],[]
	trial_decay,trial_optim,trial_bands,trial_sched = [],[],[],[]
	for trial in models[model_str]:
		log_file = f"{log_dir}/epoch_log_{trial['ID']:03d}.tsv"
		if not os.path.isfile(log_file):
			continue
		with open(log_file,'r') as fp:
			if len(fp.readlines()) < 2:
				continue
		trial_id.append(trial['ID'])
		best_miou, best_epoch = find_best_epoch(log_file,metric='v_iou')
		trial_miou.append(best_miou)
		trial_lrate.append(trial['LEARNING_RATE'])
		trial_batch.append(trial['BATCH'])
		if trial['OPTIM'] == "adamw":
			trial_decay.append(trial['DECAY'])
		else:
			trial_decay.append(0)
		trial_optim.append(trial['OPTIM'])
		trial_bands.append(trial['BANDS'])


	# PLOT BOX/STRIPLOT -- ALL RUNS BY LEARNING RATE
	unique_lr,inverse = np.unique(trial_lrate,return_inverse=True)
	grouped_miou = []
	for u in unique_lr:
		grouped_miou.append(list(np.array(trial_miou)[trial_lrate == u]))

	fig, ax  = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_miou)):
		x_positions = np.ones(len(grouped_miou[i])) * (i+1)
		ax.scatter(x_positions,grouped_miou[i],color='black',alpha=0.6)
	ax.set_xticks(range(1,len(grouped_miou)+1))
	ax.set_xticklabels([str(s) for s in unique_lr])
	ax.set_ylabel('Validation mIoU')
	ax.set_xlabel('Learning Rate')
	ax.set_title(model_str)
	ax.set_xlim(0.5,len(grouped_miou)+0.5)
	plt.savefig(f"../fig/learning_rate_{model_str}.png")


	# PLOT STRIPLOT -- ALL RUNS BY BATCH

	# GRID PLOT -- LEARNING RATE V BATCH?

	# GRID PLOT -- LEARNING RATE V DECAY



if __name__ == '__main__':

	params = '../hpo/params.json'

	parser = argparse.ArgumentParser(description="Plot and summarize train logs.")
	parser.add_argument('--logs',default=None,help="Dir to read the logs from")
	parser.add_argument('--best',default=False,action='store_true')
	args = parser.parse_args()

	assert os.path.isdir(args.logs),f"No path found for {args.logs}"

	if args.best is True:
		find_best_performer(args.logs,params)

	match_parameter_performance(args.logs,params)
	# plot_all_training_log('../../lake_logs','../../lake_logs')
