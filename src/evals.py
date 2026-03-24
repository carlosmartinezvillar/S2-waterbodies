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
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Courier'
plt.rcParams['font.size'] = 20
FIG_DIR = "../fig"

def plot_training_log(log_path):
	'''
	Plot two things. i) Loss for validation and training, ii) and validation metrics.
	'''
	# OPEN + LOAD FILE
	assert os.path.isfile(log_path), f"plot_training_log(): NO FILE IN PATH \"{log_path}\""
	model_nr = log_path.split('_')[-1].rstrip('.tsv')
	with open(log_path,'r') as fp:
		header = fp.readline().rstrip('\n').split('\t')
		lines  = [_.rstrip('\n').split('\t') for _ in list(fp)]
	assert len(header) > 0, f"Got 0 columns for log file {log_path}."
	array_log  = np.array(lines).astype(float)
	assert array_log.shape[0] > 0, f"Got 0 rows for log file {log_path}."
	assert array_log.shape[1] > 0, f"Got 0 col array for log file {log_path}."

	# TRAINING LOSSES
	tloss_idx = header.index('tloss')
	vloss_idx = header.index('vloss')
 
 	# TRAINING METRICS
	classes = [s for s in header if s[0:4] == 'viou']
	if len(classes) > 2: #plot mIoU
		viou0_idx = header.index('viou0')
		viouN_idx = header.index(classes[-1])
		mIoU = array_log[:,viou0_idx:viouN_idx].mean(axis=1,keepdims=True)
		array_log = np.append(array_log,mIoU,axis=1)
		iou_metric_index = -1
		iou_metric_string = "Valid. mean IoU"
		pass
	else: #plot IoU of water
		# metrics = set(['tacc1','viou1','vacc1'])
		tacc1_idx = header.index('tacc1')
		viou1_idx = header.index('viou1')
		vacc1_idx = header.index('vacc1')
		vtpr1_idx = header.index('vtpr1')
		vppv1_idx = header.index('vppv1')
		if 'tiou1' in header: #log includes training iou
			tiou1_idx = header.index('tiou1')
		iou_metric_index = viou1_idx
		iou_metric_string = "Valid. IoU (water)"



	# PLOT - LOSS
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	params = {'linewidth':0.8}
	ax.plot(array_log[:,tloss_idx],label='Training',linestyle='--',**params)
	ax.plot(array_log[:,vloss_idx],label='Validation',linestyle='-',**params)
	ax.set_ylabel('Loss')
	ax.set_xlabel('Epoch')
	ax.set_title(f"Training and Validation Loss - Model {model_nr}")
	plt.legend()
	plt.savefig(f'../fig/loss_{model_nr}.png')
	plt.close()

	# PLOT - METRIC(S)
	fig = plt.figure(figsize=(30,15))
	ax  = fig.add_subplot(111)
	params = {'linewidth':0.8}
	ax.plot(array_log[:,tacc1_idx],label='Train accuracy',linestyle='--',**params)
	if 'tiou1' in header:
		ax.plot(array_log[:,tiou1_idx],label='Train IoU',linestyle='-.',**params)
	ax.plot(array_log[:,vacc1_idx],label='Valid. accuracy',linestyle=':',**params)
	# ax.plot(array[:,4],label='Recall',linestyle='--',**params)
	# ax.plot(array[:,5],label='Precision',linestyle='-.',**params)
	ax.plot(array_log[:,iou_metric_index],label=iou_metric_string,linestyle='-',**params)
	# ax.set_ylim((0.0,1.0))
	ax.set_ylabel('Score')
	ax.set_xlabel('Epoch')
	ax.set_title("Validation metrics")
	plt.legend()
	plt.savefig(f'../fig/metrics_{model_nr}.png')
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


def find_best_epoch(log_path,metric='viou1'):
	'''
	Iterate thru epochs in a model's log to find the best metric
	this particular model (best_unet epoch).

	Possible metrics: 'viou1', etc.
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
	best_epoch_index = array[:,header.index(metric)].argmax()
	best_epoch_value = array[:,header.index(metric)].max()
	return best_epoch_value,best_epoch_index


def find_best_performer(log_dir,hp_file,out_str,id_list,metric='viou1'): #<--- fix to take any of the metrics
	'''
	Iterate thru logs and find the best_unet model by IoU.
	'''
	# check metric labels
	# assert metric in ['viou1'], "Wrong metric string." 
	if metric == 'viou1':
		metric_label = 'Validation IoU (water)'
	if metric == 'vppv1':
		metric_label = 'Validation PPV (Precision)'
	if metric == 'vtpr1':
		metric_label = 'Validation TPR (Recall)'
	if metric == 'vacc1':
		metric_label = 'Validation Accuracy'

	# epoch logs
	files = sorted(glob("epoch_log_*.tsv",root_dir=log_dir))

	# adjust path
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	# open each log and get highest 'metric' for a model across epochs
	model_max, model_ids = [],[]
	for f in files:
		max_metric, _ = find_best_epoch(f"{log_dir}/{f}",metric)
		model_max.append(max_metric)
		model_ids.append(int(f.split('_')[-1].rstrip('.tsv')))

	# check hyperparameter file and load
	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != '\n']

	# set log id-row as key-value 
	unet_indexed = {}
	vits_indexed = {}
	for row in HP_LIST:
		if row['ID'] in id_list:
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

	# Save file with sorted best parameters
	with open(f'../fig/best_sorted_{out_str}.tsv','w') as fp:
		for j,i in enumerate(sorted_unet_ids[::-1]):
			line = unet_indexed[i]
			line.pop('SEED')
			line.pop('SCHEDULER')
			line.pop('EPOCHS')
			line.pop('INIT')
			line.pop('OUTPUTS')
			line.pop('OPTIM')
			line['ID'] = int(i)
			iou = (sorted_unet_max[::-1])[j]
			if j == 0:
				header = '\t'.join([f"{k:7}" for k in line] + ['IoU'])
				fp.write(header)
				fp.write('\n')
			file_line = '\t'.join([f"{line[k]:7}" for k in line] + [f"{str(iou):7}"])
			fp.write(file_line)
			fp.write('\n')	

	# color coded by model name
	unique_unet_names = sorted(list(set(sorted_unet_names)))
	cmap_unet         = plt.colormaps.get_cmap('tab20')
	color_map_unet    = {model_name: cmap_unet(i) for i,model_name in enumerate(unique_unet_names)}
	bar_colors_unet   = [color_map_unet[model_name] for model_name in sorted_unet_names]
	legend_elements = [Patch(facecolor=color_map_unet[model_name],label=model_name) for model_name in color_map_unet]

	# BAR PLOT CNNs -- COLOR CODED
	plt.figure(figsize=(30,15))
	plt.bar(range(len(sorted_unet_ids)),sorted_unet_max,color=bar_colors_unet,linewidth=0.75)
	plt.title('Validation Set Performance by Model (n=320)')
	plt.xlabel('Model')
	plt.ylabel("Validation IoU (water)")
	# plt.xticks(ticks=range(len(sorted_unet_ids)),labels=sorted_unet_names,rotation=90)
	plt.xticks([])
	plt.legend(handles=legend_elements)
	plt.savefig(f"../fig/{out_str}_all.png")
	plt.close()

	# BAR PLOT CNNs -- UNIQUE MODEL NAME
	best_unet_max   = [best_unet[name]['max'] for name in best_unet]
	best_unet_names = [name for name in best_unet]
	best_unet_sorted_idx   = np.argsort(best_unet_max)
	best_unet_sorted_max   = np.array(best_unet_max)[best_unet_sorted_idx]
	best_unet_sorted_names = np.array(best_unet_names)[best_unet_sorted_idx]
	plt.figure(figsize=(30,15))
	plt.bar(best_unet_sorted_names,best_unet_sorted_max,color='C2',linewidth=0.75)	
	plt.title('Validation Set Performance by Model (n=6)')
	plt.xlabel('Model')
	plt.ylabel("Validation IoU (water)")
	plt.xticks(rotation=90)
	plt.savefig(f"../fig/{out_str}_unique.png")
	plt.close()	

	# BOXPLOT CNNs UNIQUE MODEL NAME
	boxplot_names  = [name for name in best_unet] #dict only guarantees insertion order?
	boxplot_values = [best_unet[name]['values'] for name in best_unet] #grouped
	boxplot_sorted_names  = np.array(boxplot_names)[best_unet_sorted_idx]
	boxplot_sorted_values = [boxplot_values[i] for i in best_unet_sorted_idx]
	plt.figure(figsize=(30,15))
	plt.boxplot(boxplot_sorted_values,tick_labels=boxplot_sorted_names)
	plt.xlabel('Model')
	plt.ylabel("Validation IoU (water pixels)")
	plt.title('Model Performance, Sorted by Max Validation IoU (n=320)')
	plt.xticks(rotation=90)
	plt.savefig(f"../fig/{out_str}_boxplot.png")
	plt.close()

	######
	# ViTs
	######


def get_parameter_performance(log_dir,hp_file,out_str,id_list):
	'''
	A function to plot the distribution of IoU (positive class) over the parameters
	'''
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != '\n']

	# DICT OF MODEL NAMES CONTAINING A LIST OF HP ROWS (EACH ID WITH MODEL STRING)
	# models = {}
	# for row in HP_LIST:
	# 	if row['MODEL'] in models:
	# 		models[row['MODEL']].append({k:row[k] for k in row if k!='MODEL'})
	# 	else:
	# 		models[row['MODEL']] = [{k:row[k] for k in row if k!='MODEL'}]

	#DICT OF IDS
	id_dict = {row['ID']:row for row in HP_LIST if row['ID'] in id_list}


	ious, lrates, batches, decays = [],[],[],[]
	for model_id in id_dict:
		log_file = f"{log_dir}/epoch_log_{model_id:03d}.tsv"
		if not os.path.isfile(log_file):
			continue
		best_iou1, best_epoch = find_best_epoch(log_file,metric='viou1')
		ious.append(best_iou1)
		lrates.append(id_dict[model_id]['LEARNING_RATE'])
		batches.append(id_dict[model_id]['BATCH'])
		decays.append(id_dict[model_id]['DECAY'])


	# ALL MODELS -- PLOT LEARNING RATE V IOU WATER AT BATCH 16
	b16_indices = np.array(batches) == 16
	b16_ious = np.array(ious)[b16_indices]
	b16_lrates = np.array(lrates)[b16_indices]
	b16_decays = np.array(decays)[b16_indices]
	unique_lrates = np.unique(b16_lrates)
	grouped_ious = []
	for u in unique_lrates:
		grouped_ious.append(list(np.array(b16_ious)[b16_lrates == u]))

	fig,ax = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_ious)):
		x_pos = np.ones(len(grouped_ious[i])) * (i+1)
		ax.scatter(x_pos,grouped_ious[i],color='black',alpha=0.5)
	ax.boxplot(grouped_ious,showmeans=True, meanline=True,
		meanprops={'color': 'red', 'ls': '--', 'lw': 2},
		medianprops={'color': 'blue', 'ls': '-', 'lw': 2},
		showfliers=False, showbox=False, showcaps=False)
	ax.set_xticks(range(1,len(unique_lrates)+1))
	ax.set_xticklabels([str(s) for s in unique_lrates])
	ax.set_ylabel('Validation IoU (water)')
	ax.set_xlabel('Learning Rate')
	ax.set_title(f"Batch Size 16 - IoU Across Learning Rates (n={len(b16_ious)})")
	ax.set_xlim(0.5,len(grouped_ious)+0.5)
	ax.set_ylim(0.45,0.90)
	plt.savefig(f"../fig/learning_rates_batch16_{out_str}.png")


	# ALL MODELS -- PLOT LEARNING V IOU WATER AT BATCH 32
	b32_indices = np.array(batches) == 32
	b32_ious = np.array(ious)[b32_indices]
	b32_lrates = np.array(lrates)[b32_indices]
	b32_decays = np.array(decays)[b32_indices]
	unique_lrates = np.unique(b32_lrates)
	grouped_ious = []
	for u in unique_lrates:
		grouped_ious.append(list(np.array(b32_ious)[b32_lrates == u]))
	fig,ax = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_ious)):
		x_pos = np.ones(len(grouped_ious[i])) * (i+1)
		ax.scatter(x_pos,grouped_ious[i],color='black',alpha=0.5)
	ax.boxplot(grouped_ious,showmeans=True, meanline=True,
		meanprops={'color': 'red', 'ls': '--', 'lw': 2},
		medianprops={'color': 'blue', 'ls': '-', 'lw': 2},
		showfliers=False, showbox=False, showcaps=False)
	ax.set_xticks(range(1,len(unique_lrates)+1))
	ax.set_xticklabels([str(s) for s in unique_lrates])
	ax.set_ylabel('Validation IoU (water)')
	ax.set_xlabel('Learning Rate')
	ax.set_title(f"Batch Size 32 - IoU Across Learning Rates (n={len(b32_ious)})")
	ax.set_xlim(0.5,len(grouped_ious)+0.5)
	ax.set_ylim(0.45,0.90)
	plt.savefig(f"../fig/learning_rates_batch32_{out_str}.png")

	unique_decays = np.unique(decays)
	grouped_ious  = []
	for u in unique_decays:
		grouped_ious.append(list(np.array(ious)[decays==u]))
	fig, ax = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_ious)):
		x_pos = np.ones(len(grouped_ious[i])) * (i+1)
		ax.scatter(x_pos,grouped_ious[i],color='black',alpha=0.5)
	ax.boxplot(grouped_ious,showmeans=True, meanline=True,
		meanprops={'color': 'red', 'ls': '--', 'lw': 2},
		medianprops={'color': 'blue', 'ls': '-', 'lw': 2},
		showfliers=False, showbox=False, showcaps=False)
	ax.set_xticks(range(1,len(unique_decays)+1))
	ax.set_xticklabels([str(s) for s in unique_decays])
	ax.set_ylabel('Validation IoU (water)')
	ax.set_xlabel('Weight Decay')
	ax.set_title(f"Performance Across Decay (n={len(ious)})")
	ax.set_xlim(0.5,len(grouped_ious)+0.5)
	ax.set_ylim(0.45,0.90)
	plt.savefig(f"../fig/decay_{out_str}.png")	

	#PLOT DECAY VS LR IN GRID

	return


def get_model_parameter_performance(log_dir,hp_file,model_str,id_list):

	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines() if line != '\n']

	# DICT OF MODEL NAMES CONTAINING A LIST OF HP ROWS (EACH ID WITH MODEL STRING)
	models = {}
	for row in HP_LIST:
		if row['ID'] in id_list:
			if row['MODEL'] in models:
				models[row['MODEL']].append({k:row[k] for k in row if k!='MODEL'})
			else:
				models[row['MODEL']] = [{k:row[k] for k in row if k!='MODEL'}]

	# SINGLE MODEL -- GRAB ONE MODEL AND FLATTEN RUNS
	# model_str = 'unet2_1'
	trial_id,trial_lrate,trial_iou1,trial_batch = [],[],[],[]
	trial_decay,trial_optim,trial_bands,trial_sched = [],[],[],[]
	for trial in models[model_str]:
		log_file = f"{log_dir}/epoch_log_{trial['ID']:03d}.tsv"
		if not os.path.isfile(log_file):
			continue
		with open(log_file,'r') as fp:
			if len(fp.readlines()) < 2:
				continue
		trial_id.append(trial['ID'])
		best_iou1, best_epoch = find_best_epoch(log_file,metric='viou1')
		trial_iou1.append(best_iou1)
		trial_lrate.append(trial['LEARNING_RATE'])
		trial_batch.append(trial['BATCH'])
		if trial['OPTIM'] == "adamw":
			trial_decay.append(trial['DECAY'])
		else:
			trial_decay.append(0)
		trial_optim.append(trial['OPTIM'])
		trial_bands.append(trial['BANDS'])


	# SINGLE MODEL -- PLOT BOX/STRIP PLOT -- GROUPED BY LEARNING RATE
	unique_lr,inverse = np.unique(trial_lrate,return_inverse=True)
	grouped_ious = []
	for u in unique_lr:
		grouped_ious.append(list(np.array(trial_iou1)[trial_lrate == u]))
	fig, ax  = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_ious)):
		x_positions = np.ones(len(grouped_ious[i])) * (i+1)
		ax.scatter(x_positions,grouped_ious[i],color='black',alpha=0.6)
	ax.boxplot(grouped_ious,showmeans=True, meanline=True,
		meanprops={'color': 'red', 'ls': '--', 'lw': 2},
		medianprops={'color': 'blue', 'ls': '-', 'lw': 2},
		showfliers=False, showbox=False, showcaps=False)
	ax.set_xticks(range(1,len(grouped_ious)+1))
	ax.set_xticklabels([str(s) for s in unique_lr])
	ax.set_ylabel('Validation IoU (water)')
	ax.set_xlabel('Learning Rate')
	ax.set_title(f"{model_str} (n={len(trial_iou1)})")
	ax.set_xlim(0.5,len(grouped_ious)+0.5)
	plt.savefig(f"../fig/learning_rate_{model_str}.png")

	# SINGLE MODEL -- BOXPLOT DECAY
	unique_decays = np.unique(trial_decay)
	grouped_ious = []
	for u in unique_decays:
		grouped_ious.append(list(np.array(trial_iou1)[trial_decay == u]))
	fig, ax  = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_ious)):
		x_positions = np.ones(len(grouped_ious[i])) * (i+1)
		ax.scatter(x_positions,grouped_ious[i],color='black',alpha=0.6)
	ax.boxplot(grouped_ious,showmeans=True, meanline=True,
		meanprops={'color': 'red', 'ls': '--', 'lw': 2},
		medianprops={'color': 'blue', 'ls': '-', 'lw': 2},
		showfliers=False, showbox=False, showcaps=False)
	ax.set_xticks(range(1,len(grouped_ious)+1))
	ax.set_xticklabels([str(s) for s in unique_decays])
	ax.set_ylabel('Validation IoU (water)')
	ax.set_xlabel('Decay')
	ax.set_title(f"{model_str} (n={len(trial_iou1)})")
	ax.set_xlim(0.5,len(grouped_ious)+0.5)
	plt.savefig(f"../fig/decay_{model_str}.png")	

	# SINGLE MODEL -- BOXPLOT BATCH
	unique_batches = np.unique(trial_batch)
	grouped_ious = []
	for u in unique_batches:
		grouped_ious.append(list(np.array(trial_iou1)[trial_batch == u]))
	fig, ax  = plt.subplots(figsize=(30,15))
	for i in range(len(grouped_ious)):
		x_positions = np.ones(len(grouped_ious[i])) * (i+1)
		ax.scatter(x_positions,grouped_ious[i],color='black',alpha=0.6)
	ax.boxplot(grouped_ious,showmeans=True, meanline=True,
		meanprops={'color': 'red', 'ls': '--', 'lw': 2},
		medianprops={'color': 'blue', 'ls': '-', 'lw': 2},
		showfliers=False, showbox=False, showcaps=False)
	ax.set_xticks(range(1,len(grouped_ious)+1))
	ax.set_xticklabels([str(s) for s in unique_batches])
	ax.set_ylabel('Validation IoU (water)')
	ax.set_xlabel('Batch Size')
	ax.set_title(f"{model_str} (n={len(trial_iou1)})")
	ax.set_xlim(0.5,len(grouped_ious)+0.5)
	plt.savefig(f"../fig/batch_{model_str}.png")


if __name__ == '__main__':

	#ARGS
	# params_file = '../hpo/params.json'
	parser = argparse.ArgumentParser(description="Plot and summarize train logs.")
	parser.add_argument('--logs',default=None,help="Dir to read the logs from.")
	parser.add_argument('--params',default=None,help="Parameter file (JSON).")
	args = parser.parse_args()
	assert os.path.isdir(args.logs),f"No path found for {args.logs}"
	assert args.params is not None, "No parameter file given."
	assert os.path.isfile(args.params),f"No parameter file found in {args.params}"
	params_file = args.params

	# TRIAL 1
	trial_1 = list(range(101,420+1))
	# find_best_performer(args.logs,params_file,'trial1',trial_1)
	# get_parameter_performance(args.logs,params_file,'trial1',trial_1)
	# get_model_parameter_performance(args.logs,params_file,'unet6_1',trial_1)
	# get_model_parameter_performance(args.logs,params_file,'unet3_1',trial_1)
	# get_model_parameter_performance(args.logs,params_file,'unet2_2',trial_1)
	# get_model_parameter_performance(args.logs,params_file,'unet5_1',trial_1)
	# get_model_parameter_performance(args.logs,params_file,'unet2_1',trial_1)

	#Best 2 of each
	# plot_training_log(f"{args.logs}/epoch_log_268.tsv") #unet6_1
	# plot_training_log(f"{args.logs}/epoch_log_316.tsv") #unet6_1
	# plot_training_log(f"{args.logs}/epoch_log_248.tsv") #unet3_1
	# plot_training_log(f"{args.logs}/epoch_log_280.tsv") #unet3_1
	# plot_training_log(f"{args.logs}/epoch_log_134.tsv") #unet2_2
	# plot_training_log(f"{args.logs}/epoch_log_110.tsv") #unet2_2

	# TRIAL 2
	trial_2 = list(range(421,540+1))
	find_best_performer(args.logs,params_file,'trial2',trial_2)