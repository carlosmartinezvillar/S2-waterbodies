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


def find_best_performer(log_dir,out_dir,hp_file,metric='v_iou'):
	'''
	Iterate thru logs and find the best model by IoU.
	'''
	#check metric labels
	assert metric in ['v_iou','v_ppv','v_tpr','v_acc'], "Wrong metric in find_best_performer()"

	#epoch log per model
	files = sorted(glob("epoch_log_*.tsv",root_dir=log_dir))

	#adjust path
	if log_dir[-1] == '/':
		log_dir = log_dir.rstrip('/')

	#get ids
	model_ids = [_.split('_')[-1].rstrip('.tsv') for _ in files]

	# open each log and get best epoch by 'metric'
	model_max = [find_best_epoch(f"{log_dir}/{_}",metric) for _ in files]

	assert os.path.isfile(hp_file), f"No {hp_file} found."
	with open(hp_file,'r') as fp:
		HP_LIST = [json.loads(line) for line in fp.readlines()]

	indexed = {}
	for row in HP_LIST:
		indexed[row['ID']] = {k:row[k] for k in row if k!='ID'}
	
	pass
	
	# hp_models = [row['MODEL'] for row in HP_LIST]
	# hp_ids    = [row['ID'] for row in HP_LIST]
	# matched_names = [hp_models[hp_ids.index(int(i))] for i in model_ids]
	# idx = np.argmax(model_val)
	# print(f"BEST PERFOMER: {matched_names[idx]} | {metric}: {model_max[idx]}")
	# sorted_idx = np.argsort(matched_names)
	# x = np.array(matched_names)[sorted_idx]
	# y = np.array(model_max)[sorted_idx]
	
	# names = [indexed[i]['MODEL'] for i in model_ids]
	# x = [f"{model_ids[i]} ({n})" for i,n in enumerate(names)]
	x = model_ids
	y = model_max

	plt.figure(figsize=(30,15))
	plt.plot(x,y,linestyle='-',color='C1',linewidth=0.75)
	plt.title('Performance by Model')
	plt.xlabel('model')
	plt.ylabel(metric)
	plt.xticks(x,rotation=90)
	plt.grid(True)
	plt.savefig(f"{out_dir}/model_metric.png")
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
	parser.add_argument('--out-dir',default=None,help="Dir to save plots")
	parser.add_argument('--batch',default=[],nargs=1)
	parser.add_argument('--epoch',default=False,action='store_true')
	parser.add_argument('--best',default=False,action='store_true')
	args = parser.parse_args()

	assert os.path.isdir(args.logs),f"No path found for {args.logs}"
	assert os.path.isdir(args.out_dir),f"No output directory found in {args.out_dir}"

	if args.best is True:
		find_best_performer(args.logs,args.out_dir,params)

	if len(args.batch) == 1:
		plot_batch_log(args.logs,args.out_dir)

	# plot_all_training_log('../../lake_logs','../../lake_logs')
	# plot_batch_log('../../lake_logs/train_batch_log_005.tsv','../../lake_logs')
