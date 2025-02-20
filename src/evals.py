import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_training_log(log_path,out_path):
	assert os.path.isfile(log_path), f"evals.py: NO FILE FOUND IN PATH \"{log_path}\""
	with open(log_path,'r') as fp:
		header = fp.readline().rstrip('\n').split('\t')
		lines  = [_.rstrip('\n').split('\t') for _ in list(fp)]
		array  = np.array(lines).astype(float)

		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.plot(array[:,0],label='Training')
		ax.plot(array[:,2],label='Validation')
		ax.set_ylabel('Loss')
		ax.set_xlabel('Epoch')
		ax.set_title("Training and validation loss")
		plt.legend()
		plt.savefig(f'../log/{out_path}_loss.png')

		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.plot(array[:,3],label='Accuracy')
		ax.plot(array[:,4],label='Recall')
		ax.plot(array[:,5],label='Precision')
		ax.plot(array[:,6],label='IoU')
		# ax.set_ylim((0.0,1.0))
		ax.set_ylabel('Score')
		ax.set_xlabel('Epoch')
		ax.set_title("Validation metrics")
		plt.legend()
		plt.savefig(f'../log/{out_path}_metrics.png')


if __name__ == '__main__':
	plot_training_log('../../training_logs_temp/train_log_000.tsv','test_plot')
