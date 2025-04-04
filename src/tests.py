import os
import numpy as np
import torch
import argparse
import utils
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir',required=True,help='Dataset directory.')
	parser.add_argument('--model',required=True,help='Checkpoint to use.')


def test(model,dataloader):

	model.eval()

	#ITERATE THROUGH DATA
	for X,T in dataloader:
		X = X.to(cuda_device,non_blocking=True)
		T = T.to(cuda_device,non_blocking=True)

		with torch.set_grad_enabled(False):
			output = model(X)
			_,Y    = torch.max(output,1)


if __name__ == '__main__':
	pass
	#PARSE ARGS
	#--------------------	
	args = parse_args()


	#LOAD MODEL
	#--------------------
	indexed = {}
	for row in HP_LIST:
		indexed[row['ID']] = {k:row[k] for k in row if k!='ID'}

	

	#LOAD WEIGHTS
	#--------------------