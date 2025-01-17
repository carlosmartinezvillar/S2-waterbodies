import os
import numpy as np
import torch
import argparse

def test(model,dataloader):
	pass

	#ITERATE THROUGH DATA
	for X,T in dataloader:
		X = X.to(cuda_device,non_blocking=True)
		T = T.to(cuda_device,non_blocking=True)

		with torch.set_grad_enabled(True):
			output = model(X)
			loss   = loss_fn(output,T)

if __name__ == '__main__':
	pass
	#PARSE ARGS

	#LOAD MODEL

	#LOAD WEIGHTS