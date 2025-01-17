import os
import numpy as np
import torch
import argparse

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

	#LOAD MODEL

	#LOAD WEIGHTS