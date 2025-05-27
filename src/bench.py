import torch
import torch.nn as nn
from mmseg.apis import init_segmentor, inference_segmentor
import mmcv
import timm

def load_aerialformer(pretrained=True):
	pass


def load_segnext(pretrained=True):
	config_file = 'configs/segnext/segnext.base.512x512.ade20k.py'
	checkpoint  = 'https://download.openmmlab.com/mmsegmentation/v0/\
		segnext/segnext_base_512x512_160k_ade20k/\
		segnext_base_512x512_160k_ade20k_20230306_111810-26f9e34b.pth'

	model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
	
	return model


def load_unet_convformer(pretrained=True):
	pass


def load_unet_maxvit(pretrained=True):
	pass


def reset_model(model):
	for layer in model.children():
	   if hasattr(layer, 'reset_parameters'):
	       layer.reset_parameters()


def main():
	pass


if __name__ == '__main__':
	main()