import torch
import torch.nn as nn
from mmseg.apis import init_segmentor, inference_segmentor
import mmcv
import timm
import torchvision.models as tvmodels

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


#EX change backbone to ResNet50
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        
        # Load a pre-trained ResNet50 as the backbone
        self.backbone = tvmodels.resnet50(pretrained=True)
        
        # Remove the original classification head of ResNet
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) 
        
        # Add a new head
        self.head = nn.Linear(2048, num_classes) # ResNet50's last layer before FC is 2048 features

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1) # Flatten
        output = self.head(features)
        return output


def main():
	pass


if __name__ == '__main__':
	main()