import torch
import torch.nn as nn
from mmseg.apis import init_model, inference_model
import mmcv
import timm


def load_aerialformer(pretrained=True):
	'''
	Needs to be trained from original repo first to get checkpoints.
	'''

	# config_file     = "../../AerialFormer/aerialformer_small_512x512_5_potsdam.py"
	# config_file     = "../../AerialFormer/aerialformer_small_512x512_loveda.py"	
	# config_file     = "../../AerialFormer/aerialformer_small_896x896_isaid.py"
	config_file     = "../../AerialFormer/aerialformer_tiny_512x512_5_potsdam.py"
	# config_file     = "../../AerialFormer/aerialformer_tiny_512x512_loveda.py"	
	# config_file     = "../../AerialFormer/aerialformer_tiny_896x896_isaid.py"	
	checkpoint_file = "../cfg/"

	if pretrained == True:
		model = init_model(config_file, checkpoint_file, device='cpu') 
	else:
		model = init_model(config_file,checkpoint_file=None,device='cpu')

	pass


def load_segnext(pretrained=True):
	config_file = '../../SegNeXt/local_configs/segnext/large/segnext.large.512x512.ade.160k.py'
	checkpoint  = "../cfg/segnext_large_512x512_ade_160k.pth"

	if pretrained == True:
		model = init_model(config_file, checkpoint_file, device='cpu') 
	else:
		model = init_model(config_file,checkpoint_file=None,device='cpu')

	return model


def load_convformer(pretrained=True):
	'''
	Needs a decoder.
	'''

	model = timm.create_model('convformer_m36.sail_in1k',pretrained=pretrained)
	# data_config = timm.data.resolve_model_data_config(model)
	# transforms  = timm.data.create_transform(**data_config, is_training=False)
	# output = model(transforms(img).unsqueeze(0))

	pass


def load_maxvit(pretrained=True):
	'''
	Needs a decoder.
	'''

	model = timm.create_model('maxvit_small_tf_512.in1k', pretrained=pretrained)
	# data_config = timm.data.resolve_model_data_config(model)
	# transforms  = timm.data.create_transform(**data_config, is_training=False)
	# output      = model(transforms(img).unsqueeze(0))

	pass


def train(model,s2_loaders,transforms,optimizer,loss_fn,scaler,scheduler=None,n_epochs=50,n_class=2,dev):
	'''
	Fit benchmark model to S2-DW dataset.
	'''

	for epoch in range(n_epochs):
		confusion_matrix_tr = torch.zeros((n_class,n_class))
		confusion_matrix_va = torch.zeros((n_class,n_class))
		epoch_start_time = time.time()
		print(f'\nEpoch {epoch+1}/{n_epochs}')
		print('-'*80)		


		############################################################
		# TRAINING
		############################################################	
		t = tqdm(total=len(dataloaders['training']),ncols=80,ascii=True)
		loss_sum_tr = 0.0
		samples_ran = 0
		model.train()

		for X,T in s2_loaders['training']:

			X = X.to(dev,non_blocking=True)
			T = T.to(dev,non_blocking=True)		

			# FORWARD
			with torch.autocast(device_type="cuda", dtype=torch.float16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T)

			# BACKPROP
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			#METRICS
			loss_sum_tr += loss.item() * X.size(0)

			Y = output.detach().argmax(axis=1)
			T = T.detach()
			t.update(1)

		t.close()

		if scheduler is not None:
			scheduler.step()

		loss_tr = loss_sum_tr / N_tr
		print(f'[T] LOSS: {loss_tr:.5f} | ACC: {M_tr.acc():.5f} | IoU: {M_tr.iou():.5f}')


		############################################################
		# VALIDATION
		############################################################
		t = tqdm(total=len(dataloaders['validation']),ncols=80,ascii=True)
		loss_sum_va = 0.0
		samples_ran = 0
		model.eval()

		with torch.no_grad():

			for X,T in dataloaders['validation']:
				#to device
				X = X.to(dev,non_blocking=True)
				T = T.to(dev,non_blocking=True)

				# FORWARD
				with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
					output = model(X)
					loss   = loss_fn(output,T)
				_,Y    = torch.max(output,1) #soft-prediction, hard-prediction

				# METRICS
				loss_sum_va += loss.item() * X.size(0) #sync		
				M_va.update(Y.cpu().numpy(),T.cpu().numpy()) #sync
				t.update(1)

			t.close()

		loss_va = loss_sum_va / N_va
		print(f'[V] loss: {loss_va:.5f} | acc: {M_va.acc():.5f} | iou: {M_va.iou():.5f}')	

		############################################################
		# LOG EPOCH
		############################################################
		epoch_time = time.time() - epoch_start_time
		print(f'\nEpoch time: {epoch_time:.2f}s')
		epoch_result = [loss_tr,M_tr.acc(),loss_va,M_va.acc(),M_va.tpr(),M_va.ppv(),M_va.iou()]
		epoch_logger.log(epoch_result)

		# SAVE MODEL
		epoch_iou = M_va.iou()
		if best_iou < epoch_iou:
			best_iou = epoch_iou
			best_epoch = epoch
			utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,loss_tr,loss_va,best=True)		



def test(model,s2_loader,transforms,loss_fn,scaler,n_class=2,dev):
	'''
	Test a benchmark on the S2-DW dataset.
	'''
	t = tqdm(total=len(s2_loader),ncols=80,ascii=True)
	loss_sum_va = 0.0
	samples_ran = 0
	model.eval()

	confustion_matrix_te = torch.zeros((n_class,n_class))

	with torch.no_grad():

		for X,T in s2_loader:
			#TO DEV
			X = X.to(dev,non_blocking=True)
			T = T.to(dev,non_blocking=True)

			# FORWARD
			with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
				output = model(X)
			_,Y    = torch.max(output,1) #soft-prediction, hard-prediction

			# METRICS
			confusion_matrix_te.update(Y,T) #sync <---- FIX THIS
			t.update(1)

	t.close()

	print(f'[T] acc: {confusion_matrix_te.acc():.5f} | iou: {confusion_matrix_te.iou():.5f}')
	test_result = [test_acc,test_tpr,test_ppv,test_iou]
	line = '\t'.join([f'{_:.5f}' for _ in test_result])
	with open(f"{LOG_DIR}/test_{model_name}_s2dw.tsv",'w') as fp:
		fp.write(line + '\n')


def main():
	pass


if __name__ == '__main__':
	main()


#EX change backbone to ResNet50
# class CustomModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomModel, self).__init__()
        
#         # Load a pre-trained ResNet50 as the backbone
#         self.backbone = tvmodels.resnet50(pretrained=True)
        
#         # Remove the original classification head of ResNet
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) 
        
#         # Add a new head
#         self.head = nn.Linear(2048, num_classes) # ResNet50's last layer before FC is 2048 features

#     def forward(self, x):
#         features = self.backbone(x)
#         features = features.view(features.size(0), -1) # Flatten
#         output = self.head(features)
#         return output
