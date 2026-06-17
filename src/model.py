import torch
import torch.nn as nn
import torch.nn.functional as F

all_models = [
	"UNet1_1","UNet1_2","UNet1_3","UNet1_4","UNet2_1","UNet2_2","UNet2_3",
	"UNet2_4","UNet3_1","UNet4_1","UNet4_2","UNet4_3","UNet4_4","UNet5_1",
	"UNet5_2","UNet5_3","UNet5_4","UNet6_1"]

##############################
# CONVOLUTIONAL BLOCKS
##############################
class EmbeddingLayer(nn.Module):
	def __init__(self,i_ch,o_ch):
		super(EmbeddingLayer,self).__init__()
		self.conv = nn.Conv2d(i_ch,o_ch,kernel_size=1,bias=True)

	def forward(self,x):
		return self.conv(x)

class LastLayer(nn.Module):
	def __init__(self,i_ch,o_ch):
		super(LastLayer,self).__init__()
		self.conv = nn.Conv2d(i_ch,o_ch,kernel_size=1,bias=True)

	def forward(self,x):
		return self.conv(x)

#UNet1_x
class ConvBlock1(nn.Module):
	def __init__(self,i_ch,o_ch,block_type='A'):
		super(ConvBlock1,self).__init__() #-----> switch params to dict and then iterate to set layers :TODO:
		assert block_type in ('A','B'),"BLOCK TYPE NOT WELL DEFINED IN CONV BLOCK 1."
		if block_type == 'B':
			first_stride = 2 # HALVE HxW -- strictly for 1_4
		else:
			first_stride = 1 # HALF-PADDING
		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=first_stride,padding=1,bias=True)			
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x

class UpBlock1_4(nn.Module):
	'''
	Class needed for UNet1_4. The first convolution is a ConvTranspose2d doubling HxW.
	This avoids an additional upscale operation outside the block (mirroring the downblocks in
	this particular model).
	'''
	def __init__(self,i_ch,o_ch):
		super(UpBlock1_4,self).__init__()
		# conv1_params = {'kernel_size':2,'stride':2,'padding':0,'output_padding':0,'bias':False}
		upconv_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}		
		self.C1 = nn.ConvTranspose2d(i_ch,o_ch,**upconv_params)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x

#UNet2_x
class ConvBlock2(nn.Module):
	def __init__(self,i_ch,o_ch,block_type='A'):
		super(ConvBlock2,self).__init__()

		if block_type == 'A':
			first_stride = 1 # HALF-PADDING
		elif block_type == 'B':
			first_stride = 2 # HALVE HxW -- strictly for 1_4
		else:
			raise ValueError("BLOCK TYPE NOT WELL DEFINED IN CONVOLUTION BLOCK 2.")

		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=first_stride,padding=1,bias=True)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		res = x
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x + res

class UpBlock2_4(nn.Module):
	'''
	Class needed for UNet2_4.
	'''
	def __init__(self,i_ch,o_ch):
		super(UpBlock2_4,self).__init__()
		upconv_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}
		self.C1 = nn.ConvTranspose2d(i_ch,o_ch,**upconv_params)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		res = x
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x + res

#UNet3_x
class ConvBlock3(nn.Module):
	def __init__(self,i_ch,o_ch):
		super().__init__()
		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()

	def forward(self,x):
		res = x
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)				
		return x + res

#UNet4_x
class ConvBlock4(nn.Module):
	def __init__(self,i_ch,o_ch,block_type='A'):
		super(ConvBlock4,self).__init__()

		if block_type == 'A':
			first_stride = 1 # HALF-PADDING/STRIDE 1
		elif block_type == 'B':
			first_stride = 2 	# STRIDE 2 (HALVE HxW)
		else:
			raise ValueError("BLOCK TYPE NOT WELL DEFINED IN CONVOLUTION BLOCK 4.")

		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=first_stride,padding=1,bias=True)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()
		self.C3 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B3 = nn.BatchNorm2d(o_ch)
		self.R3 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		x = self.C3(x)
		x = self.B3(x)
		x = self.R3(x)
		return x

class UpBlock4_4(nn.Module):
	def __init__(self,i_ch,o_ch):
		super(UpBlock4_4,self).__init__()
		upconv_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}		
		self.C1 = nn.ConvTranspose2d(i_ch,o_ch,**upconv_params)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()
		self.C3 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B3 = nn.BatchNorm2d(o_ch)
		self.R3 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		x = self.C3(x)
		x = self.B3(x)
		x = self.R3(x)
		return x

#UNet5_x
class ConvBlock5(nn.Module):
	def __init__(self,i_ch,o_ch,block_type='A'):
		super(ConvBlock5,self).__init__()

		if block_type == 'A':
			first_stride = 1 # HALF-PADDING
		elif block_type == 'B':
			first_stride = 2 # HALVE HxW -- strictly for 1_4
		else:
			raise ValueError("BLOCK TYPE NOT WELL DEFINED IN CONV BLOCK 5.")
			
		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=first_stride,padding=1,bias=True)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()
		self.C3 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B3 = nn.BatchNorm2d(o_ch)
		self.R3 = nn.ReLU()

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		res = x
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		x = self.C3(x)
		x = self.B3(x)
		x = self.R3(x)		
		return x + res

class UpBlock5_4(ConvBlock5):
	def __init__(self,i_ch,o_ch):
		super().__init__(i_ch,o_ch)
		upconv_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}
		self.C1 = nn.ConvTranspose2d(i_ch,o_ch,**upconv_params)

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		res = x
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		x = self.C3(x)
		x = self.B3(x)
		x = self.R3(x)		
		return x + res

# UNet6_x
class ConvBlock6(nn.Module):
	def __init__(self,i_ch,o_ch):
		super().__init__()
		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU()
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU()
		self.C3 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=True)
		self.B3 = nn.BatchNorm2d(o_ch)
		self.R3 = nn.ReLU()

	def forward(self,x):
		res = x
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		x = self.C3(x)
		x = self.B3(x)
		x = self.R3(x)					
		return x + res

################################################################################
# CNNs
################################################################################
#--- 2 LAYERS ---
class UNet1_1(nn.Module):
    def __init__(self, model_id, in_channels=3, out_channels=2):
        super(UNet1_1, self).__init__()
        #IDs
        self.model_name = 'unet1_1'
       	self.model_id   = model_id

        # ENCODER
        down_op_params = {'kernel_size':2,'stride':2,'padding':0}
        self.encoder_1 = ConvBlock1(in_channels,32,'A')
        self.down_op_1 = nn.MaxPool2d(**down_op_params)        
        self.encoder_2 = ConvBlock1(32,64,'A')
        self.down_op_2 = nn.MaxPool2d(**down_op_params)
        self.encoder_3 = ConvBlock1(64,128,'A')
        self.down_op_3 = nn.MaxPool2d(**down_op_params)
        self.encoder_4 = ConvBlock1(128,256,'A')
        self.down_op_4 = nn.MaxPool2d(**down_op_params)        
        
        # BOTTLENECK
        self.bottleneck = ConvBlock1(256,512)
        
        # DECODER
        up_op_params = {'kernel_size':2,'stride':2,'bias':False}
        self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
        self.decoder_4 = ConvBlock1(512,256,'A')
        self.up_op_3   = nn.ConvTranspose2d(256,128,**up_op_params)
        self.decoder_3 = ConvBlock1(256,128,'A')
        self.up_op_2   = nn.ConvTranspose2d(128,64,**up_op_params)        
        self.decoder_2 = ConvBlock1(128,64,'A')
        self.up_op_1   = nn.ConvTranspose2d(64,32,**up_op_params)        
        self.decoder_1 = ConvBlock1(64,32,'A')

        # LASTLAYER
        self.out_layer = LastLayer(32,out_channels)

    def forward(self, x):
        # ENCODER
        enc_1 = self.encoder_1(x)
        enc_2 = self.encoder_2(self.down_op_1(enc_1))
        enc_3 = self.encoder_3(self.down_op_2(enc_2))
        enc_4 = self.encoder_4(self.down_op_3(enc_3))
        
        # BOTTLENECK
        enc_5 = self.bottleneck(self.down_op_4(enc_4))
        
        # DECODER
        dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
        dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
        dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
       	dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
        dec_0 = self.out_layer(dec_1)
        return dec_0


class UNet1_2(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet1_2,self).__init__()
		self.model_name = 'unet1_2'
		self.model_id   = model_id

		#ENCODER
		# down_op_params = {'kernel_size':2,'stride':2,'padding':0,'bias':False} #alt--mirror upconv
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False} 
		self.encoder_1 = ConvBlock1(in_channels,32,'A')
		self.down_op_1 = nn.Conv2d(32,32,**down_op_params) #only on HxW
		self.encoder_2 = ConvBlock1(32,64,'A')
		self.down_op_2 = nn.Conv2d(64,64,**down_op_params)
		self.encoder_3 = ConvBlock1(64,128,'A')
		self.down_op_3 = nn.Conv2d(128,128,**down_op_params)
		self.encoder_4 = ConvBlock1(128,256,'A')
		self.down_op_4 = nn.Conv2d(256,256,**down_op_params)

		#BOTTLENECK
		self.bottleneck = ConvBlock1(256,512)

		#DECODER
		up_op_params   = {'kernel_size':2,'stride':2,'bias':False}
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.decoder_4 = ConvBlock1(512,256,'A')
		self.up_op_3   = nn.ConvTranspose2d(256,128,**up_op_params)
		self.decoder_3 = ConvBlock1(256,128,'A')
		self.up_op_2   = nn.ConvTranspose2d(128,64,**up_op_params)
		self.decoder_2 = ConvBlock1(128,64,'A')
		self.up_op_1   = nn.ConvTranspose2d(64,32,**up_op_params)
		self.decoder_1 = ConvBlock1(64,32,'A')

		#LAST-LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		out_1 = self.encoder_1(x)
		out_2 = self.encoder_2(self.down_op_1(out_1))
		out_3 = self.encoder_3(self.down_op_2(out_2))
		out_4 = self.encoder_4(self.down_op_3(out_3))

		#BOTTLENECK
		out_5 = self.bottleneck(self.down_op_4(out_4))

		#DECODER
		out_6 = self.decoder_4(torch.cat([out_4,self.up_op_4(out_5)],dim=1))
		out_7 = self.decoder_3(torch.cat([out_3,self.up_op_3(out_6)],dim=1))
		out_8 = self.decoder_2(torch.cat([out_2,self.up_op_2(out_7)],dim=1))
		out_9 = self.decoder_1(torch.cat([out_1,self.up_op_1(out_8)],dim=1))

		#LAST LAYER
		output = self.out_layer(out_9)
		return output


class UNet1_3(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet1_3,self).__init__()
		self.model_name = 'unet1_3'
		self.model_id = model_id

		#ENCODER
		# features = [32,64,128,256,512]
		# down_op_params = {'kernel_size':2,'stride':2,'padding':0,'bias':False}
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False} 		
		self.encoder_1 = ConvBlock1(in_channels,32)
		self.encoder_2 = ConvBlock1(64,64)
		self.encoder_3 = ConvBlock1(128,128)
		self.encoder_4 = ConvBlock1(256,256)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		#BOTTLENECK
		self.bottleneck = ConvBlock1(512,512)

		#DECODER
		# features = [512,256,128,64,32]
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock1(512,512)
		self.decoder_3 = ConvBlock1(256,256)
		self.decoder_2 = ConvBlock1(128,128)
		self.decoder_1 = ConvBlock1(64,64)		
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3   = nn.ConvTranspose2d(512,128,**up_op_params)
		self.up_op_2   = nn.ConvTranspose2d(256,64,**up_op_params)
		self.up_op_1   = nn.ConvTranspose2d(128,32,**up_op_params)

		#LAST LAYER
		self.out_layer = LastLayer(64,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet1_4(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet1_4,self).__init__()
		self.model_name = 'unet1_4'
		self.model_id = model_id

		#ENCODER
		self.encoder_1 = ConvBlock1(in_channels,32,'B') #'B' first conv reduces HxW
		self.encoder_2 = ConvBlock1(32,64,'B')
		self.encoder_3 = ConvBlock1(64,128,'B')
		self.encoder_4 = ConvBlock1(128,256,'B')

		#BOTTLENECK
		self.bottleneck = ConvBlock1(256,256)

		#DECODER
		self.decoder_4 = UpBlock1_4(512,128)
		self.decoder_3 = UpBlock1_4(256,64)
		self.decoder_2 = UpBlock1_4(128,32)
		self.decoder_1 = UpBlock1_4(64,16)

		#LAST LAYERS
		self.out_layer = LastLayer(16,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(enc_1)
		enc_3 = self.encoder_3(enc_2)
		enc_4 = self.encoder_4(enc_3)

		#BOTTLENECK
		enc_5 = self.bottleneck(enc_4)

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,enc_5],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,dec_4],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,dec_3],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,dec_2],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet2_1(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet2_1,self).__init__()
		#IDs
		self.model_name = 'unet2_1'
		self.model_id   = model_id

		# ENCODER
		down_op_params = {'kernel_size':2,'stride':2,'padding':0}
		self.encoder_1 = ConvBlock2(in_channels,32)
		self.encoder_2 = ConvBlock2(32,64)
		self.encoder_3 = ConvBlock2(64,128)
		self.encoder_4 = ConvBlock2(128,256)
		self.down_op_1 = nn.MaxPool2d(**down_op_params)
		self.down_op_2 = nn.MaxPool2d(**down_op_params)
		self.down_op_3 = nn.MaxPool2d(**down_op_params)		
		self.down_op_4 = nn.MaxPool2d(**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock2(256,512)

		# DECODER
		# features = [512,256,128,64,32]
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock2(512,256)
		self.decoder_3 = ConvBlock2(256,128)
		self.decoder_2 = ConvBlock2(128,64)
		self.decoder_1 = ConvBlock2(64,32)
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)		
		self.up_op_3   = nn.ConvTranspose2d(256,128,**up_op_params)
		self.up_op_2   = nn.ConvTranspose2d(128,64,**up_op_params)
		self.up_op_1   = nn.ConvTranspose2d(64,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet2_2(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet2_2,self).__init__()
		self.model_name = 'unet2_2'
		self.model_id   = model_id

		# ENCODER
		#features = [16,32,64,128,256,512]
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.encoder_1 = ConvBlock2(in_channels,32)
		self.encoder_2 = ConvBlock2(32,64)
		self.encoder_3 = ConvBlock2(64,128)
		self.encoder_4 = ConvBlock2(128,256)
		self.down_op_1 = nn.Conv2d(32,32,**down_op_params)
		self.down_op_2 = nn.Conv2d(64,64,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,128,**down_op_params)
		self.down_op_4 = nn.Conv2d(256,256,**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock2(256,512)

		# DECODER
		#features = [512,256,128,64,32]
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}

		self.decoder_4 = ConvBlock2(512,256)
		self.decoder_3 = ConvBlock2(256,128)
		self.decoder_2 = ConvBlock2(128,64)
		self.decoder_1 = ConvBlock2(64,32)		
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3   = nn.ConvTranspose2d(256,128,**up_op_params)
		self.up_op_2   = nn.ConvTranspose2d(128,64,**up_op_params)
		self.up_op_1   = nn.ConvTranspose2d(64,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet2_3(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet2_3,self).__init__()
		self.model_name = 'unet2_3'
		self.model_id   = model_id

		#ENCODER
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.encoder_1 = ConvBlock2(in_channels,32)
		self.encoder_2 = ConvBlock2(64,64)
		self.encoder_3 = ConvBlock2(128,128)
		self.encoder_4 = ConvBlock2(256,256)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)		
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)		
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		#BOTTLENECK
		self.bottleneck = ConvBlock2(512,512)

		#DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock2(512,512)
		self.decoder_3 = ConvBlock2(256,256)
		self.decoder_2 = ConvBlock2(128,128)		
		self.decoder_1 = ConvBlock2(64,64)		
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3   = nn.ConvTranspose2d(512,128,**up_op_params)
		self.up_op_2   = nn.ConvTranspose2d(256,64,**up_op_params)
		self.up_op_1   = nn.ConvTranspose2d(128,32,**up_op_params)

		#LAST LAYER
		self.out_layer = LastLayer(64,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet2_4(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet2_4,self).__init__()
		self.model_name = 'unet2_4'
		self.model_id   = model_id

		#ENCODER
		self.encoder_1 = ConvBlock2(in_channels,32,'B')
		self.encoder_2 = ConvBlock2(32,64,'B')
		self.encoder_3 = ConvBlock2(64,128,'B')
		self.encoder_4 = ConvBlock2(128,256,'B')

		#BOTTLENECK
		self.bottleneck = ConvBlock2(256,256)

		#DECODER
		self.decoder_4 = UpBlock2_4(512,128)
		self.decoder_3 = UpBlock2_4(256,64)
		self.decoder_2 = UpBlock2_4(128,32)
		self.decoder_1 = UpBlock2_4(64,16)

		#LAST LAYER
		self.out_layer = LastLayer(16,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(enc_1)
		enc_3 = self.encoder_3(enc_2)
		enc_4 = self.encoder_4(enc_3)

		#BOTTLENECK
		enc_5 = self.bottleneck(enc_4)

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,enc_5],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,dec_4],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,dec_3],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,dec_2],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet3_1(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet3_1,self).__init__()
		self.model_name = 'unet3_1'
		self.model_id   = model_id

		# ENCODER -- needs an embedding conv to join block input to output
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.embedding = nn.Conv2d(in_channels,32,kernel_size=3,stride=1,padding=1)
		self.encoder_1 = ConvBlock3(32,32)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)
		self.encoder_2 = ConvBlock3(64,64)
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.encoder_3 = ConvBlock3(128,128)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)
		self.encoder_4 = ConvBlock3(256,256)
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock3(512,512)

		# DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock3(512,512)
		self.up_op_4 = nn.ConvTranspose2d(512,256,**up_op_params)
		self.decoder_3 = ConvBlock3(256,256)
		self.up_op_3 = nn.ConvTranspose2d(512,128,**up_op_params)		
		self.decoder_2 = ConvBlock3(128,128)
		self.up_op_2 = nn.ConvTranspose2d(256,64,**up_op_params)
		self.decoder_1 = ConvBlock3(64,64)
		self.up_op_1 = nn.ConvTranspose2d(128,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(64,out_channels)


	def forward(self,x):
		# ENCODER
		x = self.embedding(x)
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))
		
		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))
		
		# DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0

#--- 3 LAYERS ---
class UNet4_1(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet4_1'
		self.model_id   = model_id

		# ENCODER
		down_op_params = {'kernel_size':2,'stride':2,'padding':0}		
		self.encoder_1 = ConvBlock4(in_channels,32)
		self.encoder_2 = ConvBlock4(32,64)
		self.encoder_3 = ConvBlock4(64,128)
		self.encoder_4 = ConvBlock4(128,256)
		self.down_op_1 = nn.MaxPool2d(**down_op_params)
		self.down_op_2 = nn.MaxPool2d(**down_op_params)
		self.down_op_3 = nn.MaxPool2d(**down_op_params)
		self.down_op_4 = nn.MaxPool2d(**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock4(256,512)

		# DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}		
		self.decoder_4 = ConvBlock4(512,256)
		self.decoder_3 = ConvBlock4(256,128)
		self.decoder_2 = ConvBlock4(128,64)
		self.decoder_1 = ConvBlock4(64,32)
		self.up_op_4 = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3 = nn.ConvTranspose2d(256,128,**up_op_params)
		self.up_op_2 = nn.ConvTranspose2d(128,64,**up_op_params)
		self.up_op_1 = nn.ConvTranspose2d(64,32,**up_op_params)				

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		# ENCODER
		enc_1 = self.encoder_1(x)		
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		# BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		# DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet4_2(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet4_2'
		self.model_id   = model_id

		# ENCODER
		# down_op_params = {'kernel_size':2,'stride':2,'padding':0}
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False} 				
		self.encoder_1 = ConvBlock4(in_channels,32)
		self.encoder_2 = ConvBlock4(32,64)
		self.encoder_3 = ConvBlock4(64,128)
		self.encoder_4 = ConvBlock4(128,256)
		self.down_op_1 = nn.Conv2d(32,32,**down_op_params)
		self.down_op_2 = nn.Conv2d(64,64,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,128,**down_op_params)
		self.down_op_4 = nn.Conv2d(256,256,**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock4(256,512)

		# DECODER
		# upconv_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}		
		self.decoder_4 = ConvBlock4(512,256)
		self.decoder_3 = ConvBlock4(256,128)
		self.decoder_2 = ConvBlock4(128,64)
		self.decoder_1 = ConvBlock4(64,32)
		self.up_op_4 = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3 = nn.ConvTranspose2d(256,128,**up_op_params)
		self.up_op_2 = nn.ConvTranspose2d(128,64,**up_op_params)
		self.up_op_1 = nn.ConvTranspose2d(64,32,**up_op_params)				

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		# ENCODER
		x = (x)
		enc_1 = self.encoder_1(x)		
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		# BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		# DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet4_3(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet4_3,self).__init__()
		self.model_name = 'unet4_3'
		self.model_id   = model_id

		# ENCODER
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.encoder_1 = ConvBlock4(in_channels,32)
		self.encoder_2 = ConvBlock4(64,64)
		self.encoder_3 = ConvBlock4(128,128)
		self.encoder_4 = ConvBlock4(256,256)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock4(512,512)

		# DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}		
		self.decoder_4 = ConvBlock4(512,512)
		self.decoder_3 = ConvBlock4(256,256)
		self.decoder_2 = ConvBlock4(128,128)
		self.decoder_1 = ConvBlock4(64,64)
		self.up_op_4 = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3 = nn.ConvTranspose2d(512,128,**up_op_params)
		self.up_op_2 = nn.ConvTranspose2d(256,64,**up_op_params)
		self.up_op_1 = nn.ConvTranspose2d(128,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(64,out_channels)


	def forward(self,x):
		# ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		# BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		# DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet4_4(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet4_4'
		self.model_id = model_id

		#ENCODER
		self.encoder_1 = ConvBlock4(in_channels,32,'B') #'B' first conv reduces HxW
		self.encoder_2 = ConvBlock4(32,64,'B')
		self.encoder_3 = ConvBlock4(64,128,'B')
		self.encoder_4 = ConvBlock4(128,256,'B')

		#BOTTLENECK
		self.bottleneck = ConvBlock4(256,256)

		#DECODER
		self.decoder_4 = UpBlock4_4(512,128)
		self.decoder_3 = UpBlock4_4(256,64)
		self.decoder_2 = UpBlock4_4(128,32)
		self.decoder_1 = UpBlock4_4(64,16)

		#LAST LAYERS
		self.out_layer = LastLayer(16,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(enc_1)
		enc_3 = self.encoder_3(enc_2)
		enc_4 = self.encoder_4(enc_3)

		#BOTTLENECK
		enc_5 = self.bottleneck(enc_4)

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,enc_5],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,dec_4],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,dec_3],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,dec_2],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet5_1(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet5_1'
		self.model_id   = model_id

		# ENCODER
		down_op_params = {'kernel_size':2,'stride':2,'padding':0}
		self.encoder_1 = ConvBlock5(in_channels,32)
		self.encoder_2 = ConvBlock5(32,64)
		self.encoder_3 = ConvBlock5(64,128)
		self.encoder_4 = ConvBlock5(128,256)
		self.down_op_1 = nn.MaxPool2d(**down_op_params)
		self.down_op_2 = nn.MaxPool2d(**down_op_params)
		self.down_op_3 = nn.MaxPool2d(**down_op_params)
		self.down_op_4 = nn.MaxPool2d(**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock5(256,512)

		# DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock5(512,256)
		self.decoder_3 = ConvBlock5(256,128)
		self.decoder_2 = ConvBlock5(128,64)
		self.decoder_1 = ConvBlock5(64,32)
		self.up_op_4 = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3 = nn.ConvTranspose2d(256,128,**up_op_params)
		self.up_op_2 = nn.ConvTranspose2d(128,64,**up_op_params)
		self.up_op_1 = nn.ConvTranspose2d(64,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet5_2(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet5_2'
		self.model_id   = model_id

		# ENCODER
		#features = [16,32,64,128,256,512]
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.encoder_1 = ConvBlock5(in_channels,32)
		self.encoder_2 = ConvBlock5(32,64)
		self.encoder_3 = ConvBlock5(64,128)
		self.encoder_4 = ConvBlock5(128,256)
		self.down_op_1 = nn.Conv2d(32,32,**down_op_params)
		self.down_op_2 = nn.Conv2d(64,64,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,128,**down_op_params)
		self.down_op_4 = nn.Conv2d(256,256,**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock5(256,512)

		# DECODER
		#features = [512,256,128,64,32]
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}

		self.decoder_4 = ConvBlock5(512,256)
		self.decoder_3 = ConvBlock5(256,128)
		self.decoder_2 = ConvBlock5(128,64)
		self.decoder_1 = ConvBlock5(64,32)		
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3   = nn.ConvTranspose2d(256,128,**up_op_params)
		self.up_op_2   = nn.ConvTranspose2d(128,64,**up_op_params)
		self.up_op_1   = nn.ConvTranspose2d(64,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet5_3(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet5_3'
		self.model_id   = model_id

		#ENCODER
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.encoder_1 = ConvBlock5(in_channels,32)
		self.encoder_2 = ConvBlock5(64,64)
		self.encoder_3 = ConvBlock5(128,128)
		self.encoder_4 = ConvBlock5(256,256)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)		
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)		
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		#BOTTLENECK
		self.bottleneck = ConvBlock5(512,512)

		#DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock5(512,512)
		self.decoder_3 = ConvBlock5(256,256)
		self.decoder_2 = ConvBlock5(128,128)		
		self.decoder_1 = ConvBlock5(64,64)		
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3   = nn.ConvTranspose2d(512,128,**up_op_params)
		self.up_op_2   = nn.ConvTranspose2d(256,64,**up_op_params)
		self.up_op_1   = nn.ConvTranspose2d(128,32,**up_op_params)

		#LAST LAYER
		self.out_layer = LastLayer(64,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))

		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet5_4(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet5_4'
		self.model_id   = model_id

		#ENCODER
		self.encoder_1 = ConvBlock5(in_channels,32,'B')
		self.encoder_2 = ConvBlock5(32,64,'B')
		self.encoder_3 = ConvBlock5(64,128,'B')
		self.encoder_4 = ConvBlock5(128,256,'B')

		#BOTTLENECK
		self.bottleneck = ConvBlock5(256,256)

		#DECODER
		self.decoder_4 = UpBlock5_4(512,128)
		self.decoder_3 = UpBlock5_4(256,64)
		self.decoder_2 = UpBlock5_4(128,32)
		self.decoder_1 = UpBlock5_4(64,16)

		#LAST LAYERS
		self.out_layer = LastLayer(16,out_channels)

	def forward(self,x):
		#ENCODER
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(enc_1)
		enc_3 = self.encoder_3(enc_2)
		enc_4 = self.encoder_4(enc_3)

		#BOTTLENECK
		enc_5 = self.bottleneck(enc_4)

		#DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,enc_5],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,dec_4],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,dec_3],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,dec_2],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


class UNet6_1(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super().__init__()
		self.model_name = 'unet6_1'
		self.model_id = model_id

		# ENCODER -- needs an embedding conv to join block input to output
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.embedding = nn.Conv2d(in_channels,32,kernel_size=3,stride=1,padding=1)
		self.encoder_1 = ConvBlock6(32,32)
		self.encoder_2 = ConvBlock6(64,64)
		self.encoder_3 = ConvBlock6(128,128)
		self.encoder_4 = ConvBlock6(256,256)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		# BOTTLENECK
		self.bottleneck = ConvBlock6(512,512)

		# DECODER
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.decoder_4 = ConvBlock6(512,512)
		self.decoder_3 = ConvBlock6(256,256)
		self.decoder_2 = ConvBlock6(128,128)
		self.decoder_1 = ConvBlock6(64,64)
		self.up_op_4 = nn.ConvTranspose2d(512,256,**up_op_params)
		self.up_op_3 = nn.ConvTranspose2d(512,128,**up_op_params)
		self.up_op_2 = nn.ConvTranspose2d(256,64,**up_op_params)
		self.up_op_1 = nn.ConvTranspose2d(128,32,**up_op_params)

		# LAST LAYER
		self.out_layer = LastLayer(64,out_channels)


	def forward(self,x):
		# ENCODER
		x = self.embedding(x)
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_op_1(enc_1))
		enc_3 = self.encoder_3(self.down_op_2(enc_2))
		enc_4 = self.encoder_4(self.down_op_3(enc_3))
		
		#BOTTLENECK
		enc_5 = self.bottleneck(self.down_op_4(enc_4))
		
		# DECODER
		dec_4 = self.decoder_4(torch.cat([enc_4,self.up_op_4(enc_5)],dim=1))
		dec_3 = self.decoder_3(torch.cat([enc_3,self.up_op_3(dec_4)],dim=1))
		dec_2 = self.decoder_2(torch.cat([enc_2,self.up_op_2(dec_3)],dim=1))
		dec_1 = self.decoder_1(torch.cat([enc_1,self.up_op_1(dec_2)],dim=1))
		dec_0 = self.out_layer(dec_1)
		return dec_0


################################################################################
# ViTs
################################################################################
class MultiHeadSelfAttention(nn.Module):
	'''
	MHSA layer
	D: embedding dimensino
	H: nr heads
	'''
	def __init__(self, D, H):
		super().__init__()
		assert D % H == 0
		self.D = D
		self.H = H
		self.head_dim = D // H
		self.qkv  = nn.Linear(D, D * 3)
		self.proj = nn.Linear(D, D)

	def forward(self, x):
		B, N, D = x.shape
		qkv = self.qkv(x)  # (B, N, 3D)
		qkv = qkv.reshape(B, N, 3, self.H, self.head_dim)
		qkv = qkv.permute(2,0,3,1,4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1))
		attn = attn / (self.head_dim ** 0.5)
		attn = attn.softmax(dim=-1)

		out = attn @ v
		out = out.transpose(1, 2)
		out = out.reshape(B, N, D)

		return self.proj(out)


class MLP(nn.Module):
	'''
	Vanilla MLP layer in transformer block
	'''
	def __init__(self, dim, mlp_ratio=4):
		super().__init__()
		hidden_dim = dim * mlp_ratio
		self.layers = nn.Sequential(
		    nn.Linear(dim, hidden_dim),
		    nn.GELU(),
		    nn.Linear(hidden_dim, dim)
		)

	def forward(self, x):
		return self.layers(x)


class EncoderAttentionLayer(nn.Module):
	'''
	A complete ViT layer (i.e. MHSA + MLP)
	'''
	def __init__(self,D,H,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(D)
		self.attn  = MultiHeadSelfAttention(D,H)
		self.norm2 = nn.LayerNorm(D)
		self.mlp   = MLP(D, mlp_ratio)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x


class TransformerStage(nn.Module):
	'''
	'Block' grouping multiple MHSA ops. Equivalent to 'convolutional' block in CNNs above. 
	'''
	def __init__(self,E,num_heads,depth,H,W):
		super().__init__()
		self.layers = nn.ModuleList([EncoderAttentionLayer(D,H) for _ in range(depth)])
		self.pos_embedding = nn.Parameter(torch.randn(1,H*W,dim) * 0.02)
		self.downsample = PatchMerging(D)

	def forward(self,x,H,W):
		x = x + self.pos_embedding
		for layer in self.layers:
			x = layer(x)
		skip = x #to decoder
		x, H, W = self.downsample(x,H,W)
		return x,skip,H,W


class PatchEmbedding(nn.Module):
	'''
	Standard ViT patch embedding
	'''
	def __init__(self, img_size=256,patch_size=4,in_channels=3,embed_dim=64):
		super().__init__()

		self.num_patches = (img_size // patch_size) ** 2
		self.projector   = nn.Conv2d(
		    in_channels,
		    embed_dim,
		    kernel_size=patch_size,
		    stride=patch_size
		)

	def forward(self, x): # x: (B, C, H, W)
		x = self.projector(x) # (B, E, H/P, W/P)
		x = x.flatten(2) # (B, E, N)
		x = x.transpose(1, 2) # (B, N, E)
		return x


class PatchMerging(nn.Module):
	'''
	Spatial resolution downsampler. Checkerboard pattern.
	'''
	def __init__(self, dim):
		super().__init__()
		self.norm = nn.LayerNorm(4 * dim)
		self.reduction = nn.Linear(4*dim,2*dim)

	def forward(self, x, H, W):
		# shapes & rearrange
		B, N, E = x.shape
		x = x.view(B, H, W, E)

		# sections
		x00 = x[:, 0::2, 0::2, :] #checkerboard pattern
		x01 = x[:, 1::2, 0::2, :]
		x10 = x[:, 0::2, 1::2, :]
		x11 = x[:, 1::2, 1::2, :]

		# 4C channels
		x = torch.cat([x00, x01, x10, x11],dim=-1)
		H //= 2
		W //= 2
		x = x.view(B, H * W, 4 * E)
		x = self.norm(x)
		x = self.reduction(x) #2C

		# (B,H/2*W/2,2C)
		return x, H, W	


class PatchMergingConv(nn.Module):
	'''
	Spatial resolution downsampling. Strided convolution.
	'''
	def __init__(self,dim):
		super().__init__()
		self.projector = nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=2,stride=2)
		# self.norm      = nn.BatchNorm2d(2*dim)

	def forward(self,x,H,W):
		B,N,E = x.shape
		x = x.view(B,H,W,E).permute(0,3,1,2) #->[B,C,H,W]
		x = self.projector(x)
		# x = self.norm(x)
		_,new_C,new_H,new_W = x.shape
		x = x.flatten(2).transpose(1,2) # [B,N,E]
		return x, new_H, new_W


class AttentionEncoder(nn.Module):
	'''
	N: Sequence length
	D: Embedding size
	H: Head dimensions
	'''

	def __init__(self,i_ch=3):
		super(AttentionEncoder,self).__init__()

		#LAYERS
		self.patch_embed = PatchEmbedding(img_size=256,patch_size=4,in_channels=i_ch,embed_dim=64)
		self.stage1 = TransformerStage(D=64,depth=2,num_heads=4)
		self.merge1 = PatchMerging(64)
		self.stage2 = TransformerStage(D=128,depth=2,num_heads=4)
		self.merge2 = PatchMerging(128)
		self.stage3 = TransformerStage(D=256,depth=2,num_heads=4)
		self.merge3 = PatchMerging(256)
		self.stage4 = TransformerStage(D=512,depth=2,num_heads=4)
		self.merge4 = PatchMerging(512)
		self.stage5 = TransformerStage(D=1024,depth=2,num_heads=4)	

	def forward(self,x):
		x, H, W = self.patch_embed(x) #64x64

		x = self.stage1(x)
		s1 = x
		x, H, W = self.merge1(x, H, W) # 32x32

		x = self.stage2(x)
		s2 = x
		x, H, W = self.merge2(x, H, W) # 16x16

		x = self.stage3(x)
		s3 = x
		x, H, W = self.merge3(x, H, W) # 8x8

		x = self.stage4(x)
		s4 = x
		x, H, W = self.merge4(x, H, W) # 4x4

		x = self.stage5(x)
		s5 = x

		return s1,s2,s3,s4,s5


class AttentionDecoder(nn.Module):
	'''
	Class to process embedding (encoder outputs) into image segmentation mask.

	N: Sequence length
	D: Embedding size
	H: Nr. of attention heads
	input_dim: embedding input dimensions
	output_dim: segmentation mask dimensions
	'''

	def __init__(self,N,D,H,input_dim,output_dim):
		super(AttentionEncoder,self).__init__()

	def forward(self,x):
		return x


################################################################################
# COMBINED UNETS 
################################################################################
class ConvolutionAttention(nn.Module):
	def __init__(self,N,D,H):
		'''
		CNN encoding & ViT decoding
		'''
		super(ConvolutionAttention,self).__init__()
		self.Encoder = None
		self.Decoder = AttentionDecoder(N,D,H,input_dim,output_dim)


	def forward(self,x):
		return x


class AttentionConvolution(nn.Module):
	'''
	ViT encoding & CNN decoding
	'''
	def __init__(self,N,D,H):
		super(AttentionConvolution,self).__init__()

		self.Encoder = AttentionEncoder(N,D,H)

	def forward(self,x):
		return x


class AttentionAttention(nn.Module):
	'''
	ViT encoding and decoding
	'''
	def __init__(self,N,D,H):
		super(AttentionAttention,self).__init__()

	def forward(self,x):
		return x

		
# CNN encoding and decoding: UNet6_1

################################################################################
# LOSSES
################################################################################
class EdgeWeightedLoss(nn.Module):
	def __init__(self,alpha=0.7,beta=0.3):
		super().__init__()
		self.alpha = alpha
		self.beta  = beta
		self.sigma = 5
		self.CE    = nn.CrossEntropyLoss(reduction=None)

	def forward(self,output,target,weight):
		loss = torch.exp(-weight**2)/self.sigma
		return loss

################################################################################
# OTHER/UTILITIES/TESTING
################################################################################
def batch_cpu_profiler(data_iter,model_str):
	'''
	CHECK MEMORY SIZE WITH PYTORCH PROFILER
	'''
	from torch.profiler import profile, record_function, ProfilerActivity
	print('='*30)
	print(f"PROFILER (CPU) FOR {model_str}")
	print('='*30)		
	model = eval(f"{model_str}('999',in_channels=3,out_channels=2)")
	X,T = next(data_iter)

	with profile(activities=[ProfilerActivity.CPU],profile_memory=True,record_shapes=True) as prof:
		# with record_function("other_functions"):
		Y = model(X)

	print(prof.key_averages().table())


def batch_cuda_profiler(data_iter,model_str):
	from torch.profiler import profile, record_function, ProfilerActivity
	print('='*30)
	print(f"PROFILER (CUDA) FOR {model_str}")
	print('='*30)		
	model = eval(f"{model_str}('999',in_channels=3,out_channels=2)")
	X,T = next(data_iter)

	model = model.to(0)
	X     = X.to(0)
	T     = T.to(0)

	with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
		profile_memory=True,record_shapes=True) as prof:
		Y = model(X)

	print(prof.key_averages().table())


def batch_cuda_memory(data_iter,model_str,batch_size):
	'''
	CHECK MEMORY ALLOCATION WITH torch.cuda.mem_get_info().
	'''
	print("="*60)
	print(f"MEMORY USED (CUDA) {model_str}")
	print("="*60)

	#LOAD MODEL, GET DATA
	model = eval(f"{model_str}('999',in_channels=3,out_channels=2)")
	X,T = next(data_iter)
	print(f"BATCH: {X.shape[0]}")

	# good  = '\n'.join([table_arr[0],table_arr[1],table_arr[4],
		# table_arr[5],table_arr[6],table_arr[7],table_arr[-2]])		
	free,total = torch.cuda.mem_get_info(0)
	total_gb = total/1024**3
	used_gb  = (total-free)/1024**3
	print(f"BEFORE:            {used_gb:.2f}/{total_gb:.2f} GB")

	#MODEL TO DEVICE
	model = model.to(0)
	free,total = torch.cuda.mem_get_info(0)
	used_gb  = (total-free)/1024**3
	print(f"MODEL ALONE:       {used_gb:.2f}/{total_gb:.2f} GB")

	#TENSOR TO DEVICE
	X     = X.to(0)
	T     = T.to(0)
	free,total = torch.cuda.mem_get_info(0)
	used_gb  = (total-free)/1024**3
	print(f"MODEL AND TENSORS: {used_gb:.2f}/{total_gb:.2f} GB")

	# FORWARD PASS
	Y     = model(X)
	free,total = torch.cuda.mem_get_info(0)
	used_gb  = (total-free)/1024**3
	print(f"AFTER FWD PASS:    {used_gb:.2f}/{total_gb:.2f} GB")


def check_pass(data_iter,model_str=None):
	'''
	CHECK FORWARD PASS + SHAPE + NUM OF PARAMETERS.
	If model_str is None, check all models in array
	'''
	if model_str is None:		
		test_these = all_models
	else:
		test_these = [model_str]

	for _ in test_these: #nevermind unit testing...
		print('='*60)
		print(f"TESTING FWD PASS ON {_}")
		print('='*60)		
		net = eval(f"{_}('999',in_channels=3,out_channels=2)")

		n_parameters = sum([p.numel() for p in net.parameters()])
		print(f"# OF PARAMETERS: {n_parameters}")

		X,T = next(data_iter)
		print(f"X: {X.shape} | T: {T.shape}")
		y = net(X)
		print(f"Y: {y.shape} -- GOOD.")


def print_model_parameters(model_str):
	'''
	# NUM OF PARAMETERS PER MODULE (IN A SINGLE MODEL)	
	'''
	net = eval(f"{model_str}('999',in_channels=3,out_channels=2)")
	P = [(p[0],p[1].numel(),p[1].shape) for p in net.named_parameters()]
	P_sum = 0
	print(f"{'MODULE'.ljust(20)}\t{'# PARAM'.rjust(10)}\tSHAPE")
	print("-"*80)
	for name,count,shape in P:
		P_sum += count
		print(f"{name:<20}\t{count:>10}\t{shape}")
	print("-"*80)
	print(f"{'TOTAL'.ljust(20)}\t{P_sum:>10}")


if __name__ == '__main__':
	import dload
	import argparse
	import sys
	import os

	parser = argparse.ArgumentParser()
	group  = parser.add_mutually_exclusive_group()
	parser.add_argument('--data-dir',default=None,help="Data directory")
	group.add_argument('-p','--print',default=None,nargs=1,help="Print a model's nr parameters per module")
	group.add_argument('-t','--check',default=None,nargs='?',help="Check batch forward pass")
	group.add_argument('-c','--cpu-profile',default=None,nargs=1,help="CPU PyTorch profiler on a batch")
	group.add_argument('-g','--gpu-profile',default=None,nargs=1,help="GPU PyTorch profiler on a batch")
	group.add_argument('-G','--gpu-memory',default=None,nargs=1,help="GPU Memory check 1 batch")
	parser.add_argument('-b','--batch',default=[32],nargs=1,help="Batch size")
	args = parser.parse_args()

	if args.print:
		print_model_parameters(args.print[0])
	else:
		data_dir = args.data_dir
		if data_dir is None:
			print("No data directory given.")
			sys.exit(1)
		if os.path.isdir(data_dir) == False:
			print("Incorrect data directory given.") #O.w. no error until calling next()!
			sys.exit(1)

		batch = int(args.batch[0])
		va_ds = dload.SentinelDataset(f'{data_dir}/validation')
		va_dl = torch.utils.data.DataLoader(va_ds,batch_size=batch,drop_last=False,shuffle=False)
		data_iter = iter(va_dl)

		if args.cpu_profile:
			batch_cpu_profiler(data_iter,args.cpu_profile[0])
		elif args.gpu_profile:
			batch_cuda_profiler(data_iter,args.gpu_profile[0])
		elif args.gpu_memory:
			batch_cuda_memory(data_iter,args.gpu_memory[0],batch)
		else:
			# TEST MODELS
			if args.check:
				check_pass(data_iter,args.check)
			else:
				check_pass(data_iter)
