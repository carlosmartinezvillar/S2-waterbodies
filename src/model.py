import torch
import torch.nn as nn
import nn.functional as F

#ALL
class EmbeddingLayer(nn.Module):
	def __init__(self,i_ch,o_ch):
		super(EmbeddingLayer,self).__init__()
		self.conv = nn.Conv2d(i_ch,o_ch,kernel_size=1)

	def forward(self,x):
		return self.conv(x)

class LastLayer(nn.Module):
	def __init__(self,i_ch,o_ch):
		super(LastLayer,self).__init__()
		self.conv = nn.Conv2d(i_ch,o_ch,kernel_size=1)

	def forward(self,x):
		return self.conv(x)

#UNet1_x
class ConvBlock1(nn.Module):
	def __init__(self,i_ch,o_ch,block_type='A'):
		super(ConvBlock1,self).__init__() #-----> switch params to dict and then iterate to set layers :TODO:
		if block_type == 'A':
			# HALF-PADDING/STRIDE 1 CONVOLUTION
			self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		elif block_type == 'B':
			# STRIDE 2 CONVOLUTION -- strictly for 1_4
			self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=2,padding=1,bias=False)
		else:
			raise ValueError("BLOCK TYPE NOT WELL DEFINED IN CONVOLUTION BLOCK 1.")
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU(inplace=True)
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU(inplace=True)

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
		conv1_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}		
		self.C1 = nn.ConvTranspose2d(i_ch,o_ch,**conv1_params)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU(inplace=True)
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU(inplace=True)

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x

class Bottleneck1(nn.Module):
	def __init__(self,i_ch,o_ch):
		super(Bottleneck1,self).__init__()
		self.block = ConvBlock1(i_ch,o_ch,'A')

	def forward(self,x):
		return self.block(x)

#UNet2_x
class ConvBlock2(nn.Module):
	def __init__(self,i_ch,o_ch,block_type='A'):
		super(ConvBlock2,self).__init__()

		if block_type == 'A':
			# HALF-PADDING/STRIDE 1 CONVOLUTION
			stride_1 = 1
		elif block_type == 'B':
			# STRIDE 2 CONVOLUTION (HALVE HxW) -- strictly for 1_4
			stride_1 = 2
		else:
			raise ValueError("BLOCK TYPE NOT WELL DEFINED IN CONVOLUTION BLOCK 2.")

		self.C1 = nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=stride_1,padding=1,bias=False)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU(inplace=True)
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU(inplace=True)

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		res = x.clone()
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x + res

class Bottleneck2(nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck2,self).__init__()
		self.block = ConvBlock2(i_ch,o_ch,'A')

	def forward(self,x):
		return self.block(x)

class UpBlock2_4(nn.Module):
	'''
	Class needed for UNet2_4.
	'''
	def __init__(self,i_ch,o_ch):
		super(UpBlock2_4,self).__init__()
		conv1_params = {'kernel_size':3,'stride':2,'padding':1,'output_padding':1,'bias':False}
		self.C1 = nn.ConvTranspose2d(i_ch,o_ch,**conv1_params)
		self.B1 = nn.BatchNorm2d(o_ch)
		self.R1 = nn.ReLU(inplace=True)
		self.C2 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.B2 = nn.BatchNorm2d(o_ch)
		self.R2 = nn.ReLU(inplace=True)

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		res = x.clone()
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x + res

#UNet3_x
class Bottleneck3(nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck3,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)

#UNet4_x
class Bottleneck4(nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck4,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)

#UNet5_x
class Bottleneck5(nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck5,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)

# UNet6_x
class Bottleneck6(nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck6,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)


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

        # FIRST LAYER
        self.embedding = EmbeddingLayer(in_channels,16)

        # ENCODER
        self.encoder_1 = ConvBlock1(16,32,'A')
        self.down_op_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.encoder_2 = ConvBlock1(32,64,'A')
        self.down_op_2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.encoder_3 = ConvBlock1(64,128,'A')
        self.down_op_3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)        
        self.encoder_4 = ConvBlock1(128,256,'A')
        self.down_op_4 = nn.MaxPo ol2d(kernel_size=2,stride=2,padding=0)        
        
        # BOTTLENECK
        # self.bottleneck = ConvBlock1(256,512,'A')
        self.bottleneck = Bottleneck1(256,512)
        
        # DECODER
        self.up_op_4   = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,bias=False)
        self.decoder_4 = ConvBlock1(512,256,'A')
        self.up_op_3   = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,bias=False)
        self.decoder_3 = ConvBlock1(256,128,'A')
        self.up_op_2   = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,bias=False)        
        self.decoder_2 = ConvBlock1(128,64,'A')
        self.up_op_1   = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,bias=False)        
        self.decoder_1 = ConvBlock1(64,32,'A')

        # LASTLAYER
        self.out_layer = LastLayer(32,out_channels)

    def forward(self, x):
        # ENCODER
        out_0  = self.embedding(x)
        out_1 = self.encoder_1(out_0)
        out_2 = self.encoder_2(self.down_op_1(out_1))
        out_3 = self.encoder_3(self.down_op_2(out_2))
        out_4 = self.encoder_4(self.down_op_3(out_3))
        
        # BOTTLENECK
        out_5 = self.bottleneck(self.down_op_4(out_4))
        
        # DECODER
        inp_4 = self.up_op_4(out_5)
        out_6 = self.decoder_4(torch.cat([out_4,inp_4],dim=1))
        inp_3 = self.up_op_3(out_6)
        out_7 = self.decoder_3(torch.cat([out_3,inp_3],dim=1))
        inp_2 = self.up_op_2(out_7)
        out_8 = self.decoder_2(torch.cat([out_2,inp_2],dim=1))
        inp_1 = self.up_op_1(out_8)
       	out_9 = self.decoder_1(torch.cat([out_1,inp_1],dim=1)) 

        # FINAL LAYER
        output = self.out_layer(out_9)
        
        return output


class UNet1_2(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet1_2,self).__init__()
		self.model_name = 'unet1_2'
		self.model_id   = model_id

		#EMBEDDING
		self.embedding = EmbeddingLayer(in_channels,16)

		#ENCODER
		# down_op_params = {'kernel_size':2,'stride':2,'padding':0,'bias':False} #alt--mirror upconv
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False} 
		self.encoder_1 = ConvBlock1(16,32,'A')
		self.down_op_1 = nn.Conv2d(32,32,**down_op_params) #only on HxW
		self.encoder_2 = ConvBlock1(32,64,'A')
		self.down_op_2 = nn.Conv2d(64,64,**down_op_params)
		self.encoder_3 = ConvBlock1(64,128,'A')
		self.down_op_3 = nn.Conv2d(128,128,**down_op_params)
		self.encoder_4 = ConvBlock1(128,256,'A')
		self.down_op_4 = nn.Conv2d(256,256,**down_op_params)

		#BOTTLENECK
		self.bottleneck = Bottleneck1(256,512)

		#DECODER
		self.up_op_4   = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,bias=False)
		self.decoder_4 = ConvBlock1(512,256,'A')
		self.up_op_3   = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,bias=False)
		self.decoder_3 = ConvBlock1(256,128,'A')
		self.up_op_2   = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,bias=False)
		self.decoder_2 = ConvBlock1(128,64,'A')
		self.up_op_1   = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,bias=False)
		self.decoder_1 = ConvBlock1(64,32,'A')

		#LAST-LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		out_0 = self.embedding(x)
		out_1 = self.encoder_1(out_0)
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

		#FIRST LAYER
		self.embedding = EmbeddingLayer(in_channels,32)

		#ENCODER
		# features = [32,64,128,256,512]
		# down_op_params = {'kernel_size':2,'stride':2,'padding':0,'bias':False}
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False} 		
		self.encoder_1 = ConvBlock1(32,32)
		self.down_op_1 = nn.Conv2d(32,64,**down_op_params)
		self.encoder_2 = ConvBlock1(64,64)
		self.down_op_2 = nn.Conv2d(64,128,**down_op_params)
		self.encoder_3 = ConvBlock1(128,128)
		self.down_op_3 = nn.Conv2d(128,256,**down_op_params)
		self.encoder_4 = ConvBlock1(256,256)
		self.down_op_4 = nn.Conv2d(256,512,**down_op_params)

		#BOTTLENECK
		self.bottleneck = Bottleneck1(512,512)

		#DECODER
		# features = [512,256,128,64,32]
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.decoder_4 = ConvBlock1(512,512)
		self.up_op_3   = nn.ConvTranspose2d(512,128,**up_op_params) #it gets weird here: C/4
		self.decoder_3 = ConvBlock1(256,256)
		self.up_op_2   = nn.ConvTranspose2d(256,64,**up_op_params)
		self.decoder_2 = ConvBlock1(128,128)
		self.up_op_1   = nn.ConvTranspose2d(128,32,**up_op_params)
		self.decoder_1 = ConvBlock1(64,64)

		#LAST LAYER
		self.out_layer = LastLayer(64,out_channels)

	def forward(self,x):
		#ENCODER
		out_0 = self.embedding(x)
		out_1 = self.encoder_1(out_0)
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


class UNet1_4(nn.Module):
	def __init__(self,model_id,in_channels=3,out_channels=2):
		super(UNet1_4,self).__init__()
		self.model_name = 'unet1_4'
		self.model_id = model_id

		#FIRST LAYER
		self.embedding = EmbeddingLayer(in_channels,16)

		#ENCODER
		self.encoder_1 = ConvBlock1(16,32,'B')
		self.encoder_2 = ConvBlock1(32,64,'B')
		self.encoder_3 = ConvBlock1(64,128,'B')
		self.encoder_4 = ConvBlock1(128,256,'B')

		#BOTTLENECK
		self.bottleneck = Bottleneck1(256,256)

		#DECODER
		self.decoder_4 = UpBlock1_4(512,128)
		self.decoder_3 = UpBlock1_4(256,64)
		self.decoder_2 = UpBlock1_4(128,32)
		self.decoder_1 = UpBlock1_4(64,16)

		#LAST LAYERS
		self.out_layer = LastLayer(16,out_channels)

	def forward(self,x):
		#ENCODER
		out_0 = self.embedding(x)
		out_1 = self.encoder_1(out_0)
		out_2 = self.encoder_2(out_1)
		out_3 = self.encoder_3(out_2)
		out_4 = self.encoder_4(out_3)

		#BOTTLENECK
		out_5 = self.bottleneck(out_4)

		#DECODER
		out_6 = self.decoder_4(torch.cat([out_4,out_5],dim=1))
		out_7 = self.decoder_3(torch.cat([out_3,out_6],dim=1))
		out_8 = self.decoder_2(torch.cat([out_2,out_7],dim=1))
		out_9 = self.decoder_1(torch.cat([out_1,out_8],dim=1))

		#LAST LAYER
		output = self.out_layer(out_9)
		return output

#TODO
class UNet2_1(nn.Module):
	def __init__(self,in_channels=3,out_channels=2):
		super(UNet2_1,self).__init__()
		#IDs
		self.model_name = 'unet2_1'
		self.model_id   = model_id

		# FIRST LAYER
		self.embedding  = EmbeddingLayer(in_channels,16)

		# ENCODER
		# features = [16,32,64,128,256,512]
		self.encoder_1 = ConvBlock2(16,32,'A')
		self.down_op_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.encoder_2 = ConvBlock2(32,64,'A')
		self.down_op_2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.encoder_3 = ConvBlock2(64,128,'A')
		self.down_op_3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
		self.encoder_4 = ConvBlock2(128,256,'A')
		self.down_op_4 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

		# BOTTLENECK
		self.bottleneck = Bottleneck2(256,512)

		# DECODER
		# features = [512,256,128,64,32]
		self.up_op_4   = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,bias=False)
		self.decoder_4 = ConvBlock2(512,256,'A')
		self.up_op_3   = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,bias=False)
		self.decoder_3 = ConvBlock2(256,128,'A')
		self.up_op_2   = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,bias=False)
		self.decoder_2 = ConvBlock2(128,64,'A')
		self.up_op_1   = nn.ConvTranspose2d(64,32,'A')
		self.decoder_1 = ConvBlock2(64,32,'A')

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		out_0 = self.embedding(x)
		out_1 = self.encoder_1(out_0)
		out_2 = self.encoder_2(self.down_op_1(out_1))
		out_3 = self.encoder_3(self.down_op_2(out_2))
		out_4 = self.encoder_4(self.down_op_3(out_3))

		#BOTTLENECK
		out_5 = self.bottleneck(self.down_op_4(out_4))

		#DECODER
		inp_4 = self.up_op_4(out_5)
		out_6 = self.decoder_4(torch.cat([out_4,inp_4],dim=1))
		inp_3 = self.up_op_3(out_6)
		out_7 = self.decoder_3(torch.cat([out_3,inp_3],dim=1))
		inp_2 = self.up_op_2(out_7)
		out_8 = self.decoder_2(torch.cat([out_2,inp_2],dim=1))
		inp_1 = self.up_op_1(out_8)
		out_9 = self.decoder_1(torch.cat([out_1,inp_1],dim=1))

		# LAST LAYER
		output = self.out_layer(out_9)
		return output

#TODO
class UNet2_2(nn.Module):
	def __init__(self,in_channels=3,out_channels=2):
		super(UNet2_2,self).__init__()
		self.model_name = 'unet2_2'
		self.model_id   = model_id

		self.embedding  = EmbeddingLayer(in_channels,16)

		# ENCODER
		#features = [16,32,64,128,256,512]
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False} 		
		self.encoder_1 = ConvBlock2(16,32)
		self.down_op_1 = nn.Conv2d(32,32,**down_op_params)		
		self.encoder_2 = ConvBlock2(32,64)
		self.down_op_2 = nn.Conv2d(64,64,**down_op_params)
		self.encoder_3 = ConvBlock2(64,128)
		self.down_op_3 = nn.Conv2d(128,128,**down_op_params)
		self.encoder_4 = ConvBlock2(128,256)
		self.down_op_4 = nn.Conv2d(256,256,**down_op_params)

		# BOTTLENECK
		self.bottleneck = Bottleneck2(256,512)

		# DECODER
		#features = [512,256,128,64,32]
		up_op_params = {'kernel_size':2,'stride':2,'bias':False}
		self.up_op_4   = nn.ConvTranspose2d(512,256,**up_op_params)
		self.decoder_4 = ConvBlock2(512,256)
		self.up_op_3   = nn.ConvTranspose2d(256,128,**up_op_params)
		self.decoder_3 = ConvBlock2(256,128)
		self.up_op_2   = nn.ConvTranspose2d(128,64,**up_op_params)
		self.decoder_2 = ConvBlock2(128,64)
		self.up_op_1   = nn.ConvTranspose2d(64,32,**up_op_params)
		self.decoder_1 = ConvBlock2(64,32)

		# LAST LAYER
		self.out_layer = LastLayer(32,out_channels)

	def forward(self,x):
		#ENCODER
		enc_0 = self.embedding(x)
		enc_1 = self.encoder_1(enc_0)
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

		#LAST LAYER
		output = self.out_layer(dec_1)
		return output

#TODO
class UNet2_3(nn.Module):
	def __init__(self,in_channels=3,out_channels=2):
		super(UNet2_3,self).__init__()
		self.model_name = 'unet2_3'
		self.model_id   = model_id

		# ENCODER
		features = [32,64,128,256,512]
		down_op_params = {'kernel_size':3,'stride':2,'padding':1,'bias':False}
		self.encoder = nn.ModuleList([
			ConvBlock2(32,32)
		])


		# BOTTLENECK

		# DECODER
		# 

		# LAST LAYER

	def forward(self,x):
		# ENCODER

		# BOTTLENECK

		# DECODER

		# LAST LAYER
		output = self.out_layer(dec_1)

		return output

#TODO
class UNet2_4(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet2_4,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet3_1(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet3_1,self).__init__()

	def forward(self,x):
		return output

#--- 3 LAYERS ---
#TODO
class UNet4_1(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_1,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet4_2(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_2,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet4_3(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_3,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet4_4(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_4,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet5_1(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_1,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet5_2(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_2,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet5_3(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_3,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet5_4(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_4,self).__init__()

	def forward(self,x):
		return output

#TODO
class UNet6_1(nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet3_1,self).__init__()

	def forward(self,x):
		return output


################################################################################
# TRANSFORMERS
################################################################################
class SegFormer_0(nn.Module):
	def __init__(self,in_channels=3,out_channels=1)
	super(SegFormer,self).__init__()
	self.model_name = 'segformer_0'
	self.model._id  = model_id


################################################################################
# LOSS FUNCTIONS
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


if __name__ == '__main__':
	# TEST MODELS
	import dload
	tr_idx,va_idx,_ = dload.sentinel_split_indices()
	dataset = SentinelDataset('../../Desktop/chips/')
	va_ds   = torch.utils.data.Subset(dataset,va_idx)
	va_dl   = torch.utils.data.DataLoader(va_dl,batch_size=8,drop_last=False,shuffle=False)

	test_these = ["Unet1_1","UNet1_2","UNet1_3","UNet1_4","UNet2_1","UNet2_2","UNet2_3",
	"UNet2_4","UNet3_1","UNet4_1","UNet4_2","UNet4_3","UNet4_4"]

	for _ in test_these: #nevermind unit testing...
		print('='*60)
		print("TESTING {_}...")
		print('='*60)		
		exec(f"net = model.{_}('999')")

		net.to(CUDA_DEV)
		X,T = next(va_dl)
		X.to(CUDA_DEV)
		T.to(CUDA_DEV)

		print(X.shape)
		print(T.shape)

		y = net(X)

		print(y.shape)

