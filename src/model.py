import torch
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
	def __init__(self,i_ch,o_ch):
		super(ResBlock,self).__init__()
		self.conv1 = torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn1   = torch.nn.BatchNorm2d(o_ch)
		self.relu1 = torch.nn.ReLU(inplace=True)
		self.conv2 = torch.nn.Conv2d(o_ch,i_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn2   = torch.nn.BatchNorm2d(i_ch)
		self.relu2 = torch.nn.ReLU(inplace=True)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out = out + x 
		return out

################################################################################

class EmbeddingLayer(torch.nn.Module):
	def __init__(self,i_ch,o_ch):
		super(EmbeddingLayer,self).__init__()
		self.conv = torch.nn.Conv2d(i_ch,o_ch,kernel_size=1)

	def forward(self,x):
		return self.conv(x)

class LastLayer(torch.nn.Module):
	def __init__(self,i_ch,o_ch):
		super(LastLayer,self).__init__()
		self.conv = torch.nn.Conv2d(i_ch,o_ch,kernel_size=1)

	def forward(self,x):
		return self.conv(x)

class ConvBlock1(torch.nn.Module):
	def __init__(self,i_ch,o_ch,block_type):
		super(ConvBlock1,self).__init__()
		if block_type == 'A':
			# HALF-PADDING/STRIDE 1 CONVOLUTION
			self.C1 = torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		elif block_type == 'B':
			# STRIDE 2 CONVOLUTION
			self.C1 = torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=2,padding=1,bias=False)
		else:
			raise ValueError("BLOCK TYPE NOT DEFINED IN CONVOLUTION BLOCK 1.")
		self.B1 = torch.nn.BatchNorm2d(o_ch)
		self.R1 = torch.nn.ReLU(inplace=True)
		self.C2 = torch.nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.B2 = torch.nn.BatchNorm2d(o_ch)
		self.R2 = torch.nn.ReLU(inplace=True)

	def forward(self,x):
		x = self.C1(x)
		x = self.B1(x)
		x = self.R1(x)
		x = self.C2(x)
		x = self.B2(x)
		x = self.R2(x)
		return x

class Bottleneck1(torch.nn.Module): # --------> SAME AS ConvBlock1 ABOVE, LEAVE 1 ONLY:TODO
	def __init__(self,i_ch,o_ch):
		super(Bottleneck1,self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(o_ch),
			torch.nn.ReLU(o_ch),
			torch.nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(o_ch),
			torch.nn.ReLU(o_ch)		
		)

	def forward(self,x):
		return self.layers(x)

class Bottleneck2(torch.nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck2,self).__init__()
		self.conv1 = torch.nn.Conv2d(i_ch,i_ch,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv2 = torch.nn.Conv2d(i_ch,i_ch,kernel_size=3,stride=1,padding=1,bias=False)

	def forward(self,x):
		o1 = self.conv1(x)
		return self.conv2(o1) + o1

class Bottleneck3(torch.nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck3,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)

class Bottleneck4(torch.nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck4,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)

class Bottleneck5(torch.nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck5,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)

class Bottleneck6(torch.nn.Module):
	def __init__(self,i_ch):
		super(Bottleneck6,self).__init__()
		pass

	def forward(self,x):
		return self.layers(x)


################################################################################
# NETWORKS
################################################################################
#--- 2 LAYERS ---
class UNet1_1(torch.nn.Module):
    def __init__(self, model_id, in_channels=3, out_channels=1):
        super(UNet1_1, self).__init__()

        self.model_name = 'unet1_1'
       	self.model_id   = model_id

        # FIRST LAYER
        self.embedding = EmbeddingLayer(in_channels,16)

        # ENCODER
        self.encoder_1 = ConvBlock1(16,32,'A')
        self.down_op_1 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.encoder_2 = ConvBlock1(32,64,'A')
        self.down_op_2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.encoder_3 = ConvBlock1(64,128,'A')
        self.down_op_3 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)        
        self.encoder_4 = ConvBlock1(128,256,'A')
        self.down_op_4 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)        
        
        # BOTTLENECK
        self.bottleneck = ConvBlock1(256,512,'A')
        
        # DECODER
        self.up_op_4   = torch.nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,bias=False)
        self.decoder_4 = ConvBlock1(512,256,'A')
        self.up_op_3   = torch.nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,bias=False)
        self.decoder_3 = ConvBlock1(256,128,'A')
        self.up_op_2   = torch.nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,bias=False)        
        self.decoder_2 = ConvBlock1(128,64,'A')
        self.up_op_1   = torch.nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,bias=False)        
        self.decoder_1 = ConvBlock1(64,32,'A')

        # LAST LAYER
        self.out_layer = LastLayer(32,2)

    def forward(self, x):
        # ENCODER
        x  = self.embedding(x)
        o1 = self.encoder_1(x)
        o2 = self.encoder_2(self.down_op_1(o1))
        o3 = self.encoder_3(self.down_op_2(o2))
        o4 = self.encoder_4(self.down_op_3(o3))
        
        # BOTTLENECK
        o5 = self.bottleneck(self.down_op_4(o4))
        
        # DECODER
        i4 = self.up_op_4(o5)
        o6 = self.decoder_4(torch.cat([o4,i4],dim=1))
        i3 = self.up_op_3(o6)
        o7 = self.decoder_3(torch.cat([o3,i3],dim=1))
        i2 = self.up_op_2(o7)
        o8 = self.decoder_2(torch.cat([o2,i2],dim=1))
        i1 = self.up_op_1(o8)
       	o9 = self.decoder_1(torch.cat([o1,i1],dim=1)) 

        # FINAL LAYER
        output = self.out_layer(o9)
        
        return output


class UNet1_2(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet1_2,self).__init__()

	def forward(self,x):
		return output


class UNet1_3(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet1_3,self).__init__()

	def forward(self,x):
		return output


class UNet1_4(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet1_4,self).__init__()

	def forward(self,x):
		return output


class UNet2_1(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet2_1,self).__init__()

	def forward(self,x):
		return output


class UNet2_2(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet2_2,self).__init__()

	def forward(self,x):
		return output


class UNet2_3(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet2_3,self).__init__()

	def forward(self,x):
		return output


class UNet2_4(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet2_4,self).__init__()

	def forward(self,x):
		return output


class UNet3_1(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet3_1,self).__init__()

	def forward(self,x):
		return output

#--- 3 LAYERS ---
class UNet4_1(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_1,self).__init__()

	def forward(self,x):
		return output


class UNet4_2(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_2,self).__init__()

	def forward(self,x):
		return output


class UNet4_3(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_3,self).__init__()

	def forward(self,x):
		return output


class UNet4_4(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet4_4,self).__init__()

	def forward(self,x):
		return output


class UNet5_1(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_1,self).__init__()

	def forward(self,x):
		return output


class UNet5_2(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_2,self).__init__()

	def forward(self,x):
		return output


class UNet5_3(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_3,self).__init__()

	def forward(self,x):
		return output


class UNet5_4(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet5_4,self).__init__()

	def forward(self,x):
		return output


class UNet6_1(torch.nn.Module):
	def __init__(self,in_channels=3,out_channels=1):
		super(UNet3_1,self).__init__()

	def forward(self,x):
		return output

if __name__ == '__main__':
	test_net = UNet1_1()