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


class DecBlock1(torch.nn.Module):
	def __init__(self,i_ch,o_ch):
		super(DecBlock1,self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(i_ch,o_ch,kernel_size=2,stride=2),
			torch.nn.ReLU(inplace=True),
			ConvBlock(o_ch,o_ch)
		)

	def forward(self,x):
		x = self.layers(x)
		return x

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

class EncBlock1(torch.nn.Module):
	def __init__(self,i_ch,o_ch,block_type):
		super(EncBlock,self).__init__()
		if block_type == 'A'
			C1 = torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		elif block_type == 'B'
			C1 = torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=2,padding=1,bias=False)
		else:
			raise ValueError("BLOCK TYPE NOT DEFINED IN CONVOLUTION BLOCK 1.")
		B1 = torch.nn.BatchNorm2d(o_ch)
		R1 = torch.nn.ReLU(inplace=True)
		C2 = torch.nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False)
		B2 = torch.nn.BatchNorm2d(o_ch)
		R2 = torch.nn.ReLU(inplace=True)

	def forward(self,x):
		x = C1(x)
		x = B1(x)
		x = R1(x)
		x = C2(x)
		x = B2(x)
		x = R2(x)
		return x

class Bottleneck1(torch.nn.Module): # --------> SAME AS ConvBlock1 ABOVE, LEAVE 1 ONLY:TODO
	def __init__(self,i_ch):
		super(Bottleneck1,self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.Conv2d(i_ch,i_ch,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(i_ch),
			torch.nn.ReLU(i_ch),
			torch.nn.Conv2d(i_ch,i_ch,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(i_ch),
			torch.nn.ReLU(i_ch)		
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
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet1_1, self).__init__()
        
        # ENCODER
        self.embedding = EmbeddingLayer(in_channels,16)
        self.encoder_1 = EncBlock1(16,32,'A')
        self.down_op_1 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.encoder_2 = EncBlock1(32,64,'A')
        self.down_op_2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.encoder_3 = EncBlock1(64,128,'A')
        self.down_op_3 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)        
        self.encoder_4 = EncBlock1(128,256,'A')
        self.down_op_4 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)        
        
        # BOTTLENECK
        self.bottleneck = EncBlock1(256,512,'A')
        
        # DECODER
        self.up_op_1   = torch.nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,bias=False)
        self.decoder_4 = DecBlock1(512,256)
        self.decoder_3 = DecBlock1(256,128)
        self.decoder_2 = DecBlock1(128,64)
        self.decoder_1 = DecBlock1(64,32)
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
        i4 = F.conv_transpose2d(o5,(512,256,2,2),stride=2)
        dec4 = self.decoder_4(torch.cat([o4,i4], dim=1))
        dec4 = 
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        # FINAL LAYER
        output = self.last_layer(dec1)
        
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