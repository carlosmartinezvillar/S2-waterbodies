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


class UpBlock(torch.nn.Module):
	def __init__(self,i_ch,o_ch):
		super(UpBlock,self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(i_ch,o_ch,kernel_size=2,stride=2),
			torch.nn.ReLU(inplace=True),
			ConvBlock(o_ch,o_ch)
		)

	def forward(self,x):
		x = self.layers(x)
		return x

    

class ConvBlock(torch.nn.Module):
	def __init__(self,i_ch,o_ch):
		super(ConvBlock,self).__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.Conv2d(i_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(o_ch),			
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(o_ch),			
			torch.nn.ReLU(inplace=True)
		)

	def forward(self,x):
		return self.layers(x)


class BaseUNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(BaseUNet, self).__init__()
        
        # ENCODER
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        
        # BOTTLENECK
        self.bottleneck = ConvBlock(512, 1024)
        
        # DECODER
        self.decoder4 = UpBlock(1024, 512)
        self.decoder3 = UpBlock(512, 256)
        self.decoder2 = UpBlock(256, 128)
        self.decoder1 = UpBlock(128, 64)
        
        # LAST LAYER
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        # BOTTLENECK
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # DECODER
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        # FINAL LAYER
        output = self.final_conv(dec1)
        
        return output

if __name__ == '__main__':
	test_net = BaseUNet()