import torch 
import torch.nn as nn
import torchvision.transforms.functional as F

from torchsummary import summary 

from torch.ao.quantization import QuantStub, DeQuantStub
class CNNBlock(nn.Sequential):
    """
    Classique CNN Block avec une couche de convolution 3x3, une batch normalalisation 
    et une fonction d'activation ReLU
    """

    def __init__(
        self,
        in_channels : int, 
        out_channels : int
    ) -> None:

        super(CNNBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, bias=False, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class CNNBlockX2(nn.Module):
    """
    Deux bloques classiques l'un après l'autre
    """

    def __init__(
        self,
        in_channels : int,
        out_channels : int
    ) -> None: 

        super(CNNBlockX2, self).__init__()
        self.cnnBlockX2 = nn.Sequential(
            CNNBlock(in_channels = in_channels, out_channels = out_channels),
            CNNBlock(in_channels = out_channels, out_channels = out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnnBlockX2(x)
    
class EncoderBlock(nn.Module):
    """
    Bloque complet d'encoder avec un downsampling (maxpool) puis un double bloque de convolution
    """

    def __init__(self,
        in_channels : int,
        out_channels : int
    ) -> None:
        super(EncoderBlock,self).__init__()
        self.encoderBlock = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            CNNBlockX2(in_channels = in_channels, out_channels = out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoderBlock(x)

class DecoderBlock(nn.Module):
    """
    Decoder Block recuperant les résidus pour les concatenés, 
    applique une double convolution puis un upsampling
    """      

    def __init__(self,
        in_channels : int,
        out_channels : int
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.convTranspose = nn.ConvTranspose2d(
                                in_channels = in_channels, 
                                out_channels = out_channels, 
                                kernel_size = 2, 
                                stride = 2)
        self.cnnBlockX2 = CNNBlockX2(in_channels = in_channels, out_channels = out_channels)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:

        x = self.convTranspose(x)

        height_offset = (res.size(dim=2) - x.size(dim=2))//2
        width_offset = (res.size(dim=3) - x.size(dim=3))//2

        res_cropped = F.crop(res,height_offset, width_offset, x.size(dim=2), x.size(dim=3))

        x = torch.cat((res_cropped,x), dim = 1)
        x = self.cnnBlockX2(x)
        return x
         
class UNet(nn.Module):
    """
    Model Complet UNet provenant du papier : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, 
        input_channel : int,
        output_class : int
    ) -> None:

        super(UNet,self).__init__()
        self.entryDoubleBlock = CNNBlockX2(in_channels = input_channel, out_channels = 32)
        self.encoderBlock1 = EncoderBlock(in_channels = 32, out_channels = 64)
        self.encoderBlock2 = EncoderBlock(in_channels = 64, out_channels = 128)
        self.encoderBlock3 = EncoderBlock(in_channels = 128, out_channels = 256)

        self.bottleNeck = EncoderBlock(in_channels=256, out_channels=512)

        self.decoderBlock1 = DecoderBlock(in_channels = 512, out_channels = 256)
        self.decoderBlock2 = DecoderBlock(in_channels = 256, out_channels = 128)
        self.decoderBlock3 = DecoderBlock(in_channels = 128, out_channels = 64)
        self.decoderBlock4 = DecoderBlock(in_channels = 64, out_channels = 32)

        self.outputConv = nn.Conv2d(in_channels = 32, out_channels = output_class, kernel_size = 1)
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x1 = self.entryDoubleBlock(x)
        x2 = self.encoderBlock1(x1)
        x3 = self.encoderBlock2(x2)
        x4 = self.encoderBlock3(x3)
        x = self.bottleNeck(x4)
        x = self.decoderBlock1(x, x4)
        x = self.decoderBlock2(x, x3)
        x = self.decoderBlock3(x, x2)
        x = self.decoderBlock4(x, x1)
        x = self.outputConv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == CNNBlock:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)

def UNet_test():
    device = torch.device("cuda")
    class_nb = 3

    X = torch.rand(1, 3, 1024, 1216, device=device)
    float_model = UNet(3, class_nb).to(device)
    summary(float_model,(3,1024,1216))
    torch.save(float_model,"Model.pth")

    float_model.qconfig = torch.ao.quantization.default_qconfig
    print(float_model.qconfig)
    torch.ao.quantization.prepare(float_model, inplace=True)
    torch.ao.quantization.convert(float_model, inplace=True)
    torch.save(float_model.state_dict(), "teeest")
    #float_model.eval()
    #print(float_model.fuse_model())

if __name__ == "__main__":
    UNet_test()