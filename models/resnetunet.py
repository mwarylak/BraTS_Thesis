import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=(3,3), stride=1, padding=1),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
            activation
        )

    def forward(self, x):
        return self.encoder_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  in_channels//2, kernel_size=(3,3), stride=1, padding=1),
            activation,
            nn.Conv2d(in_channels//2, out_channels, kernel_size=(3,3), stride=1, padding=1),
            activation
        )
    def forward(self, x):
        return self.decoder_block(x)

class Net(nn.Module):
    def __init__(self, activation_fun):
        super().__init__()

        in_channels  = 4  
        out_channels = 3   
        n_filters    = 32
        activation = nn.LeakyReLU(negative_slope=0.1)
        if activation_fun == 'ReLU':
          activation = nn.ReLU()
          print('Activation: ReLU')
        else:
            print('Activation: Leaky ReLU')

        # Pre-trained encoder (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder_pretrained = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_pretrained[0] = nn.Conv2d(
            in_channels, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False
        )

        # Down and upsampling
        self.downsample = nn.MaxPool2d((2, 2), stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16* n_filters, 16* n_filters, kernel_size=(3, 3), stride=1, padding=1),
            activation,
            nn.Conv2d(16* n_filters, 16* n_filters, kernel_size=(3, 3), stride=1, padding=1),
            activation
        )

        # Decoder
        self.dec_block_4 = DecoderBlock(24* n_filters, 4* n_filters, activation)
        self.dec_block_3 = DecoderBlock(8* n_filters, 2* n_filters, activation)
        self.dec_block_2 = DecoderBlock(4* n_filters, 1* n_filters, activation)
        self.dec_block_1 = DecoderBlock(3* n_filters, 1* n_filters, activation)

        # Output 
        self.output = nn.Conv2d(32, out_channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        # Encoder (using pre-trained ResNet18)
        skip_1 = self.encoder_pretrained[0:3](x)  
        skip_2 = self.encoder_pretrained[3:5](skip_1)  
        skip_3 = self.encoder_pretrained[5](skip_2)  
        skip_4 = self.encoder_pretrained[6](skip_3) 
        x = self.encoder_pretrained[7](skip_4)      

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x      = self.upsample(x)
        x      = torch.cat((x, skip_4), axis=1)  
        x      = self.dec_block_4(x)

        x      = self.upsample(x)
        x      = torch.cat((x, skip_3), axis=1) 
        x      = self.dec_block_3(x)

        x      = self.upsample(x)
        x      = torch.cat((x, skip_2), axis=1)  
        x      = self.dec_block_2(x)

        x      = self.upsample(x)
        x      = torch.cat((x, skip_1), axis=1)  
        x      = self.dec_block_1(x)

        x      = self.output(x)
        return x