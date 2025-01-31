import torch
import torch.nn.functional as F
from torch import nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )
    def forward(self, x):
        return self.encoder_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  in_channels//2, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            activation,
            nn.Conv2d(in_channels//2, out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )
    def forward(self, x):
        return self.decoder_block(x)

class AttentionBlock(nn.Module):
    def __init__(self, gate_channels, connections_channels, intermediate_channels):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(connections_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, x):
        g1 = self.W_g(gate)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Net(nn.Module):
    def __init__(self, activation_fun):
        super().__init__()

        # Config
        in_channels  = 4   # Input images have 4 channels
        out_channels = 3   # Mask has 3 channels
        n_filters    = 32  # Scaled down from 64 in original paper
        activation   = nn.ReLU()
        if activation_fun == 'LeakyReLU':
          activation = nn.LeakyReLU(negative_slope=0.1)
          print('Activation: Leaky ReLU')
        else:
            print('Activation: ReLU')

        self.downsample  = nn.MaxPool2d((2,2), stride=2)
        self.upsample    = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_block_1 = EncoderBlock(in_channels, 1*n_filters, activation)
        self.enc_block_2 = EncoderBlock(1*n_filters, 2*n_filters, activation)
        self.enc_block_3 = EncoderBlock(2*n_filters, 4*n_filters, activation)
        self.enc_block_4 = EncoderBlock(4*n_filters, 8*n_filters, activation)

        self.bottleneck  = nn.Sequential(
            nn.Conv2d(8*n_filters, 16*n_filters, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(16*n_filters),
            activation,
            nn.Conv2d(16*n_filters,  8*n_filters, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(8*n_filters),
            activation
        )

        self.dec_block_4 = DecoderBlock(16*n_filters, 4*n_filters, activation)
        self.dec_block_3 = DecoderBlock(8*n_filters, 2*n_filters, activation)
        self.dec_block_2 = DecoderBlock(4*n_filters, 1*n_filters, activation)
        self.dec_block_1 = DecoderBlock(2*n_filters, 1*n_filters, activation)

        self.output      = nn.Conv2d(1*n_filters,  out_channels, kernel_size=(1,1), stride=1, padding=0)

        self.att_4 = AttentionBlock(8 * n_filters, 8 * n_filters, 4 * n_filters)
        self.att_3 = AttentionBlock(4 * n_filters, 4 * n_filters, 2 * n_filters)
        self.att_2 = AttentionBlock(2 * n_filters, 2 * n_filters, 1 * n_filters)
        self.att_1 = AttentionBlock(1 * n_filters, 1 * n_filters, n_filters // 2)

    def forward(self, x):
        skip_1 = self.enc_block_1(x)
        x = self.downsample(skip_1)
        skip_2 = self.enc_block_2(x)
        x = self.downsample(skip_2)
        skip_3 = self.enc_block_3(x)
        x = self.downsample(skip_3)
        skip_4 = self.enc_block_4(x)
        x = self.downsample(skip_4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with Attention
        x = self.upsample(x)
        skip_4 = self.att_4(x, skip_4)
        x = torch.cat((x, skip_4), axis=1)
        x = self.dec_block_4(x)

        x = self.upsample(x)
        skip_3 = self.att_3(x, skip_3)
        x = torch.cat((x, skip_3), axis=1)
        x = self.dec_block_3(x)

        x = self.upsample(x)
        skip_2 = self.att_2(x, skip_2)
        x = torch.cat((x, skip_2), axis=1)
        x = self.dec_block_2(x)

        x = self.upsample(x)
        skip_1 = self.att_1(x, skip_1)
        x = torch.cat((x, skip_1), axis=1)
        x = self.dec_block_1(x)

        x = self.output(x)
        return x