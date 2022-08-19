__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
import torchvision
from model.main_blocks import conv_block, transpose_conv_block


class VGG_Unet(nn.Module):
    def __init__(self,
                 out_features=1,
                 norm_type='gn',
                 pretrained=True,
                 reg_grad=False,
                 device='cuda'):
        nn.Module.__init__(self)

        self.pretrained_mean = torch.tensor([0.485, 0.456, 0.406],
                                            requires_grad=False).view((1, 3, 1, 1)).to(device)
        self.pretrained_std = torch.tensor([0.229, 0.224, 0.225],
                                           requires_grad=False).view((1, 3, 1, 1)).to(device)

        vgg = torchvision.models.vgg16(pretrained=pretrained).features
        for param in vgg.parameters():
            param.requires_grad = reg_grad

        concats = [4, 9, 16, 23, 30]

        self.enc1 = vgg[:concats[0]]
        self.enc2 = vgg[concats[0]:concats[1]]
        self.enc3 = vgg[concats[1]:concats[2]]
        self.enc4 = vgg[concats[2]:concats[3]]
        self.enc5 = vgg[concats[3]:concats[4]]

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)

        self.decode1 = nn.Sequential(conv_block(in_features=512,
                                                out_features=512,
                                                norm_type=norm_type),
                                     conv_block(in_features=512,
                                                out_features=512,
                                                norm_type=norm_type))

        self.transpose1 = transpose_conv_block(in_features=512,
                                               out_features=512,
                                               norm_type=norm_type)

        self.decode2 = nn.Sequential(conv_block(in_features=1024,
                                                out_features=512,
                                                norm_type=norm_type),
                                     conv_block(in_features=512,
                                                out_features=512,
                                                norm_type=norm_type))

        self.transpose2 = transpose_conv_block(in_features=512,
                                               out_features=512,
                                               norm_type=norm_type)

        self.decode3 = nn.Sequential(conv_block(in_features=1024,
                                                out_features=512,
                                                norm_type=norm_type),
                                     conv_block(in_features=512,
                                                out_features=512,
                                                norm_type=norm_type))

        self.transpose3 = transpose_conv_block(in_features=512,
                                               out_features=256,
                                               norm_type=norm_type)

        self.decode4 = nn.Sequential(conv_block(in_features=512,
                                                out_features=256,
                                                norm_type=norm_type),
                                     conv_block(in_features=256,
                                                out_features=256,
                                                norm_type=norm_type))

        self.transpose4 = transpose_conv_block(in_features=256,
                                               out_features=128,
                                               norm_type=norm_type)

        self.decode5 = nn.Sequential(conv_block(in_features=256,
                                                out_features=128,
                                                norm_type=norm_type),
                                     conv_block(in_features=128,
                                                out_features=128,
                                                norm_type=norm_type))
        self.transpose5 = transpose_conv_block(in_features=128,
                                               out_features=64,
                                               norm_type=norm_type)

        self.decode6 = nn.Sequential(conv_block(in_features=128,
                                                out_features=64,
                                                norm_type=norm_type),
                                     conv_block(in_features=64,
                                                out_features=64,
                                                norm_type=norm_type))

        self.output = nn.Sequential(
            nn.Conv2d(64,
                      out_features,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.Sigmoid())


    def forward(self, x):
        x = (x - self.pretrained_mean) / self.pretrained_std
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x = self.maxpool(x5)
        x = self.transpose1(self.decode1(x))
        x = torch.cat([x, x5], dim=1)
        x = self.transpose2(self.decode2(x))
        x = torch.cat([x, x4], dim=1)
        x = self.transpose3(self.decode3(x))
        x = torch.cat([x, x3], dim=1)
        x = self.transpose4(self.decode4(x))
        x = torch.cat([x, x2], dim=1)
        x = self.transpose5(self.decode5(x))
        x = torch.cat([x, x1], dim=1)
        x = self.decode6(x)
        x = self.output(x)

        return x