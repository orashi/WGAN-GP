import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as M
from torch.autograd import Variable
from PIL import Image
import math
import numpy as np

# import srWDenseGAN

ngf = 64


# class _netG(nn.Module):
#     def __init__(self):
#         super(_netG, self).__init__()
#         # self.Bavgpool = nn.AvgPool2d(4)
#
#         self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
#         # state size. (ngf) x W x H
#
#         self.convB11 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB11 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB12 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB12 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB21 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB21 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB22 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB22 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB31 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB31 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB32 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB32 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB41 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB41 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB42 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB42 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB51 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB51 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB52 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB52 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB61 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB61 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB62 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB62 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB71 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB71 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB72 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB72 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB81 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB81 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB82 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB82 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB91 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB91 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB92 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB92 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB101 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB101 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB102 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB102 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB111 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB111 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB112 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB112 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB121 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB121 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB122 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB122 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB131 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB131 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB132 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB132 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB141 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB141 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB142 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB142 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB151 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB151 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB152 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB152 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB161 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB161 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB162 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB162 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bn6 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
#         self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H
#
#         self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
#         self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H
#
#         self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x), True)
#
#         #####
#         residual = out
#
#         out = F.relu(self.bnB11(self.convB11(out)), True)
#         out = self.bnB12(self.convB12(out)) + residual
#         #
#         res2 = out
#
#         out = F.relu(self.bnB21(self.convB21(out)), True)
#         out = self.bnB22(self.convB22(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB31(self.convB31(out)), True)
#         out = self.bnB32(self.convB32(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB41(self.convB41(out)), True)
#         out = self.bnB42(self.convB42(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB51(self.convB51(out)), True)
#         out = self.bnB52(self.convB52(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB61(self.convB61(out)), True)
#         out = self.bnB62(self.convB62(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB71(self.convB71(out)), True)
#         out = self.bnB72(self.convB72(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB81(self.convB81(out)), True)
#         out = self.bnB82(self.convB82(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB91(self.convB91(out)), True)
#         out = self.bnB92(self.convB92(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB101(self.convB101(out)), True)
#         out = self.bnB102(self.convB102(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB111(self.convB111(out)), True)
#         out = self.bnB112(self.convB112(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB121(self.convB121(out)), True)
#         out = self.bnB122(self.convB122(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB131(self.convB131(out)), True)
#         out = self.bnB132(self.convB132(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB141(self.convB141(out)), True)
#         out = self.bnB142(self.convB142(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB151(self.convB151(out)), True)
#         out = self.bnB152(self.convB152(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB161(self.convB161(out)), True)
#         out = self.bnB162(self.convB162(out)) + res2
#         ######
#
#         out = self.bn6(self.conv6(out))
#         out = out + residual
#
#         out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
#         out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)
#
#         out = F.tanh(self.conv9(out))
#
#         return out
# class _netG(nn.Module):
#     def __init__(self):
#         super(_netG, self).__init__()
#         self.convN = nn.ConvTranspose2d(100, 4, 96 // 4, 1, 0, bias=False)
#
#         self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
#         # state size. (ngf) x W x H
#
#         self.convB11 = nn.Conv2d(ngf + 4, ngf, 3, 1, 1, bias=False)
#         self.bnB11 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB12 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB12 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB21 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB21 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB22 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB22 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB31 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB31 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB32 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB32 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB41 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB41 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB42 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB42 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB51 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB51 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB52 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB52 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB61 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB61 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB62 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB62 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB71 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB71 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB72 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB72 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB81 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB81 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB82 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB82 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB91 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB91 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB92 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB92 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB101 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB101 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB102 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB102 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB111 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB111 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB112 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB112 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB121 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB121 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB122 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB122 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB131 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB131 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB132 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB132 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB141 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB141 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB142 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB142 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB151 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB151 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB152 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB152 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB161 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB161 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB162 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB162 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bn6 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
#         self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H
#
#         self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
#         self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H
#
#         self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x), True)
#
#         #####
#         residual = out
#
#         noise = torch.FloatTensor(out.size()[0], 4, out.size()[2], out.size()[3])
#         noise = Variable(noise)
#         noise = noise.cuda()
#         noise.data.normal_(0, 1)
#
#         out = F.relu(self.bnB11(self.convB11(torch.cat((out, noise), 1))), True)
#         out = self.bnB12(self.convB12(out)) + residual
#         #
#         res2 = out
#
#         out = F.relu(self.bnB21(self.convB21(out)), True)
#         out = self.bnB22(self.convB22(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB31(self.convB31(out)), True)
#         out = self.bnB32(self.convB32(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB41(self.convB41(out)), True)
#         out = self.bnB42(self.convB42(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB51(self.convB51(out)), True)
#         out = self.bnB52(self.convB52(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB61(self.convB61(out)), True)
#         out = self.bnB62(self.convB62(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB71(self.convB71(out)), True)
#         out = self.bnB72(self.convB72(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB81(self.convB81(out)), True)
#         out = self.bnB82(self.convB82(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB91(self.convB91(out)), True)
#         out = self.bnB92(self.convB92(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB101(self.convB101(out)), True)
#         out = self.bnB102(self.convB102(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB111(self.convB111(out)), True)
#         out = self.bnB112(self.convB112(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB121(self.convB121(out)), True)
#         out = self.bnB122(self.convB122(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB131(self.convB131(out)), True)
#         out = self.bnB132(self.convB132(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB141(self.convB141(out)), True)
#         out = self.bnB142(self.convB142(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB151(self.convB151(out)), True)
#         out = self.bnB152(self.convB152(out)) + res2
#         #
#         res2 = out
#
#         out = F.relu(self.bnB161(self.convB161(out)), True)
#         out = self.bnB162(self.convB162(out)) + res2
#         ######
#
#         out = self.bn6(self.conv6(out))
#         out = out + residual
#
#         out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
#         out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)
#
#         out = F.tanh(self.conv9(out))
#
#         return out

# class _netG(nn.Module):
#     def __init__(self):
#         super(_netG, self).__init__()
#         #      self.convN = nn.ConvTranspose2d(100, 4, opt.imageSize // 4, 1, 0, bias=False)
#
#         self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
#         # state size. (ngf) x W x H
#
#         self.convB11 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB11 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB12 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB12 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB21 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB21 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB22 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB22 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB31 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB31 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB32 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB32 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB41 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB41 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB42 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB42 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB51 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB51 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB52 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB52 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB61 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB61 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB62 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB62 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB71 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB71 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB72 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB72 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB81 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB81 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB82 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB82 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB91 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB91 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB92 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB92 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB101 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB101 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB102 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB102 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB111 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB111 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB112 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB112 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB121 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB121 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB122 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB122 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB131 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB131 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB132 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB132 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB141 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB141 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB142 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB142 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB151 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB151 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB152 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB152 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB161 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB161 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB162 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB162 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bn6 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
#         self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H
#
#         self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
#         self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H
#
#         self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)
#
#     def forward(self, x):
#         out = self.conv1(x)
#
#         #####  0.3??????
#         residual = out
#
#         # noise = torch.FloatTensor(out.size()[0], 4, out.size()[2], out.size()[3])
#         # noise = Variable(noise)
#         # if opt.cuda:
#         #    noise = noise.cuda()
#         # noise.data.normal_(0, 1)
#
#         #        noise = self.convN(noise)
#         out = self.convB11(F.relu(self.bnB11(out), True))
#         out = self.convB12(F.relu(self.bnB12(out), True)) + residual
#
#         #
#         res2 = out
#
#         out = self.convB21(F.relu(self.bnB21(out), True))
#         out = self.convB22(F.relu(self.bnB22(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB31(F.relu(self.bnB31(out), True))
#         out = self.convB32(F.relu(self.bnB32(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB41(F.relu(self.bnB41(out), True))
#         out = self.convB42(F.relu(self.bnB42(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB51(F.relu(self.bnB51(out), True))
#         out = self.convB52(F.relu(self.bnB52(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB61(F.relu(self.bnB61(out), True))
#         out = self.convB62(F.relu(self.bnB62(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB71(F.relu(self.bnB71(out), True))
#         out = self.convB72(F.relu(self.bnB72(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB81(F.relu(self.bnB81(out), True))
#         out = self.convB82(F.relu(self.bnB82(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB91(F.relu(self.bnB91(out), True))
#         out = self.convB92(F.relu(self.bnB92(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB101(F.relu(self.bnB101(out), True))
#         out = self.convB102(F.relu(self.bnB102(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB111(F.relu(self.bnB111(out), True))
#         out = self.convB112(F.relu(self.bnB112(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB121(F.relu(self.bnB121(out), True))
#         out = self.convB122(F.relu(self.bnB122(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB131(F.relu(self.bnB131(out), True))
#         out = self.convB132(F.relu(self.bnB132(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB141(F.relu(self.bnB141(out), True))
#         out = self.convB142(F.relu(self.bnB142(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB151(F.relu(self.bnB151(out), True))
#         out = self.convB152(F.relu(self.bnB152(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB161(F.relu(self.bnB161(out), True))
#         out = self.convB162(F.relu(self.bnB162(out), True)) + res2
#         ######
#
#         out = self.conv6(F.relu(self.bn6(out), True))
#         out = out + residual
#
#         out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
#         out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)
#
#         out = F.tanh(self.conv9(out))
#
#         return out

# class _netG(nn.Module):
#     def __init__(self):
#         super(_netG, self).__init__()
#         #      self.convN = nn.ConvTranspose2d(100, 4, opt.imageSize // 4, 1, 0, bias=False)
#
#         self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
#         # state size. (ngf) x W x H
#
#         self.convB11 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB11 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB12 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB12 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB21 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB21 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB22 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB22 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB31 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB31 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB32 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB32 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB41 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB41 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB42 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB42 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB51 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB51 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB52 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB52 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB61 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB61 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB62 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB62 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB71 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB71 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB72 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB72 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB81 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB81 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB82 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB82 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB91 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB91 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB92 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB92 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB101 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB101 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB102 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB102 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB111 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB111 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB112 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB112 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB121 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB121 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB122 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB122 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB131 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB131 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB132 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB132 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB141 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB141 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB142 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB142 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB151 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB151 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB152 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB152 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB161 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB161 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.convB162 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         self.bnB162 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         # self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
#         # self.bn6 = nn.BatchNorm2d(ngf)
#         # state size. (ngf) x W x H
#
#         self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
#         self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H
#
#         self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
#         self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H
#
#         self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)
#
#     def forward(self, x):
#         out = self.conv1(x)
#
#         #####  0.3??????
#         # residual = out
#         res2 = out
#
#         # noise = torch.FloatTensor(out.size()[0], 4, out.size()[2], out.size()[3])
#         # noise = Variable(noise)
#         # if opt.cuda:
#         #    noise = noise.cuda()
#         # noise.data.normal_(0, 1)
#
#         #        noise = self.convN(noise)
#         out = self.convB11(F.relu(self.bnB11(out), True))
#         out = self.convB12(F.relu(self.bnB12(out), True)) + res2
#
#         #
#         res2 = out
#
#         out = self.convB21(F.relu(self.bnB21(out), True))
#         out = self.convB22(F.relu(self.bnB22(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB31(F.relu(self.bnB31(out), True))
#         out = self.convB32(F.relu(self.bnB32(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB41(F.relu(self.bnB41(out), True))
#         out = self.convB42(F.relu(self.bnB42(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB51(F.relu(self.bnB51(out), True))
#         out = self.convB52(F.relu(self.bnB52(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB61(F.relu(self.bnB61(out), True))
#         out = self.convB62(F.relu(self.bnB62(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB71(F.relu(self.bnB71(out), True))
#         out = self.convB72(F.relu(self.bnB72(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB81(F.relu(self.bnB81(out), True))
#         out = self.convB82(F.relu(self.bnB82(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB91(F.relu(self.bnB91(out), True))
#         out = self.convB92(F.relu(self.bnB92(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB101(F.relu(self.bnB101(out), True))
#         out = self.convB102(F.relu(self.bnB102(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB111(F.relu(self.bnB111(out), True))
#         out = self.convB112(F.relu(self.bnB112(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB121(F.relu(self.bnB121(out), True))
#         out = self.convB122(F.relu(self.bnB122(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB131(F.relu(self.bnB131(out), True))
#         out = self.convB132(F.relu(self.bnB132(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB141(F.relu(self.bnB141(out), True))
#         out = self.convB142(F.relu(self.bnB142(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB151(F.relu(self.bnB151(out), True))
#         out = self.convB152(F.relu(self.bnB152(out), True)) + res2
#         #
#         res2 = out
#
#         out = self.convB161(F.relu(self.bnB161(out), True))
#         out = self.convB162(F.relu(self.bnB162(out), True)) + res2
#         ######
#
#         # out = self.conv6(F.relu(self.bn6(out), True))
#         # out = out + residual
#
#         out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
#         out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)
#
#         out = F.tanh(self.conv9(out))
#
#         return out
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        #      self.convN = nn.ConvTranspose2d(100, 4, opt.imageSize // 4, 1, 0, bias=False)

        self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
        # state size. (ngf) x W x H

        self.convB11 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB11 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB12 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB12 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB21 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB21 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB22 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB22 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB31 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB31 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB32 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB32 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB41 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB41 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB42 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB42 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB51 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB51 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB52 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB52 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB61 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB61 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB62 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB62 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB71 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB71 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB72 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB72 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB81 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB81 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB82 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB82 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB91 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB91 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB92 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB92 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB101 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB101 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB102 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB102 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB111 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB111 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB112 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB112 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB121 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB121 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB122 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB122 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB131 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB131 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB132 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB132 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB141 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB141 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB142 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB142 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB151 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB151 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB152 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB152 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB161 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB161 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.convB162 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bnB162 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        # self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        # self.bn6 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H

        self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H

        self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)

        #####  0.3??????
        # residual = out
        res2 = out

        # noise = torch.FloatTensor(out.size()[0], 4, out.size()[2], out.size()[3])
        # noise = Variable(noise)
        # if opt.cuda:
        #    noise = noise.cuda()
        # noise.data.normal_(0, 1)

        #        noise = self.convN(noise)
        out = self.convB11(F.relu(self.bnB11(out), True))
        out = self.convB12(F.relu(self.bnB12(out), True)) + res2

        #
        res2 = out

        out = self.convB21(F.relu(self.bnB21(out), True))
        out = self.convB22(F.relu(self.bnB22(out), True)) + res2
        #
        res2 = out

        out = self.convB31(F.relu(self.bnB31(out), True))
        out = self.convB32(F.relu(self.bnB32(out), True)) + res2
        #
        res2 = out

        out = self.convB41(F.relu(self.bnB41(out), True))
        out = self.convB42(F.relu(self.bnB42(out), True)) + res2
        #
        res2 = out

        out = self.convB51(F.relu(self.bnB51(out), True))
        out = self.convB52(F.relu(self.bnB52(out), True)) + res2
        #
        res2 = out

        out = self.convB61(F.relu(self.bnB61(out), True))
        out = self.convB62(F.relu(self.bnB62(out), True)) + res2
        #
        res2 = out

        out = self.convB71(F.relu(self.bnB71(out), True))
        out = self.convB72(F.relu(self.bnB72(out), True)) + res2
        #
        res2 = out

        out = self.convB81(F.relu(self.bnB81(out), True))
        out = self.convB82(F.relu(self.bnB82(out), True)) + res2
        #
        res2 = out

        out = self.convB91(F.relu(self.bnB91(out), True))
        out = self.convB92(F.relu(self.bnB92(out), True)) + res2
        #
        res2 = out

        out = self.convB101(F.relu(self.bnB101(out), True))
        out = self.convB102(F.relu(self.bnB102(out), True)) + res2
        #
        res2 = out

        out = self.convB111(F.relu(self.bnB111(out), True))
        out = self.convB112(F.relu(self.bnB112(out), True)) + res2
        #
        res2 = out

        out = self.convB121(F.relu(self.bnB121(out), True))
        out = self.convB122(F.relu(self.bnB122(out), True)) + res2
        #
        res2 = out

        out = self.convB131(F.relu(self.bnB131(out), True))
        out = self.convB132(F.relu(self.bnB132(out), True)) + res2
        #
        res2 = out

        out = self.convB141(F.relu(self.bnB141(out), True))
        out = self.convB142(F.relu(self.bnB142(out), True)) + res2
        #
        res2 = out

        out = self.convB151(F.relu(self.bnB151(out), True))
        out = self.convB152(F.relu(self.bnB152(out), True)) + res2
        #
        res2 = out

        out = self.convB161(F.relu(self.bnB161(out), True))
        out = self.convB162(F.relu(self.bnB162(out), True)) + res2
        ######

        # out = self.conv6(F.relu(self.bn6(out), True))
        # out = out + residual

        out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
        out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)

        out = F.tanh(self.conv9(out))

        return out


netG = _netG()
netG.load_state_dict(torch.load('15_100K_UNP_P.pth'))
netG.cuda().eval()
img = Image.open('LR (1).jpg')

ts = transforms.Compose([  # transforms.RandomCrop(96),
    transforms.Scale((img.size[0]//4, img.size[1]//4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ts2 = transforms.Compose([  # transforms.RandomCrop(96),
    # transforms.Scale((img.size[0] * 4, img.size[1] * 4), Image.BICUBIC),
    transforms.Scale((img.size[0], img.size[1]), Image.BICUBIC),

    # transforms.Scale((200, 50), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

im = ts(img)
im2 = ts2(img)

x = Variable(im.view(1, 3, im.size()[1], im.size()[2]), volatile=True).cuda()
out = netG(x)
vutils.save_image((out.data.cpu() / 1.5 + im2.view(1, 3, im2.size()[1], im2.size()[2])).mul(0.5).add(0.5),
                  '15_100K_UNP_P.jpg')
