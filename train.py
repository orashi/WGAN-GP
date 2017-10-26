import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import toyData as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as M
from torch.autograd import Variable
from PIL import Image
import math
import numpy as np
from visdom import Visdom

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cut', type=int, default=1)
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')
parser.add_argument('--ganW', type=float, default=0.01, help='gan penalty weight')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=12500, help='start base of pure mse loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')

opt = parser.parse_args()
print(opt)

viz = Visdom()

imageW = viz.images(
    np.random.rand(3, 512, 256),
    opts=dict(title='fakeHR', caption='fakeHR')
)

cuts = opt.cut  # save division
ngf = opt.ngf
ndf = opt.ndf
gen_iterations = opt.geni

try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed setup
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# folder dataset
LRTrans = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage(),
    transforms.Scale(opt.imageSize // 4, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

HRTrans = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage(),
    transforms.Scale(opt.imageSize, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               # transforms.RandomSizedCrop(),
                               transforms.RandomCrop(opt.imageSize),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]),
                           target_transform=LRTrans,
                           target_transform2=HRTrans,
                           )

# def convert(input, i, LBA):
#     return LBA(Image.fromarray(
#         input[i].mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()))  # bicubic


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


############################
# G network
###########################
# custom weights initialization called on netG
def G_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data = init.kaiming_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def D_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        # self.Bavgpool = nn.AvgPool2d(4)

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

        self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H

        self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H

        self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x), True)

        #####
        residual = out

        out = F.relu(self.bnB11(self.convB11(out)), True)
        out = self.bnB12(self.convB12(out)) + residual
        #
        res2 = out

        out = F.relu(self.bnB21(self.convB21(out)), True)
        out = self.bnB22(self.convB22(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB31(self.convB31(out)), True)
        out = self.bnB32(self.convB32(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB41(self.convB41(out)), True)
        out = self.bnB42(self.convB42(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB51(self.convB51(out)), True)
        out = self.bnB52(self.convB52(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB61(self.convB61(out)), True)
        out = self.bnB62(self.convB62(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB71(self.convB71(out)), True)
        out = self.bnB72(self.convB72(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB81(self.convB81(out)), True)
        out = self.bnB82(self.convB82(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB91(self.convB91(out)), True)
        out = self.bnB92(self.convB92(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB101(self.convB101(out)), True)
        out = self.bnB102(self.convB102(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB111(self.convB111(out)), True)
        out = self.bnB112(self.convB112(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB121(self.convB121(out)), True)
        out = self.bnB122(self.convB122(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB131(self.convB131(out)), True)
        out = self.bnB132(self.convB132(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB141(self.convB141(out)), True)
        out = self.bnB142(self.convB142(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB151(self.convB151(out)), True)
        out = self.bnB152(self.convB152(out)) + res2
        #
        res2 = out

        out = F.relu(self.bnB161(self.convB161(out)), True)
        out = self.bnB162(self.convB162(out)) + res2
        ######

        out = self.bn6(self.conv6(out))
        out = out + residual

        out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
        out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)

        out = F.tanh(self.conv9(out))

        return out


netG = _netG()
netG.apply(G_weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


############################
# D network
###########################
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        # input is 3 x 96 x 96
        self.conv11 = nn.Conv2d(3, ndf, 3, 1, 1, bias=False)
        # state size. (ndf) x 96 x 96

        self.conv12 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
        self.bn12 = nn.BatchNorm2d(ndf)
        # state size. (ndf) x 48 x 48

        self.conv21 = nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False)
        self.bn21 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf * 2) x 48 x 48

        self.conv22 = nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False)
        self.bn22 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf * 2) x 24 x 24

        self.conv31 = nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False)
        self.bn31 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf * 4) x 24 x 24

        self.conv32 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.bn32 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf * 4) x 12 x 12

        self.conv41 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)
        self.bn41 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf * 8) x 12 x 12

        self.conv42 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False)
        self.bn42 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf * 8) x 6 x 6

        self.dense1 = nn.Linear(ndf * 8 * 6 * 6, 1)
        # self.dense2 = nn.Linear(1024, 1)

    def forward(self, x):
        out = F.leaky_relu(self.conv11(x), 0.2, True)
        out = F.leaky_relu(self.bn12(self.conv12(out)), 0.2, True)
        out = F.leaky_relu(self.bn21(self.conv21(out)), 0.2, True)
        out = F.leaky_relu(self.bn22(self.conv22(out)), 0.2, True)
        out = F.leaky_relu(self.bn31(self.conv31(out)), 0.2, True)
        out = F.leaky_relu(self.bn32(self.conv32(out)), 0.2, True)
        out = F.leaky_relu(self.bn41(self.conv41(out)), 0.2, True)
        out = F.leaky_relu(self.bn42(self.conv42(out)), 0.2, True)

        out = self.dense1(out.view(out.size(0), -1))
        # out = self.dense2(out)

        return out


netD = _netD()
netD.apply(D_weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion_MSE = nn.MSELoss()
L2_dist = nn.PairwiseDistance(2)
############################
# VGG loss
###########################
vgg19 = M.vgg19()
vgg19.load_state_dict(torch.load('vgg19.pth'))


class vgg54(nn.Module):
    def __init__(self):
        super(vgg54, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(vgg19.features.children())[:-7]
        )

    def forward(self, x):
        x = self.features(x)
        return x


vgg_features = vgg54()
for p in vgg_features.parameters():
    p.requires_grad = False  # to avoid computation

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
zasshu = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
fixed_zasshu = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
one = torch.FloatTensor([1])
mone = one * -1
suber = torch.FloatTensor(
    [np.full((96, 96), 0.485 - 0.5), np.full((96, 96), 0.456 - 0.5), np.full((96, 96), 0.406 - 0.5)])
diver = torch.FloatTensor([np.full((96, 96), 0.229), np.full((96, 96), 0.224), np.full((96, 96), 0.225)])

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    zasshu, fixed_zasshu = zasshu.cuda(), fixed_zasshu.cuda()
    criterion_MSE.cuda()
    L2_dist.cuda()
    vgg_features.cuda()
    suber, diver = suber.cuda(), diver.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))

flag = 1
flag2 = 1
flag3 = 1

for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # train the discriminator Diters times
        Diters = opt.Diters
        if gen_iterations < opt.baseGeni:  # MSE stage
            Diters = 1
        elif gen_iterations % 500 == 0:
            Diters = 100

        j = 0
        while j < Diters and i < len(dataloader):
            j += 1
            netD.zero_grad()

            data = data_iter.next()
            real_cpu, fake_cpu, perceptualRes_cpu = data

            i += 1
            ###############################
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
                fake_cpu = fake_cpu.cuda()
                perceptionRes_cpu = perceptionRes_cpu.cuda()

            if flag:  # fix samples
                viz.images(
                    real_cpu.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='real_cpu', caption='original')
                )
                vutils.save_image(real_cpu.mul(0.5).add(0.5),
                                  '%s/real_samples.png' % opt.outf)
                fixed_zasshu.resize_as_(fake_cpu).copy_(fake_cpu)
                viz.images(
                    fixed_zasshu.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='LR', caption='LR')
                )
                vutils.save_image(fixed_zasshu.mul(0.5).add(0.5),
                                  '%s/ori_samples.png' % opt.outf)
                flag -= 1

            input.resize_as_(real_cpu).copy_(real_cpu - perceptionRes_cpu)
            zasshu.resize_as_(fake_cpu).copy_(fake_cpu)

            inputv = Variable(input)
            errD_real_vec = netD(inputv)
            errD_real = errD_real_vec.mean(0).view(1)
            errD_real.backward(mone, retain_variables=True)  # backward on score on real

            # train with fake
            inputv = Variable(zasshu, volatile=True)  # totally freeze netG
            fake = Variable(netG(inputv).data - perceptionRes_cpu)
            errD_fake_vec = netD(fake)
            errD_fake = errD_fake_vec.mean(0).view(1)
            errD_fake.backward(one, retain_variables=True)
            errD = errD_real - errD_fake

            # GP term
            dist = L2_dist(Variable(input).view(batch_size, -1), fake.view(batch_size, -1))
            lip_est = (errD_real_vec - errD_fake_vec).abs() / (dist + 1e-8)
            lip_loss = opt.gpW * ((1.0 - lip_est) ** 2).mean(0).view(1)
            lip_loss.backward(one)
            errD = errD + lip_loss

            optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        if i < len(dataloader):
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of LR pic
            data = data_iter.next()
            real_cpu, fake_cpu, perceptualRes_cpu = data
            i += 1

            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
                fake_cpu = fake_cpu.cuda()
                perceptionRes_cpu = perceptionRes_cpu.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            zasshu.resize_as_(fake_cpu).copy_(fake_cpu)

            inputv = Variable(zasshu)
            fake = netG(inputv)

            if gen_iterations < opt.baseGeni:
                MSEloss = criterion_MSE(fake, Variable(input))
                MSEloss.backward(one)
                errG = MSEloss
            else:
                subb = Variable(torch.stack([suber for ww in range(batch_size)])).cuda()
                divv = Variable(torch.stack([diver for ww in range(batch_size)])).cuda()

                errG = opt.ganW * netD(fake - Variable(perceptionRes_cpu)).mean(0).view(1)
                MSEloss = (vgg_features((fake.mul(0.5) - subb) / divv) - (
                    vgg_features((Variable(input).mul(0.5) - subb) / divv))).pow(2).mean()
                errG -= MSEloss
                errG.backward(mone)

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        if gen_iterations < opt.baseGeni:
            if flag2:
                window1 = viz.line(
                    np.array([errD.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='errD(distance)')
                )
                window2 = viz.line(
                    np.array([MSEloss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='MSE/VGG L2 loss')
                )
                window3 = viz.line(
                    np.array([lip_loss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gradient penalty ' + str(opt.gpW))
                )
                flag2 -= 1
            viz.line(np.array([errD.data[0]]), np.array([gen_iterations]), update='append', win=window1)
            viz.line(np.array([MSEloss.data[0]]), np.array([gen_iterations]), update='append', win=window2)
            viz.line(np.array([lip_loss.data[0]]), np.array([gen_iterations]), update='append', win=window3)

            print('[%d/%d][%d/%d][%d] distance: %f err_D_real: %f err_D_fake %f mse %f GPLoss %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errD_real.data[0], errD_fake.data[0], MSEloss.data[0], lip_loss.data[0]))
        else:
            if flag2:
                window1 = viz.line(
                    np.array([errD.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='errD(distance)')
                )
                window2 = viz.line(
                    np.array([MSEloss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='MSE/VGG L2 loss')
                )
                window3 = viz.line(
                    np.array([lip_loss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gradient penalty ' + str(opt.gpW))
                )
                window4 = viz.line(
                    np.array([-errG.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gnet loss (' + str(opt.ganW) + 'GAN loss + VGG L2)', caption='Gnet loss')
                )
                flag2 -= 1
                flag3 -= 1
            elif flag3:
                window4 = viz.line(
                    np.array([-errG.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gnet loss (' + str(opt.ganW) + 'GAN loss + VGG L2)', caption='Gnet loss')
                )
                flag3 -= 1
            viz.line(np.array([errD.data[0]]), np.array([gen_iterations]), update='append', win=window1)
            viz.line(np.array([MSEloss.data[0]]), np.array([gen_iterations]), update='append', win=window2)
            viz.line(np.array([lip_loss.data[0]]), np.array([gen_iterations]), update='append', win=window3)
            viz.line(np.array([-errG.data[0]]), np.array([gen_iterations]), update='append', win=window4)

            print('[%d/%d][%d/%d][%d] distance: %f err_G: %f err_D_real: %f err_D_fake %f VGGMSE %f GPLoss %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], -errG.data[0], errD_real.data[0], errD_fake.data[0], MSEloss.data[0],
                     lip_loss.data[0]))

        if gen_iterations % 100 == 0:
            fake = netG(Variable(fixed_zasshu, volatile=True)).data.mul(0.5).add(0.5)
            viz.images(
                fake.cpu().numpy(),
                win=imageW
            )
            vutils.save_image(fake,
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

        gen_iterations += 1

    # do checkpointing
    if epoch % cuts == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + opt.epoi))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + opt.epoi))

# TODO: retain_graph
