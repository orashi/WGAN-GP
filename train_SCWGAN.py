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
import styleData as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as M
from torch.autograd import Variable
from PIL import Image
import math
import numpy as np
from visdom import Visdom

# B -> A

parser = argparse.ArgumentParser()
parser.add_argument('--datarootA', required=True, help='path to datasetA')
parser.add_argument('--datarootB', required=True, help='path to datasetB')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cut', type=int, default=1)
parser.add_argument('--flag', type=int, default=0, help='Dnet pretrain flag')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--scale', type=float, default=0.5, help='RES scale')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=100, help='start base of pure mse loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default='main', help='visdom env')

opt = parser.parse_args()
print(opt)

viz = Visdom(env=opt.env)

imageW = viz.images(
    np.random.rand(3, 512, 256),
    opts=dict(title='fakeHR', caption='fakeHR')
)
imageW2 = viz.images(
    np.random.rand(3, 512, 256),
    opts=dict(title='fakeHRRes', caption='fakeHRRes')
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

datasetA = dset.ImageFolder(root=opt.datarootA,
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize, Image.BICUBIC),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                            )

assert datasetA
dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

datasetB = dset.ImageFolder(root=opt.datarootB,
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize, Image.BICUBIC),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                            )

assert datasetB
dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))


class LayerNormalization(nn.Module):  # PR this!!
    def __init__(self, channel_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.hidden_size = channel_size
        self.a2 = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def forward(self, z):
        sizer = z.size()
        z = z.view(sizer[0], -1)

        mu = z.mean(1)
        sigma = z.std(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.view(sizer) * self.a2.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(sizer) + self.b2.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand(sizer)
        return ln_out


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        # self.bn1 = nn.InstanceNorm2d(ngf, affine=True)
        # self.bn1 = LayerNormalization(ngf)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        # self.bn2 = nn.InstanceNorm2d(ngf, affine=True)
        # self.bn2 = LayerNormalization(ngf)
        self.bn2 = nn.BatchNorm2d(ngf)

    def forward(self, x):
        residual = x

        out = self.conv1(F.relu(self.bn1(x), True))
        out = self.conv2(F.relu(self.bn2(out), True)) * 0.3 + residual

        return out


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        #      self.convN = nn.ConvTranspose2d(100, 4, opt.imageSize // 4, 1, 0, bias=False)

        self.conv1 = nn.Conv2d(3, ngf // 2, 7, 2, 3, bias=False)
        # self.bn1 = nn.InstanceNorm2d(ngf // 2, affine=True)
        # self.bn1 = LayerNormalization(ngf // 2)
        self.bn1 = nn.BatchNorm2d(ngf // 2)

        # state size. (ngf) x W x H
        self.conv2 = nn.Conv2d(ngf // 2, ngf, 4, 2, 1, bias=False)

        self.resnet = self._make_layer(16)
        # state size. (ngf) x W x H

        self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H

        self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H

        self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)

    def _make_layer(self, blocks):
        layers = []
        for i in range(0, blocks):
            layers.append(BasicBlock())

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), True)
        out = self.conv2(out)

        out = self.resnet(out)
        ######

        out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
        out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)

        out = F.tanh(self.conv9(out) / 5.)

        return out


netG = _netG()
netG.apply(G_weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


# class LayerNormalization(nn.Module):  # PR this!!
#     def __init__(self, hidden_size, eps=1e-5):
#         super(LayerNormalization, self).__init__()
#
#         self.eps = eps
#         self.hidden_size = hidden_size
#         self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
#         self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
#
#     def forward(self, z):
#         sizer = z.size()
#         z = z.view(sizer[0], -1)
#
#         mu = z.mean(1)
#         sigma = z.std(1)
#         ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
#         ln_out = ln_out * self.a2.expand_as(ln_out) + self.b2.expand_as(ln_out)
#         return ln_out.view(sizer)


############################
# D network
###########################
def D_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)


# class _netD(nn.Module):
#     def __init__(self):
#         super(_netD, self).__init__()
#
#         # self.bn0 = LayerNormalization(3 * 96 * 96)
#         # self.bn0 = nn.BatchNorm2d(3)
#
#         # input is 3 x 256 x 256
#         self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)
#         # state size. (ndf) x 128 x 128
#
#         self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
#         self.bn2 = LayerNormalization(ndf * 2)
#         # state size. (ndf * 2) x 64 x 64
#
#         self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
#         self.bn3 = LayerNormalization(ndf * 4)
#         # state size. (ndf * 4) x 32 x 32
#
#         self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
#         self.bn4 = LayerNormalization(ndf * 8)
#         # state size. (ndf * 8) x 16 x 16
#
#         self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False)
#
#     def forward(self, x):
#         # x = torch.cat((self.bn0(input), per), 1)
#         out = F.leaky_relu(self.conv1(x), 0.2, True)
#         out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2, True)
#         out = F.leaky_relu(self.bn3(self.conv3(out)), 0.2, True)
#         out = F.leaky_relu(self.bn4(self.conv4(out)), 0.2, True)
#         out = self.conv5(out).view(out.size()[0], -1).mean(1)
#         # out = self.dense1(out.view(out.size(0), -1))
#         # out = self.dense2(out)
#
#         # return out
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        # self.bn0 = LayerNormalization(3 * 96 * 96)
        # self.bn0 = nn.BatchNorm2d(3)

        # input is 6 x 96 x 96
        self.conv11 = nn.Conv2d(3, ndf, 3, 1, 1, bias=False)
        # state size. (ndf) x 96 x 96

        self.conv12 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
        self.bn12 = LayerNormalization(ndf)
        # self.bn12 = nn.BatchNorm2d(ndf)
        # state size. (ndf) x 48 x 48

        self.conv21 = nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False)
        self.bn21 = LayerNormalization(ndf * 2)
        # self.bn21 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf * 2) x 48 x 48

        self.conv22 = nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False)
        self.bn22 = LayerNormalization(ndf * 2)
        # self.bn22 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf * 2) x 24 x 24

        self.conv31 = nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False)
        self.bn31 = LayerNormalization(ndf * 4)
        # self.bn31 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf * 4) x 24 x 24

        self.conv32 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.bn32 = LayerNormalization(ndf * 4)
        # self.bn32 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf * 4) x 12 x 12

        self.conv41 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)
        self.bn41 = LayerNormalization(ndf * 8)
        # self.bn41 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf * 8) x 12 x 12

        self.conv42 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False)
        self.bn42 = LayerNormalization(ndf * 8)
        # self.bn42 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf * 8) x 6 x 6

        self.dense1 = nn.Linear(ndf * 8 * 6 * 6, 1)
        # self.dense2 = nn.Linear(1024, 1)

    def forward(self, x):
        # x = torch.cat((self.bn0(input), per), 1)
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
# vgg19 = M.vgg19()
# vgg19.load_state_dict(torch.load('vgg19.pth'))


# class vgg54(nn.Module):
#    def __init__(self):
#       super(vgg54, self).__init__()
#      self.features = nn.Sequential(
#         # stop at conv4
#        *list(vgg19.features.children())[:-7]
#   )

# def forward(self, x):
#    x = self.features(x)
#   return x


# vgg_features = vgg54()
# for p in vgg_features.parameters():
#    p.requires_grad = False  # to avoid computation

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
zasshu = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
fixed_zasshu = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
fixed_PER = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    zasshu, fixed_zasshu, fixed_PER = zasshu.cuda(), fixed_zasshu.cuda(), fixed_PER.cuda()
    criterion_MSE.cuda()
    L2_dist.cuda()
    #   vgg_features.cuda()
    #   suber, diver = suber.cuda(), diver.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))

flag = 1
flag2 = 1
flag3 = 1
flag4 = opt.flag
scaler = opt.scale

for epoch in range(opt.niter):
    data_iterA = iter(dataloaderA)
    data_iterB = iter(dataloaderB)
    i = 0
    while i < len(dataloaderA):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # train the discriminator Diters times
        Diters = opt.Diters

        if gen_iterations < opt.baseGeni:  # MSE stage
            Diters = 0

        j = 0
        while j < Diters and i < len(dataloaderA):
            if flag4:  # MSE stage
                Diters = 100
                flag4 -= 1

            j += 1
            netD.zero_grad()

            dataA = data_iterA.next()
            dataB = data_iterB.next()

            i += 1
            ###############################
            batch_size = dataA.size(0)

            if opt.cuda:
                dataA = dataA.cuda()
                dataB = dataB.cuda()

            input.resize_as_(dataA).copy_(dataA)
            zasshu.resize_as_(dataB).copy_(dataB)

            errD_real_vec = netD(Variable(input))
            errD_real = errD_real_vec.mean(0).view(1)
            errD_real.backward(mone, retain_graph=True)  # backward on score on real

            # train with fake
            inputv = Variable(zasshu, volatile=True)  # totally freeze netG

            fake = Variable(netG(inputv).data)
            errD_fake_vec = netD(fake / scaler + Variable(zasshu))
            errD_fake = errD_fake_vec.mean(0).view(1)
            errD_fake.backward(one, retain_graph=True)
            errD = errD_real - errD_fake

            # GP term
            dist = L2_dist(Variable(input).view(batch_size, -1),
                           fake.div(scaler).add(Variable(zasshu)).view(batch_size, -1))
            lip_est = (errD_real_vec - errD_fake_vec).abs() / (dist + 1e-8)
            lip_loss = opt.gpW * ((1.0 - lip_est) ** 2).mean(0).view(1)
            lip_loss.backward(one)
            errD = errD + lip_loss

            optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        if i < len(dataloaderA):
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of LR pic
            dataA = data_iterA.next()
            dataB = data_iterB.next()
            i += 1

            batch_size = dataA.size(0)

            if opt.cuda:
                dataA = dataA.cuda()
                dataB = dataB.cuda()

            if flag:  # fix samples
                viz.images(
                    dataA.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='manifold_A', caption='target')
                )

                fixed_zasshu.resize_as_(dataB).copy_(dataB)
                vutils.save_image(fixed_zasshu.mul(0.5).add(0.5),
                                  '%s/ori_samples.png' % opt.outf)
                viz.images(
                    fixed_zasshu.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='manifold_B', caption='source')
                )
                flag -= 1

            input.resize_as_(dataA).copy_(dataA)
            zasshu.resize_as_(dataB).copy_(dataB)

            inputv = Variable(zasshu)

            fake = netG(inputv)

            if gen_iterations < opt.baseGeni:
                MSEloss = criterion_MSE(fake / scaler + Variable(zasshu), Variable(zasshu))
                MSEloss.backward(one)
                errG = MSEloss

            else:
                errG = netD(fake / scaler + Variable(dataB)).mean(0).view(1)
                errG.backward(mone)

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################

        if gen_iterations < opt.baseGeni:
            if flag2:
                window2 = viz.line(
                    np.array([MSEloss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='MSE/VGG L2 loss')
                )
                flag2 -= 1
            viz.line(np.array([MSEloss.data[0]]), np.array([gen_iterations]), update='append', win=window2)

            print('[%d/%d][%d/%d][%d] mse %f '
                  % (epoch, opt.niter, i, len(dataloaderA), gen_iterations, MSEloss.data[0]))

        else:

            if flag3:
                window1 = viz.line(
                    np.array([errD.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='errD(distance)')
                )
                window3 = viz.line(
                    np.array([lip_loss.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gradient penalty ' + str(opt.gpW))
                )
                window4 = viz.line(
                    np.array([-errG.data[0]]), np.array([gen_iterations]),
                    opts=dict(title='Gnet loss', caption='Gnet loss')
                )
                flag3 -= 1
            else:
                viz.line(np.array([errD.data[0]]), np.array([gen_iterations]), update='append', win=window1)
                viz.line(np.array([lip_loss.data[0]]), np.array([gen_iterations]), update='append', win=window3)
                viz.line(np.array([-errG.data[0]]), np.array([gen_iterations]), update='append', win=window4)

            print('[%d/%d][%d/%d][%d] distance: %f err_G: %f err_D_real: %f err_D_fake %f GPLoss %f'
                  % (epoch, opt.niter, i, len(dataloaderA), gen_iterations,
                     errD.data[0], -errG.data[0], errD_real.data[0], errD_fake.data[0],
                     lip_loss.data[0]))

        if gen_iterations % 100 == 0:
            fake = netG(Variable(fixed_zasshu, volatile=True)).data
            real = fake / scaler + fixed_zasshu
            viz.images(
                fake.mul(0.5).add(0.5).cpu().numpy(),
                win=imageW2,
                opts=dict(title='fakeHRRes', caption='fakeHRRes')
            )
            viz.images(
                real.mul(0.5).add(0.5).clamp(0, 1).cpu().numpy(),
                win=imageW,
                opts=dict(title='fakeHR', caption='fakeHR')
            )

            vutils.save_image(real.mul(0.5).add(0.5),
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

        gen_iterations += 1

    # do checkpointing
    if epoch % cuts == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + opt.epoi))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + opt.epoi))

        # TODO: retain_graph
