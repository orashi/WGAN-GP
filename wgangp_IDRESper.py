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
from tensorboardX import SummaryWriter
import torchvision.models as M
from torch.autograd import Variable
from PIL import Image
import math
import numpy as np

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cut', type=int, default=1)
parser.add_argument('--flag', type=int, default=0, help='Dnet pretrain flag')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--scale', type=float, default=1.5, help='RES scale')

parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')
# parser.add_argument('--ganW', type=float, default=0.01, help='gan penalty weight')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netM', default='', required=True, help="path to netM (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure mse loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default=None, help='tensorboard env')
parser.add_argument('--optim', action='store_true', help='load optimizer\'s checkpoint')


opt = parser.parse_args()
print(opt)

writer = SummaryWriter(log_dir=opt.env, comment='this is great')

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

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


############################
# G network
###########################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(F.relu(x, True))
        out = self.conv2(F.relu(out, True)) * 0.3 + residual

        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf)

    def forward(self, x):
        residual = x

        out = self.conv1(F.relu(self.bn1(x), True))
        out = self.conv2(F.relu(self.bn2(out), True)) * 0.3 + residual

        return out


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
        # state size. (ngf) x W x H

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
        out = self.conv1(x)

        out = self.resnet(out)
        ######

        out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
        out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)

        out = F.tanh(self.conv9(out) / 5.)

        return out


netG = _netG()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        # self.bn0 = LayerNormalization(3 * 96 * 96)
        # self.bn0 = nn.BatchNorm2d(3)

        # input is 6 x 96 x 96
        self.conv11 = nn.Conv2d(6, ndf, 3, 1, 1, bias=False)
        # state size. (ndf) x 96 x 96

        self.conv12 = nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)
        # self.bn12 = nn.BatchNorm2d(ndf)
        # state size. (ndf) x 48 x 48

        self.conv21 = nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False)
        # self.bn21 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf * 2) x 48 x 48

        self.conv22 = nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False)
        # self.bn22 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf * 2) x 24 x 24

        self.conv31 = nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False)
        # self.bn31 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf * 4) x 24 x 24

        self.conv32 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        # self.bn32 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf * 4) x 12 x 12

        self.conv41 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)
        # self.bn41 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf * 8) x 12 x 12

        self.conv42 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False)
        # self.bn42 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf * 8) x 6 x 6

        self.dense1 = nn.Linear(ndf * 8 * 6 * 6, 1)
        # self.dense2 = nn.Linear(1024, 1)

    def forward(self, x):
        # x = torch.cat((self.bn0(input), per), 1)
        out = F.leaky_relu(self.conv11(x), 0.2, True)
        out = F.leaky_relu(self.conv12(out), 0.2, True)
        out = F.leaky_relu(self.conv21(out), 0.2, True)
        out = F.leaky_relu(self.conv22(out), 0.2, True)
        out = F.leaky_relu(self.conv31(out), 0.2, True)
        out = F.leaky_relu(self.conv32(out), 0.2, True)
        out = F.leaky_relu(self.conv41(out), 0.2, True)
        out = F.leaky_relu(self.conv42(out), 0.2, True)

        out = self.dense1(out.view(out.size(0), -1))
        # out = self.dense2(out)

        return out


netD = _netD()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


############################
# MSE network
###########################
class _netM(nn.Module):
    def __init__(self):
        super(_netM, self).__init__()
        #      self.convN = nn.ConvTranspose2d(100, 4, opt.imageSize // 4, 1, 0, bias=False)

        self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
        # state size. (ngf) x W x H

        self.resnet = self._make_layer(3)
        # state size. (ngf) x W x H

        self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H

        self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H

        self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)

    def _make_layer(self, blocks):
        layers = []
        for i in range(0, blocks):
            layers.append(BasicBlock2())

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.resnet(out)
        ######

        out = F.relu(self.pixel_shuffle1(self.conv7(out)), True)
        out = F.relu(self.pixel_shuffle2(self.conv8(out)), True)

        out = F.tanh(self.conv9(out) / 5.)

        return out


netM = _netM()
netM.load_state_dict(torch.load(opt.netM))
print(netM)

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
# suber = torch.FloatTensor(
#    [np.full((96, 96), 0.485 - 0.5), np.full((96, 96), 0.456 - 0.5), np.full((96, 96), 0.406 - 0.5)])
# diver = torch.FloatTensor([np.full((96, 96), 0.229), np.full((96, 96), 0.224), np.full((96, 96), 0.225)])

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netM.cuda()
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
if opt.optim:
    optimizerG.load_state_dict(torch.load('%s/optimG_checkpoint.pth' % opt.outf))
    optimizerD.load_state_dict(torch.load('%s/optimD_checkpoint.pth' % opt.outf))

flag = 1
flag4 = opt.flag
scaler = opt.scale

netM.eval()

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

        if flag4:  # pretrain stage
            Diters = 4000
            flag4 -= 1

        if gen_iterations < opt.baseGeni:  # MSE stage
            Diters = 0

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
                perceptualRes_cpu = perceptualRes_cpu.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            zasshu.resize_as_(fake_cpu).copy_(fake_cpu)

            # print(input.mean(), input.std(), real_cpu.mean(), real_cpu.std())
            # plt.hist(np.array(input.cpu().numpy()).resize(96*96*3))
            # plt.show()
            # viz.histogram(input.cpu().view(-1))

            inputv = Variable(zasshu, volatile=True)  # totally freeze netG
            mse = netM(inputv).data / scaler + perceptualRes_cpu

            # inputv = Variable(torch.cat((input * 10, perceptualRes_cpu), 1))


            errD_real_vec = netD(Variable(torch.cat((input, mse), 1)))
            errD_real = errD_real_vec.mean(0).view(1)
            errD_real.backward(mone, retain_graph=True)  # backward on score on real

            # train with fake

            fake = Variable(netG(inputv).data)
            errD_fake_vec = netD(
                torch.cat((fake / scaler + Variable(mse), Variable(mse)), 1))
            errD_fake = errD_fake_vec.mean(0).view(1)
            errD_fake.backward(one, retain_graph=True)
            errD = errD_real - errD_fake

            # GP term
            dist = L2_dist(Variable(input).view(batch_size, -1),
                           fake.div(scaler).add(Variable(mse)).view(batch_size, -1))
            lip_est = (errD_real_vec - errD_fake_vec).abs() / (dist + 1e-8)
            lip_loss = opt.gpW * ((1.0 - lip_est) ** 2).mean(0).view(1)
            lip_loss.backward(one)
            errD = errD + lip_loss
            # errG = errD

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
                perceptualRes_cpu = perceptualRes_cpu.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            zasshu.resize_as_(fake_cpu).copy_(fake_cpu)

            inputv = Variable(zasshu)
            mse = netM(inputv).data / scaler + perceptualRes_cpu

            if flag:  # fix samples
                writer.add_image('real_cpu imgs', vutils.make_grid(real_cpu.mul(0.5).add(0.5), nrow=16))
                writer.add_image('LRMSE imgs', vutils.make_grid(fixed_PER.mul(0.5).add(0.5).clamp(0, 1), nrow=16))
                writer.add_image('HRRes imgs', vutils.make_grid((real_cpu - fixed_PER).mul(0.5 * scaler).add(0.5).clamp(0, 1), nrow=16))
                vutils.save_image(real_cpu.mul(0.5).add(0.5),
                                  '%s/real_samples.png' % opt.outf)
                vutils.save_image(fixed_zasshu.mul(0.5).add(0.5),
                                  '%s/ori_samples.png' % opt.outf)
                fixed_zasshu.resize_as_(fake_cpu).copy_(fake_cpu)
                fixed_PER.resize_as_(real_cpu).copy_(mse)

                flag -= 1

            fake = netG(inputv)

            if gen_iterations < opt.baseGeni:
                MSEloss = criterion_MSE(fake / scaler + Variable(perceptualRes_cpu), Variable(input))
                MSEloss.backward(one)
                errG = MSEloss
            else:
                # subb = Variable(torch.stack([suber for ww in range(batch_size)])).cuda()
                # divv = Variable(torch.stack([diver for ww in range(batch_size)])).cuda()

                errG = netD(
                    torch.cat((fake / scaler + Variable(mse), Variable(mse)), 1)).mean(
                    0).view(1)
                # MSEloss = 0  # (vgg_features((fake.mul(0.5) - subb) / divv) - (
                # vgg_features((Variable(input).mul(0.5) - subb) / divv))).pow(2).mean()
                # errG -= MSEloss
                errG.backward(mone)

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        if gen_iterations < opt.baseGeni:
            writer.add_scalar('MSE/VGG L2 loss', MSEloss.data[0], gen_iterations)
            print('[%d/%d][%d/%d][%d] mse %f '
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations, MSEloss.data[0]))
        else:
            writer.add_scalar('MSE/VGG L2 loss', MSEloss.data[0], gen_iterations)
            writer.add_scalar('errD(wasserstein distance)', errD.data[0], gen_iterations)
            writer.add_scalar('errD_real', errD_real.data[0], gen_iterations)
            writer.add_scalar('errD_fake', errD_fake.data[0], gen_iterations)
            writer.add_scalar('Gnet loss toward real', -errG.data[0], gen_iterations)
            writer.add_scalar('gradient_penalty'+ str(opt.gpW), lip_loss.data[0], gen_iterations)

            print('[%d/%d][%d/%d][%d] distance: %f err_G: %f err_D_real: %f err_D_fake %f GPLoss %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], -errG.data[0], errD_real.data[0], errD_fake.data[0],
                     lip_loss.data[0]))

        if gen_iterations % 500 == 0:
            fake = netG(Variable(fixed_zasshu, volatile=True)).data
            real = fake / scaler + fixed_PER
            writer.add_image('fakeHRRes', vutils.make_grid(fake.mul(0.5).add(0.5), nrow=16),
                             gen_iterations)
            writer.add_image('fakeHR', vutils.make_grid(real.mul(0.5).add(0.5).clamp(0, 1), nrow=16),
                             gen_iterations)
            vutils.save_image(real.mul(0.5).add(0.5),
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

        gen_iterations += 1

    # do checkpointing
    if opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_only.pth' % opt.outf)
        torch.save(netD.state_dict(), '%s/netD_epoch_only.pth' % opt.outf)
    elif epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(optimizerG.state_dict(), '%s/optimG_checkpoint.pth' % opt.outf)
    torch.save(optimizerD.state_dict(), '%s/optimD_checkpoint.pth' % opt.outf)
