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
parser.add_argument('--Diters', type=int, default=4, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=12500, help='start base of pure mse loss')
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

        self.conv6 = nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x W x H

        self.conv7 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x W x H
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # ngf x 2W x 2H

        self.conv8 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)  # ngf*4 x 2W x 2H
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # ngf x 4W x 4H

        self.conv9 = nn.Conv2d(ngf, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)

        #####  0.3??????
        residual = out

        # noise = torch.FloatTensor(out.size()[0], 4, out.size()[2], out.size()[3])
        # noise = Variable(noise)
        # if opt.cuda:
        #    noise = noise.cuda()
        # noise.data.normal_(0, 1)

        #        noise = self.convN(noise)
        out = self.convB11(F.relu(self.bnB11(out), True))
        out = self.convB12(F.relu(self.bnB12(out), True)) + residual

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

        out = self.conv6(F.relu(self.bn6(out), True))
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


criterion_MSE = nn.MSELoss()
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
one = torch.FloatTensor([1])
mone = one * -1
# suber = torch.FloatTensor(
#    [np.full((96, 96), 0.485 - 0.5), np.full((96, 96), 0.456 - 0.5), np.full((96, 96), 0.406 - 0.5)])
# diver = torch.FloatTensor([np.full((96, 96), 0.229), np.full((96, 96), 0.224), np.full((96, 96), 0.225)])

if opt.cuda:
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    zasshu, fixed_zasshu = zasshu.cuda(), fixed_zasshu.cuda()
    criterion_MSE.cuda()
    #   vgg_features.cuda()
    #   suber, diver = suber.cuda(), diver.cuda()

# setup optimizer
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
        # train the discriminator Diters times
        Diters = opt.Diters
        if gen_iterations < opt.baseGeni:  # MSE stage
            Diters = 0

        j = 0


        ############################
        # (2) Update G network
        ############################
        if i < len(dataloader):
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of LR pic
            data = data_iter.next()
            real_cpu, fake_cpu, _ = data
            i += 1

            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
                fake_cpu = fake_cpu.cuda()

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


            input.resize_as_(real_cpu).copy_(real_cpu)
            zasshu.resize_as_(fake_cpu).copy_(fake_cpu)

            inputv = Variable(zasshu)

            fake = netG(inputv)

            if gen_iterations < opt.baseGeni:
                MSEloss = criterion_MSE(fake, Variable(input))
                MSEloss.backward(one)
                errG = MSEloss

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
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations, MSEloss.data[0]))

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

# TODO: retain_graph
