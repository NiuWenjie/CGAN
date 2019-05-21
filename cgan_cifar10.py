import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

parser = argparse.ArgumentParser()
parser.add_argument('--Epoch', type=int, default=200, help='number of epochs of training ')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--img_size', type=int, default=32, help='the size of image')
parser.add_argument('--channel', type=int, default=3)
parser.add_argument('--n_classes', type=int, default=10, help='the number of classes')
parser.add_argument('--latent_dim', type=int, default=100, help='the number of latent dim')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--save_path', type=str, default='Save_cgan')
parser.add_argument('--data_dir', type=str, default='../data/mnist')
opt = parser.parse_args()
print(opt)

train_dataloader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/cifar10',train=True,
                                                              transform=transforms.Compose([transforms.Resize(opt.img_size),
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                                                              download=True),
                                               batch_size=opt.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/cifar10', train=False,
                                                             transform=transforms.Compose([transforms.Resize(opt.img_size),
                                                                                           transforms.ToTensor(),
                                                                                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                                                             download=False),
                                              batch_size=opt.batch_size, shuffle=False)
img_shape = (opt.channel, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim+opt.n_classes, 128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        input = torch.cat((self.emb(labels), noise), -1)
        return self.model(input).view(input.shape[0], *img_shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(
            nn.Linear(opt.n_classes+int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        output = torch.cat((img.view(img.size(0), -1), self.emb(labels)), -1)
        return self.model(output)


G = Generator()
D = Discriminator()
loss = nn.MSELoss()
Optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr)
Optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr)
cuda = True if torch.cuda.is_available else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
G.cuda()
D.cuda()
loss.cuda()

if not os._exists(opt.save_path):
    os.mkdir(opt.save_path)
# ##############
# # training
# #############
for epoch in range(opt.Epoch):
    for i, (img, label) in enumerate(train_dataloader):
        real = torch.ones(img.shape[0], 1)
        real = Variable(real.type(FloatTensor))
        fake = torch.zeros(img.shape[0], 1)
        fake = Variable(fake.type(FloatTensor))

        real_img = Variable(img.type(FloatTensor))
        real_label = Variable(label.type(LongTensor))

        Optimizer_G.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (img.shape[0], opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, img.shape[0])))
        gen_imgs = G(z, gen_labels)
        g_loss = loss(D(gen_imgs, gen_labels), real)
        g_loss.backward()
        Optimizer_G.step()

        Optimizer_D.zero_grad()
        real_loss = loss(D(real_img, real_label), real)
        fake_loss = loss(D(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) /2
        d_loss.backward()
        Optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.Epoch, i, len(train_dataloader),
                                                            d_loss.item(), g_loss.item()))
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % 400 == 0:
            save_image(gen_imgs.data[:25], '%s/%d.png' % (opt.save_path, batches_done), nrow=5, normalize=True)
    if epoch % 199 == 0 or epoch % 10 == 0:
        torch.save(G.state_dict(), '%s/generator_%03d.pkl' % (opt.save_path, epoch))
    writer.add_scalar('Train/G_Loss', g_loss.data[0], epoch)
    writer.add_scalar('Train/D_Loss', d_loss.data[0], epoch)


#############
# testing
#############
G.eval()
if not os._exists('./%s/test' %(opt.save_path)):
    os.mkdir('./%s/test' %(opt.save_path))
save_path = os.path.join('./%s' %(opt.save_path), 'test')
G.load_state_dict(torch.load('./%s/generator_199.pkl' %(opt.save_path)))
for i, (imgs, labels) in enumerate(test_dataloader):
    real_imgs = Variable(imgs.type(FloatTensor))
    z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    labels = Variable(labels.type(LongTensor))
    gen_imgs = G(z, labels)


    fake = gen_imgs.view(gen_imgs.size(0), opt.channel, opt.img_size, opt.img_size)
    real = real_imgs.view(real_imgs.size(0), opt.channel, opt.img_size, opt.img_size)
    print("[Batch %d/%d]" % (i, len(test_dataloader)))

    save_image(real.data, '%s/test/real_%03d.png' % (opt.save_path, i), nrow=1, normalize=True)
    save_image(fake.data, '%s/test/fake_%03d.png' % (opt.save_path, i), nrow=1, normalize=True)

