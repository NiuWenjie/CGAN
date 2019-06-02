import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dcgan_network import Discriminator, Generator
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import argparse
import numpy as np
import os
from logger import Logger


parser = argparse.ArgumentParser()
parser.add_argument('--Epoch', type=int, default=200, help='number of epochs of training ')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--channel', type=int, default=3)
parser.add_argument('--img_size', type=int, default=64, help='the size of image')
parser.add_argument('--n_row', type=int, default=10)
parser.add_argument('--latent_dim', type=int, default=100, help='the number of latent dim')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--save_path', type=str, default='Save_cgan')
opt = parser.parse_args()
print(opt)

transform=transforms.Compose([transforms.Resize(opt.img_size),
                              transforms.CenterCrop(opt.img_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_dataset = datasets.ImageFolder(root='../celebA_train', transform= transform)
test_dataset = datasets.ImageFolder(root='../celebA_test', transform= transform)

train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=opt.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size=opt.batch_size, shuffle=False)

img_shape = (opt.channel, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available else False


class trainer_gan(nn.Module):
    def __init__(self):
        super(trainer_gan, self).__init__()

        self.G = Generator(opt.channel)
        self.D = Discriminator(opt.channel)
        self.loss = nn.BCELoss()
        self.Optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt.lr, betas=(0.5,  0.999))
        self.Optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(0.5,  0.999))
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        if cuda:
            self.G.cuda()
            self.D.cuda()
            self.loss.cuda()


    def train(self):
        os.makedirs(opt.save_path, exist_ok=True)
        for epoch in range(opt.Epoch):
            for i, (img, label) in enumerate(train_dataloader):
                real = Variable(self.FloatTensor(img.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.FloatTensor(img.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)
                real_img = Variable(img.type(self.FloatTensor))

                self.Optimizer_G.zero_grad()
                z = torch.randn(img.shape[0], 100, 1, 1)
                z = Variable(z.type(self.FloatTensor))
                gen_imgs = self.G(z)
                g_loss = self.loss(self.D(gen_imgs), real)
                g_loss.backward()
                self.Optimizer_G.step()

                self.Optimizer_D.zero_grad()
                real_loss = self.loss(self.D(real_img), real)
                fake_loss = self.loss(self.D(gen_imgs.detach()), fake)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.Optimizer_D.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.Epoch, i, len(train_dataloader),
                                                                    d_loss.item(), g_loss.item()))
                gen_imgs = gen_imgs.view(img.shape[0], *img_shape)
                batches_done = epoch * len(train_dataloader) + i
                if batches_done % 400 == 0:
                    save_image(real_img.data, '%s/real_%d.png' % (opt.save_path, batches_done), normalize=True)
                    save_image(gen_imgs.data, '%s/%d.png' % (opt.save_path, batches_done), normalize=True)
            save_image(gen_imgs.data, '%s/%d.png' % (opt.save_path, epoch), normalize=True)
            save_image(real_img.data, '%s/real_%d.png' % (opt.save_path, epoch), normalize=True)
            if (epoch % 199 == 0) or (epoch % 10 == 0):
                torch.save(self.G.state_dict(), '%s/generator_%03d.pkl' % (opt.save_path, epoch))

    def test(self):
        self.G.eval()
        os.makedirs('%s/test' %opt.save_path, exist_ok=True)
        save_path = os.path.join('%s' % (opt.save_path), 'test')
        self.G.load_state_dict(torch.load('%s/generator_090.pkl' %(opt.save_path)))
        for i, (imgs, labels) in enumerate(test_dataloader):
            real_imgs = Variable(imgs.type(self.FloatTensor))
            z = torch.randn(imgs.shape[0], 100, 1, 1)
            z = Variable(z.type(self.FloatTensor))
            gen_imgs = self.G(z)

            fake = gen_imgs.view(gen_imgs.size(0), opt.channel, opt.img_size, opt.img_size)
            real = real_imgs.view(real_imgs.size(0), opt.channel, opt.img_size, opt.img_size)
            print("[Batch %d/%d]" % (i, len(test_dataloader)))


            save_image(real.data, '%s/test/real_%03d.png' % (opt.save_path, i), nrow=10, normalize=True)
            save_image(fake.data, '%s/fake_%03d.png' % (save_path, i), nrow=10, normalize=True)



trainer = trainer_gan()
if cuda:
    trainer.cuda()

#############
# train
############
trainer.train()

#############
# test
############
# trainer.test()


