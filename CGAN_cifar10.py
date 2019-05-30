import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
from network import Generator, Discriminator
import argparse
import numpy as np
import os
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--Epoch', type=int, default=200, help='number of epochs of training ')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--img_size', type=int, default=32, help='the size of image')
parser.add_argument('--channel', type=int, default=3)
parser.add_argument('--n_classes', type=int, default=10, help='the number of classes')
parser.add_argument('--n_row', type=int, default=10)
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
cuda = True if torch.cuda.is_available else False


class trainer_gan(nn.Module):
    def __init__(self):
        super(trainer_gan, self).__init__()

        self.G = Generator(opt.n_classes, opt.latent_dim, int(np.prod(img_shape)))
        self.D = Discriminator(opt.n_classes, opt.latent_dim, int(np.prod(img_shape)))
        self.loss = nn.MSELoss()
        self.Optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt.lr)
        self.Optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt.lr)
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
        self.logger = Logger('./logs')
        if cuda:
            self.G.cuda()
            self.D.cuda()
            self.loss.cuda()


    def train(self):
        os.makedirs(opt.save_path, exist_ok=True)
        for epoch in range(opt.Epoch):
            for i, (img, label) in enumerate(train_dataloader):
                real = torch.ones(img.shape[0], 1)
                real = Variable(real.type(self.FloatTensor))
                fake = torch.zeros(img.shape[0], 1)
                fake = Variable(fake.type(self.FloatTensor))

                real_img = Variable(img.type(self.FloatTensor))
                real_label = Variable(label.type(self.LongTensor))

                self.Optimizer_G.zero_grad()
                z = Variable(self.FloatTensor(np.random.normal(0, 1, (img.shape[0], opt.latent_dim))))
                gen_labels = Variable(self.LongTensor(np.random.randint(0, opt.n_classes, img.shape[0])))
                gen_imgs = self.G(z, gen_labels)
                g_loss = self.loss(self.D(gen_imgs, gen_labels), real)
                g_loss.backward()
                self.Optimizer_G.step()

                self.Optimizer_D.zero_grad()
                real_loss = self.loss(self.D(real_img, real_label), real)
                fake_loss = self.loss(self.D(gen_imgs.detach(), gen_labels), fake)
                d_loss = (real_loss + fake_loss) /2
                d_loss.backward()
                self.Optimizer_D.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.Epoch, i, len(train_dataloader),
                                                                    d_loss.item(), g_loss.item()))
                gen_imgs = gen_imgs.view(img.shape[0], *img_shape)
                batches_done = epoch * len(train_dataloader) + i
                if batches_done % 400 == 0:
                    save_image(gen_imgs.data[:25], '%s/%d.png' % (opt.save_path, batches_done), nrow=5, normalize=True)
            if (epoch % 199 == 0) or (epoch % 10 == 0):
                torch.save(self.G.state_dict(), '%s/generator_%03d.pkl' % (opt.save_path, epoch))
                # 1. Log scalar values (scalar summary)
                info = {'g_loss': g_loss.item(), 'd_loss': d_loss.item()}

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, epoch + 1)

                # 3. Log training images (image summary)
                info = {'fake_images': gen_imgs.view(img.shape[0],3, 32, 32)[:10].cpu().detach().numpy(),
                        'real_images': img.view(img.shape[0],3, 32, 32)[:10].cpu().numpy()}

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, epoch + 1)

    def test(self):
        self.G.eval()
        save_path = os.path.join('./%s' %(opt.save_path), 'test')
        self.G.load_state_dict(torch.load('./%s/generator_199.pkl' %(opt.save_path)))
        for i, (imgs, labels) in enumerate(test_dataloader):
            real_imgs = Variable(imgs.type(self.FloatTensor))
            z = Variable(self.FloatTensor(np.random.normal(0, 1, (opt.n_row ** 2, opt.latent_dim))))
            labels = np.array([num for _ in range(opt.n_row) for num in range(opt.n_row)])
            labels = Variable(self.LongTensor(labels))
            gen_imgs = self.G(z, labels)


            fake = gen_imgs.view(gen_imgs.size(0), opt.channel, opt.img_size, opt.img_size)
            real = real_imgs.view(real_imgs.size(0), opt.channel, opt.img_size, opt.img_size)
            print("[Batch %d/%d]" % (i, len(test_dataloader)))

            save_image(real.data, '%s/real_%03d.png' % (save_path, i), nrow=10, normalize=True)
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
trainer.test()
