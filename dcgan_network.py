class Generator(nn.Module):
    def __init__(self, channels):
        super(dcgan_Generator, self).__init__()
        self.model = nn.Sequential(
            # input z=100, output [1024, 4, 4], kernal_size (4,4), stride=2
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # input [1024, 4, 4], output [512, 8, 8]
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # input [512, 8, 8] output [256, 16, 16]
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # input [256, 16, 16] output [128, 32, 32]
            nn.ConvTranspose2d(256, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # input [128, 32, 32] output [opt.channel, opt.img_size, opt.img_size]  
            nn.ConvTranspose2d(128, channels, 4, 2, 1,bias=False),
            nn.Tanh()
        )
    def forward(self, noise):
        return self.model(noise)

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(dcgan_Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input [1, 64, 64] output [128, 32, 32]
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # input [128, 32, 32] output [256, 16, 16]
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # input [256, 16, 16] output [512, 8, 8]
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # input [512, 8, 8] output [1024, 4, 4]
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            # input [1024, 4, 4] output [1,1,1                                              ]
            nn.Conv2d(1024, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img)
