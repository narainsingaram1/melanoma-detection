import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=7, embed_size=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.embed = nn.Embedding(num_classes, embed_size)
        self.project = nn.Linear(z_dim + embed_size, 512 * 7 * 7)
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        """
        Forward pass for generator.
        z: (batch, z_dim)
        labels: (batch)
        """
        embedding = self.embed(labels)
        x = torch.cat([z, embedding], dim=1)
        x = self.project(x)
        x = x.view(-1, 512, 7, 7)
        return self.net(x)

    def weights_init(self):
        """Initialize weights with normal_(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, num_classes=7, img_size=224):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        """
        Forward pass for discriminator.
        img: (batch, 3, 224, 224)
        labels: (batch)
        """
        embedding = self.embed(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, embedding], dim=1)
        x = self.net(x)
        x = self.flatten(x)
        return self.fc(x)

    def weights_init(self):
        """Initialize weights with normal_(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
