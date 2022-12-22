import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, seq_len=100):
        super(Generator, self).__init__()
        """
        conv2d : (x - kernelSz + 2 * pad) / stride + 1
        conv2dT: (x - 1) * stride - 2 * pad + kernelSz
        change kernel shape to match cifar
        """
        self.fc1 = nn.Sequential(
            nn.Linear(seq_len, 1024, bias=True),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128 * 8 * 8, bias=True),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.conv_t1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.conv_t2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=(5, 5), padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        x = x.view(-1, 128, 8, 8)  # reshape

        x = self.conv_t1(x)
        x = self.conv1(x)
        x = self.conv_t2(x)
        x = self.conv2(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, init_w=True):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), padding=2),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=2),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5, 5), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024, bias=True),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.fc2 = nn.Linear(1024, 1, bias=True)
        if init_w:
            self.weights_init(self.conv2[0], method="he_normal")
            self.weights_init(self.conv3[0], method="he_normal")
            self.weights_init(self.fc1[0], method="he_normal")
            self.weights_init(self.fc2, method="he_normal")

    def weights_init(self, lyr, method="he_normal", bias0=True):
        if method == "he_normal":
            nn.init.kaiming_normal_(lyr.weight.data)
        elif method == "glorot_uniform":
            nn.init.xavier_uniform_(lyr.weight.data)
        if bias0:
            lyr.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
