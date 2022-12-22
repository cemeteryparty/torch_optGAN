from torch.autograd import grad as Gradient
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as T
from torchvision import datasets
import torch
import os

# https://stackoverflow.com/questions/71263622/sslcertverificationerror-when-downloading-pytorch-datasets-via-torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_term = 10

def load_data(train=True):
    if not os.path.isdir("data"):
        os.mkdir("data")
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10("data/CIFAR10", download=True, transform=trans, train=train)
    return dataset

def calc_gp(x_real, x_fake, D): # calculate_gradient_penalty(real_images, fake_images)
    batch_size = x_real.size(0)
    if x_fake.size(0) != batch_size:
        raise ValueError("brain doko")
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(*list(x_real.size())) # eta.expand(batch_size, x_real.size(1), x_real.size(2), x_real.size(3))
    eta = eta.to(device)

    inter = eta * x_real + ((1 - eta) * x_fake) # interpolated
    inter = inter.to(device)
    inter = Variable(inter, requires_grad=True)

    inter_p = D(inter) # probability of interpolated samples

    y_pos = torch.ones(inter_p.size())
    y_pos = y_pos.to(device)
    grads = Gradient(
        outputs=inter_p, inputs=inter, grad_outputs=y_pos, create_graph=True, retain_graph=True
    )[0]
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

def vis_latent(G, fname=None):
    G.eval()
    noises = torch.randn(64, 100).to(device)
    gen_images = G(noises)
    gen_images = gen_images.mul(0.5).add(0.5)
    gen_images = gen_images.detach().cpu().numpy()
    gen_images = gen_images.transpose((0, 2, 3, 1))

    fig = plt.figure(figsize=(10, 10))
    for i, img in enumerate(gen_images):
        plt.subplot(8, 8, i + 1)
        plt.imshow(img)
    if fname is not None:
        plt.savefig(fname)
    plt.close("all")
