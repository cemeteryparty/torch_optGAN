from utils.general import vis_latent
from utils.general import load_data
from utils.general import calc_gp
from utils.models import Discriminator
from utils.models import Generator
from utils.optimizers import optimAdam
from utils.optimizers import SOMD

from torch import from_numpy as np2TT
from datetime import datetime
from torchinfo import summary
from scipy.io import savemat
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import json
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class conf:
    optimizer = "Adam"
    epochs, batch_size = 1000, 128
    dis_ratio = 50
    saved = "saved/{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    g_lr, d_lr = 1e-3, 1e-3#5e-2, 1e-2
    ckpt = None #"saved/20221220_131029_adam/checkpoints/hook_ep2000.pth"
    def __init__(self):
        self.d = dict(optimizer=self.optimizer)
        self.d["epochs"] = self.epochs
        self.d["batch_size"] = self.batch_size
        self.d["dis_ratio"] = self.dis_ratio
        self.d["saved"] = self.saved
        self.d["g_lr"] = self.g_lr
        self.d["d_lr"] = self.d_lr
        self.d["ckpt"] = self.ckpt
args = conf()

# main
if __name__ == '__main__':
    os.makedirs(args.saved, exist_ok=True)
    os.makedirs(os.path.join(args.saved, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.saved, "checkpoints"), exist_ok=True)
    fd = open(os.path.join(args.saved, "conf.json"), "w")
    json.dump(args.d, fd, indent=4)
    fd.close()

    trainset = load_data()  # load cifar10
    tra_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=True
    )
    G = Generator().to(device)
    D = Discriminator().to(device)
    if args.ckpt is not None:
        state_path = os.path.join(args.ckpt)
        state = torch.load(state_path, map_location="cpu")
        G.load_state_dict(state["gen"])
        D.load_state_dict(state["dis"])
    # _ = input("(pause)")
    # exit(0)

    if args.optimizer == "Adam":
        optG = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
        optD = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
    elif args.optimizer == "SOMD_v1":
        optG = SOMD(G.parameters(), lr=args.g_lr)
        optD = SOMD(D.parameters(), lr=args.d_lr)
    elif args.optimizer == "SOMD_v2":
        optG = SOMD(G.parameters(), lr=args.g_lr, version=2)
        optD = SOMD(D.parameters(), lr=args.d_lr, version=2)
    elif args.optimizer == "SOMD_v3":
        optG = SOMD(G.parameters(), lr=args.g_lr, version=3)
        optD = SOMD(D.parameters(), lr=args.d_lr, version=3)
    elif args.optimizer == "optimAdam":
        optG = optimAdam(G.parameters(), lr=args.g_lr)
        optD = optimAdam(D.parameters(), lr=args.d_lr)
    else:
        raise ValueError("brain doko")

    log = ""
    loss_curve = dict(
        epochs=np.arange(args.epochs, dtype=int),
        loss_G=np.zeros((args.epochs, ), dtype=np.float32),
        loss_D=np.zeros((args.epochs, ), dtype=np.float32),
        loss_real=np.zeros((args.epochs, ), dtype=np.float32),
        loss_fake=np.zeros((args.epochs, ), dtype=np.float32),
        gp_term=np.zeros((args.epochs, ), dtype=np.float32)
    )

    y_pos = torch.tensor(1, dtype=torch.float32) # one
    y_neg = y_pos * -1 # mone
    y_pos, y_neg = y_pos.to(device), y_neg.to(device)
    # train loop
    for epoch in range(args.epochs):
        G.train()
        D.train()
        ep_time0 = time.time()

        for p in D.parameters():
            p.requires_grad = True
        for i, (x_batch, _) in enumerate(tra_loader):
            if i == args.dis_ratio:
               break
            D.zero_grad()
            x_batch = x_batch.to(device) # images
            noises = torch.randn(x_batch.size(0), 100).to(device)
            # real loss
            loss_D_real = D(x_batch) # d_loss_real
            loss_D_real = loss_D_real.mean()
            loss_D_real.backward(y_neg)
            # fake loss
            xf_batch = G(noises) # fake_images
            loss_D_fake = D(xf_batch) # d_loss_fake
            loss_D_fake = loss_D_fake.mean()
            loss_D_fake.backward(y_pos)
            # gradient penalty
            gp_term = calc_gp(x_batch.detach(), xf_batch.detach(), D) # gradient_penalty
            gp_term.mul_(10)  # * lambda
            gp_term.backward()
            # overall loss
            loss_D = loss_D_fake - loss_D_real + gp_term
            optD.step()
            # stdout
            print("\r{}".format(" " * len(log)), end="")  # flush outputs
            log = "{}_{}/{} - loss_real: {:.4f}, loss_fake: {:.4f}, gp: {:.4f}, loss_D: {:.4f}".format(
                epoch, i + 1, min(args.dis_ratio, len(tra_loader)), loss_D_real.item(), loss_D_fake.item(), gp_term.item(), loss_D.item()
            )
            print("\r{}".format(log), end="")
        for p in D.parameters():
            p.requires_grad = False
        G.zero_grad()
        noises = torch.randn(args.batch_size, 100).to(device)
        xf_batch = G(noises) # fake_images
        loss_G = D(xf_batch) # g_loss
        loss_G = loss_G.mean()
        loss_G.backward(y_neg)
        optG.step()

        loss_curve["loss_G"][epoch] = loss_G.item()
        loss_curve["loss_D"][epoch] = loss_D.item()
        loss_curve["loss_real"][epoch] = loss_D_real.item()
        loss_curve["loss_fake"][epoch] = loss_D_fake.item()
        loss_curve["gp_term"][epoch] = gp_term.item()
        print("\r{}".format(" " * len(log)), end="")  # flush outputs
        log = "Epoch {}/{} - {:.4f}, loss_G: {:.4f}, loss_D: {:.4f}, loss_real: {:.4f}, loss_fake: {:.4f}".format(
            epoch + 1, args.epochs, time.time() - ep_time0, loss_G.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()
        )
        print("\r{}".format(log))

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            vis_latent(G, fname=os.path.join(args.saved, f"images/{epoch + 1}.png"))
        if epoch % 100 == 0 or epoch + 1 == args.epochs:    
            state = dict(
                epoch=epoch + 1, loss_G=loss_G.item(), loss_D=loss_D.item(),
                gen=G.state_dict(), dis=D.state_dict()
            )
            torch.save(
                state, os.path.join(args.saved, f"checkpoints/hook_ep{epoch + 1}.pth")
            )
    # end train loop
    savemat(
        os.path.join(args.saved, "{}.mat".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )), loss_curve
    )    
