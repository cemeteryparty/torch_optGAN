from utils.models import Generator
from utils.general import load_data

from torchvision.models.inception import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from torch import from_numpy as np2TT
from matplotlib import pyplot as plt
import torchvision.transforms as T
from scipy.stats import entropy
from scipy.io import loadmat
import torch.nn as nn
import numpy as np
import torch
import json
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_inception_score(data_loader, N=1):
    # np.seterr(all='raise')
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=Inception_V3_Weights.IMAGENET1K_V1)
    model.eval()
    
    upbiln = nn.Upsample(size=(299, 299), mode="bilinear")
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    shave = nn.Softmax(dim=0)
    
    scores = []
    for i, (batch, _) in enumerate(data_loader):
        print("\r{}/{} [{}{}]".format(i, N, "*" * i, " " * (N - i)), end="")
        if i == N:
            break
        batch = batch.to(device)
        with torch.no_grad():
            batch = batch.mul_(0.5).add_(0.5)
            batch = upbiln(batch)
            batch = norm(batch)
            out = model(batch)
            prob = shave(out).detach().cpu().numpy()
        py = prob.mean(axis=0)
        # batch_score = np.asarray([entropy(py, px) for px in prob])
        # scores.append(np.exp(np.mean(batch_score)))
        kl = np.log(prob) - np.log(np.expand_dims(py, axis=0))
        kl = np.sum(prob * kl, axis=1)
        scores.append(np.log(1 + kl.mean()))
    print()
    return np.mean(scores), np.std(scores)

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content

if __name__ == '__main__':
    trial_name = "saved/adam_rat50_1e-2"
    print(read_json(f"{trial_name}/conf.json"))

    G = Generator().to(device)
    model_list = os.listdir(os.path.join(trial_name, "checkpoints"))
    model_list = sorted(model_list, key=lambda x: (len(x), x))
    print(model_list)
    _ = input("continue? (y/N)")

    fd = open("saved/a.csv", "w", newline="")
    csv_writer = csv.writer(fd)
    for model_name in model_list:
        model_name = os.path.join(trial_name, f"checkpoints/{model_name}")
        state = torch.load(model_name, map_location="cpu")
        G.load_state_dict(state["gen"])
        G.eval()
        X = np.zeros((0, 3, 32, 32), dtype=np.float32)
        for i in range(64):
            z = torch.randn(128, 100).to(device)
            x = G(z)
            x = x.mul_(0.5).add_(0.5)
            x = x.detach().cpu().numpy()
            X = np.append(X, x, axis=0)
        X, y = np2TT(X), np2TT(np.arange(X.shape[0]))
        genset = torch.utils.data.TensorDataset(X, y)
        gen_loader = torch.utils.data.DataLoader(
            genset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True
        )
        mu, s = calc_inception_score(gen_loader, N=64)
        csv_writer.writerow([state["epoch"], mu, s])
    fd.close()
