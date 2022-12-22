from torch.autograd import grad as Gradient
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as T
from torchvision import datasets
import torch
import os
from torch import from_numpy as np2TT
from torchinfo import summary
from scipy.io import savemat
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time

print(torch.cuda.is_available())
print(torch.cuda.device_count())
dev_id = torch.cuda.current_device()
print(torch.cuda.get_device_name(dev_id))
