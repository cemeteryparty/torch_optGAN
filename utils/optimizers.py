from torch.optim import Optimizer
import torch
import copy

class SOMD(Optimizer):
    def __init__(self, params, lr=0.001, version=1):
        defaults = dict(lr=lr)
        super(SOMD, self).__init__(params, defaults)
        self.ver = version
        self.schedule = None

        self.M = copy.deepcopy(self.param_groups)  # predictable
        self.S = copy.deepcopy(self.param_groups)  # scheduler
        for pred, sche in zip(self.M, self.S):
            for m, s in zip(pred["params"], sche["params"]):
                m.data = torch.zeros_like(m.data)
                s.data = torch.ones_like(s.data)
        self.t = torch.tensor(1, dtype=torch.float32)  # iteration
    def step(self):
        for group, pred, sche in zip(self.param_groups, self.M, self.S):
            for p, m, s in zip(group["params"], pred["params"], sche["params"]):
                if p.grad is None:
                    continue
                if self.ver == 1:
                    new_m = copy.deepcopy(p.grad.data)
                if self.ver == 2:
                    new_m = (m.data * (self.t - 1) + p.grad.data) / self.t
                elif self.ver == 3:
                    new_m = m.data * self.m_rho + (1-self.m_rho) * p.grad.data
                else:
                    raise ValueError("brain doko")
                if self.schedule is None:
                    new_s = copy.deepcopy(s)
                elif self.schedule == "adagrad":
                    pass
                else:
                    raise ValueError("brain doko")
                
                # Update param
                p.data.mul_(new_s / s)
                new_d = new_s * (m - new_m - p.grad.data)
                p.data.add_(new_d, alpha=group['lr'])
                # Update predictor
                m.data.mul_(0)
                m.data.add_(new_m, alpha=1)
        self.t += 1
