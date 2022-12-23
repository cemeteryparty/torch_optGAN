from torch.optim import Optimizer
import torch

class optimAdam(Optimizer):
    """ https://github.com/201419/Optimizer-PyTorch/blob/master/omd.py """
    """ https://github.com/vsyrgkanis/optimistic_GAN_training/blob/master/script/optimizer.py """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                 weight_decay=weight_decay, amsgrad=amsgrad)
        super(optimAdam, self).__init__(params, defaults)
        self.betas = betas
    def __setstate__(self, state):
        super(optimAdam, self).__setstate__(state)

    def step(self):
        beta1, beta2 = self.betas
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return None
                # get grads
                d_p = p.grad.data
                if group["weight_decay"] != 0:
                    d_p = d_p.add_(p.data, alpha=group["weight_decay"])
                d_p_sqr = d_p * d_p

                state = self.state[p]  # stored t - 1 state
                if len(state) == 0:  # State initialization
                    state["step"] = 0
                    state["m_t"] = torch.zeros_like(p.data)  # first momentum
                    state["v_t"] = torch.zeros_like(p.data)  # second momentum
                    state["m_tHat"] = torch.zeros_like(p.data)  # first momentum with bias correction
                    state["v_tHat"] = torch.zeros_like(p.data)  # second momentum with bias correction

                state["step"] += 1
                m_t, v_t = state["m_t"], state["v_t"]
                # update first and second momentum
                m_t.mul_(beta1).add_(d_p, alpha=1 - beta1)
                v_t.mul_(beta2).add_(d_p_sqr, alpha=1 - beta2)
                # update first and second momentum with bias correction
                m_tHat = m_t.clone()
                m_tHat.div_(1 - beta1 ** state["step"])
                v_tHat = v_t.clone()
                v_tHat.div_(1 - beta2 ** state["step"])
                # update weight through present state
                denom_a = m_tHat.clone()
                denom_b = v_tHat.sqrt().add_(group["eps"])
                p.data.addcdiv_(denom_a, denom_b, value=(-2) * group["lr"])
                # update weight through prev state
                denom_a = state["m_tHat"].clone()
                denom_b = state["v_tHat"].sqrt().add_(group["eps"])
                p.data.addcdiv_(denom_a, denom_b, value=group["lr"])

                # store state
                state["m_t"] = m_t.clone()
                state["v_t"] = v_t.clone()
                state['m_tHat'] = m_tHat.clone()
                state['v_tHat'] = v_tHat.clone()

class SOMD(Optimizer):
    def __init__(self, params, lr=0.001, version=1, m_rho=0.1):
        defaults = dict(lr=lr)
        super(SOMD, self).__init__(params, defaults)
        self.ver = version
        self.m_rho = m_rho

    def __setstate__(self, state):
        super(OptimisticAdam, self).__setstate__(state)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]  # stored t - 1 state
                if len(state) == 0:  # State initialization
                    state["step"] = 0
                    state["m"] = torch.zeros_like(d_p)  # predictor
                
                state["step"] += 1
                # update predictor
                if self.ver == 1:
                    m = d_p.clone()
                elif self.ver == 2:
                    m = (m.data * (state["step"] - 1) + d_p) / state["step"]
                elif self.ver == 3:
                    m = m.data * self.m_rho + (1 - self.m_rho) * d_p
                else:
                    raise ValueError("brain doko")
                # update weight
                new_d = d_p + m - state["m"]
                p.data.add_(new_d, alpha=-group['lr'])

                # store state
                state["m"] = m
