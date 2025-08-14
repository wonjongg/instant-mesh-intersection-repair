#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

class GradientDescent(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """
    def __init__(self, params, lr=0.1, betas=(0.9,0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(GradientDescent, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GradientDescent, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state)==0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                gr = grad
                p.data.sub_(gr, alpha=lr)
