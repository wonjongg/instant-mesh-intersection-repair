#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

class MomentumBrake(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """
    def __init__(self, params, lr=0.1, betas=(0.99,0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(MomentumBrake, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MomentumBrake, self).__setstate__(state)

    def _check_within_bands(self, grad, exp_avg, exp_avg_sq, step, beta1, beta2, band_width):
        """
        Check if current gradient is within bands computed from moment estimates.
        Uses bias-corrected estimates for more accurate bounds early in training.
        """
        # Compute bias corrections
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        #  bias_correction1 = 1
        #  bias_correction2 = 1
        
        # Get bias-corrected first moment (mean)
        mean = exp_avg / bias_correction1
        
        # Compute bias-corrected second moment and standard deviation
        second_moment = exp_avg_sq / bias_correction2
        variance = second_moment - mean.pow(2)
        std = variance.sqrt()
        
        # Compute bands using the band_width multiplier
        upper_band = mean + band_width * std
        lower_band = mean - band_width * std
        
        # Check if gradient components are within bands
        within_bands = torch.logical_and(
            grad >= lower_band,
            grad <= upper_band
        )
        within_bands = torch.all(within_bands, dim=1, keepdim=True)
        
        return within_bands, std

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

                within_bands, std = self._check_within_bands(
                    grad, g1, g2, state['step'], b1, b2, 3.0
                )
                if state["step"] > 10:
                    g1.mul_(b1 * within_bands).add_(grad * (1-b1*within_bands))

                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)
                    m1 = g1 / (1-(b1**state["step"]))
                    m2 = g2 / (1-(b2**state["step"]))
                    # This is the only modification we make to the original Adam algorithm
                    #  gr = m1 / (1e-8 + m2.sqrt().max())
                    gr = 2 * m1

                    p.data.sub_(gr, alpha=lr)
                else:
                    g1.mul_(b1).add_(grad * (1-b1))

                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)
                    m1 = g1 / (1-(b1**state["step"]))
                    m2 = g2 / (1-(b2**state["step"]))
                    # This is the only modification we make to the original Adam algorithm
                    gr = 2 * m1
                    #  gr = m1 / (1e-8 + m2.sqrt().max())

                    p.data.sub_(gr, alpha=lr)
