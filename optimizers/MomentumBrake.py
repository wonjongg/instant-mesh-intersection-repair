"""
Momentum Brake optimizer with adaptive gradient filtering.
"""

import torch


class MomentumBrake(torch.optim.Optimizer):
    """
    Momentum Brake optimizer with adaptive gradient filtering.

    This optimizer extends Adam-style momentum by selectively applying momentum
    updates only when gradients fall within statistical confidence bands. This
    helps prevent oscillations and provides more stable convergence.

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Learning rate. Defaults to 0.1.
        betas (tuple, optional): Coefficients for first and second moment estimates. Defaults to (0.99, 0.999).
    """

    def __init__(self, params, lr=0.1, betas=(0.99, 0.999)):
        """Initialize the optimizer."""
        defaults = dict(lr=lr, betas=betas)
        super(MomentumBrake, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set state for serialization."""
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
        """
        Perform a single optimization step with momentum braking.

        Applies momentum updates selectively based on whether gradients fall within
        statistical confidence bands computed from moment estimates. After a warmup
        period (10 steps), momentum is only applied when gradients are within bounds.
        """
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']

            for p in group["params"]:
                state = self.state[p]

                # Lazy initialization of moment estimates
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)  # First moment (mean)
                    state["g2"] = torch.zeros_like(p.data)  # Second moment (variance)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                # Check if gradient is within confidence bands
                within_bands, std = self._check_within_bands(
                    grad, g1, g2, state['step'], b1, b2, 3.0
                )

                # After warmup, apply conditional momentum updates
                if state["step"] > 10:
                    # Update first moment with braking: only apply momentum when within bands
                    g1.mul_(b1 * within_bands).add_(grad * (1 - b1 * within_bands))

                    # Update second moment (always applied)
                    g2.mul_(b2).add_(grad.square(), alpha=1 - b2)

                    # Compute bias-corrected moments
                    m1 = g1 / (1 - (b1 ** state["step"]))
                    m2 = g2 / (1 - (b2 ** state["step"]))

                    # Use scaled first moment instead of adaptive denominator
                    gr = 2 * m1

                    # Apply parameter update
                    p.data.sub_(gr, alpha=lr)
                else:
                    # Warmup phase: standard momentum updates
                    g1.mul_(b1).add_(grad * (1 - b1))

                    g2.mul_(b2).add_(grad.square(), alpha=1 - b2)

                    # Compute bias-corrected moments
                    m1 = g1 / (1 - (b1 ** state["step"]))
                    m2 = g2 / (1 - (b2 ** state["step"]))

                    # Use scaled first moment
                    gr = 2 * m1

                    # Apply parameter update
                    p.data.sub_(gr, alpha=lr)
