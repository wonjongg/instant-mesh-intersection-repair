"""
Standard Gradient Descent optimizer.
"""

import torch


class GradientDescent(torch.optim.Optimizer):
    """
    Standard Gradient Descent optimizer.

    Implements basic gradient descent with learning rate scheduling support.
    Unlike Adam, this optimizer does not use adaptive learning rates or momentum.

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Learning rate. Defaults to 0.1.
        betas (tuple, optional): Coefficients for momentum (unused, kept for compatibility). Defaults to (0.9, 0.999).
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999)):
        """Initialize the optimizer."""
        defaults = dict(lr=lr, betas=betas)
        super(GradientDescent, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set state for serialization."""
        super(GradientDescent, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        """
        Perform a single optimization step.

        Updates parameters using standard gradient descent: p = p - lr * grad
        """
        for group in self.param_groups:
            lr = group['lr']

            for p in group["params"]:
                state = self.state[p]

                # Lazy initialization of state
                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1
                grad = p.grad.data

                # Apply gradient descent update: p = p - lr * grad
                gr = grad
                p.data.sub_(gr, alpha=lr)
