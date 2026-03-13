import torch

class GradientDescent(torch.optim.Optimizer):
    """
    Standard GradientDescent optimizer
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
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state)==0:
                    state["step"] = 0

                state["step"] += 1
                grad = p.grad.data

                gr = grad
                p.data.sub_(gr, alpha=lr)
