import torch

class GaussianAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, miu: float = 0, sigma: float = 1):
        self.miu = miu
        self.sigma = sigma
        super().__init__(optimizer, -1)

    def get_lr(self):
        step = self.last_epoch
        factor = torch.exp(torch.tensor([-0.5 * ((step - self.miu) / self.sigma) ** 2]))
        return [base_lr * factor.item() for base_lr in self.base_lrs]

def get_scheduler(NAME: str, optimizer: torch.optim.Optimizer, T_max: int, miu: float, sigma: float, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    if NAME == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, last_epoch=-1)
    elif NAME == "gaussian_annealing":
        return GaussianAnnealingLR(optimizer, miu=miu, sigma=sigma)