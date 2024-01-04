import torch
from .optimizer import lm_optimize
from typing import Callable

def bundle_adjust(residual_function: Callable, X_0: torch.Tensor, theta_0: torch.Tensor, observations: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, D  = X_0.shape
    with torch.no_grad():
        X, theta = lm_optimize(residual_function, X_0, theta_0, observations, num_steps = 10)
    return X, theta