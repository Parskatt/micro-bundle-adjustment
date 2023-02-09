import torch
from .optimizer import lm_optimize

def bundle_adjust(residual_function, scene_points, noisy_r, noisy_t, image_A_points, image_B_points):
    N, D  = scene_points.shape
    noisy_r = noisy_r[None].expand(N,D)
    noisy_t = noisy_t[None].expand(N,D)
    with torch.no_grad():
        X, r, t = lm_optimize(residual_function, scene_points, noisy_r, noisy_t, image_A_points, image_B_points, num_steps=10)
    return X, r, t