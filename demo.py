import torch
import kornia
from micro_bundle_adjustment import bundle_adjust, angle_axis_to_rotation_matrix


def projection(X, r, t):
    if len(X.shape) == 1:
        X = X[None]
    N, D = X.shape
    if len(r.shape) == 1:
        r = r.expand(N,D)
        t = t.expand(N,D)
    R = angle_axis_to_rotation_matrix(r)
    x = (R @ X[...,None]) + t[...,None]
    x = x[...,0]
    return x[...,:2]/x[...,[2]]

def gold_standard_residuals(X, r, t, x_a, x_b):
    r_a = x_a - projection(X, torch.zeros_like(r), torch.zeros_like(t))
    r_b = x_b - projection(X, r, t)
    return torch.cat((r_a, r_b), dim=1)

if __name__ == "__main__":
    N = 1_000_000
    dtype = torch.float32
    device = "cuda"
    X = torch.rand(N, 3).to(device=device,dtype=dtype)
    X[...,2] = X[...,2] + 10
    
    r = torch.rand(3).to(device=device,dtype=dtype) * 1 # A large rotation
    t = torch.rand(3).to(device=device,dtype=dtype) * 0.5 # A small translation
    
    x_a = projection(X, torch.zeros_like(r), torch.zeros_like(t))
    x_b = projection(X, r, t)

    image_A_points = x_a + 0.001*torch.rand_like(x_a)
    image_B_points = x_b + 0.001*torch.rand_like(x_b)
    noisy_scene_points = X + 0.1*torch.rand_like(X)
    noisy_r = r + torch.rand_like(r)*0.1
    noisy_t = t + torch.rand_like(t)*0.1
    
    X_hat, r_hat, t_hat = bundle_adjust(gold_standard_residuals, noisy_scene_points, noisy_r, noisy_t, image_A_points, image_B_points, )
    r_error_opt = (r_hat-r).norm()
    r_error_init = (noisy_r-r).norm()
    
    t_error_opt = (t_hat-t).norm()
    t_error_init = (noisy_t-t).norm()
    
    X_error_opt = (X-X_hat).norm(dim=-1).mean()
    X_error_init = (noisy_scene_points-X).norm(dim=-1).mean()
    
    print(f"Errors: {r_error_init=} {r_error_opt=} {t_error_init=} {t_error_opt=} {X_error_init=} {X_error_opt=}")