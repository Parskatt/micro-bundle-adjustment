import torch
from torch.func import vmap
from kornia.geometry import axis_angle_to_rotation_matrix, relative_camera_motion, rotation_matrix_to_axis_angle
from micro_bundle_adjustment import lm_optimize


def projection(X, r, t):
    R = axis_angle_to_rotation_matrix(r[None])[0]
    if len(X.shape) > 1:#TODO: don't want this
        x = (R @ X.mT).mT + t[None]
    else:        
        x = (R @ X) + t
    return x[...,:2]/x[...,[2]]

def gold_standard_residuals(X, theta, x_im):
    r, t, intrinsics = theta.chunk(3)
    offset = intrinsics[1:]
    f = intrinsics[0]
    r_im = f*projection(X, r, t) + offset - x_im
    return r_im

if __name__ == "__main__":
    N = 100_000
    dtype = torch.float64
    device = "cuda"
    X = torch.randn(N, 3).to(device=device,dtype=dtype)
    X[...,2] = X[...,2] + 10
    
    r = torch.randn(2,3).clamp(-0.5,0.5).to(device=device,dtype=dtype) * 1 # A large rotation
    t = torch.randn(2,3).clamp(-1,1).to(device=device,dtype=dtype) * 0.5 # A small translation
    f = 100 * torch.ones((2,1), device = device, dtype = dtype)
    i_x = 200 * torch.ones((2,1), device = device, dtype = dtype)
    i_y = 100 * torch.ones((2,1), device = device, dtype = dtype)
    principal_point = torch.cat((i_x, i_y), dim = -1)
    batch_projection = vmap(projection, in_dims=(0, None, None))
    x_a = f[0,0]*batch_projection(X, r[0], t[0]) + principal_point[:1]
    x_b = f[1,0]*batch_projection(X, r[1], t[1]) + principal_point[1:]

    observations = [(x_a, torch.arange(len(X), device = device))] + \
        [(x_b, torch.arange(len(X), device = device))]
    X_0 = X + torch.randn_like(X).clamp(-1,1)*0.05
    noisy_r = r + torch.randn_like(r).clamp(-1,1)*0.01
    noisy_t = t + torch.randn_like(t).clamp(-1,1)*0.5
    noisy_f, noisy_i_x, noisy_i_y = f  + torch.randn_like(f).clamp(-1,1)*5, i_x  + torch.randn_like(i_x).clamp(-1,1)*5, i_y  + torch.randn_like(i_y).clamp(-1,1)*5
    theta_0 = torch.cat((noisy_r, noisy_t, noisy_f, noisy_i_x, noisy_i_y), dim = -1)
    
    print(theta_0)
    
    with torch.no_grad():
        X_hat, theta_hat = lm_optimize(gold_standard_residuals, X_0, theta_0, observations, dtype=dtype, L_0 = 1e-2, num_steps = 20)
        
    print(theta_hat)
    
    X = (X-X.mean(dim=0))
    X_hat = (X_hat-X_hat.mean(dim=0))
    
    X_error_opt = (X-X_hat).norm(dim=-1).mean()
    
    print(f"Residuals: {X_error_opt=}")