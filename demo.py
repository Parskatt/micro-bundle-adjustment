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
    r, t = theta.chunk(2)
    r_im = x_im - projection(X, r, t)
    return r_im

if __name__ == "__main__":
    N = 100_000
    dtype = torch.float64
    device = "cuda"
    X = torch.randn(N, 3).to(device=device,dtype=dtype)
    X[...,2] = X[...,2] + 10
    
    r = torch.randn(2,3).clamp(-0.5,0.5).to(device=device,dtype=dtype) * 1 # A large rotation
    t = torch.randn(2,3).clamp(-1,1).to(device=device,dtype=dtype) * 0.5 # A small translation
    
    batch_projection = vmap(projection, in_dims=(0, None, None))
    x_a = batch_projection(X, r[0], t[0])
    x_b = batch_projection(X, r[1], t[1])

    observations = [(x_a, torch.arange(len(X), device = device))] + \
        [(x_b, torch.arange(len(X), device = device))]
    X_0 = X + torch.randn_like(X).clamp(-1,1)*0.05
    noisy_r = r + torch.randn_like(r).clamp(-1,1)*0.01
    noisy_t = t + torch.randn_like(t).clamp(-1,1)*0.5
    theta_0 = torch.cat((noisy_r, noisy_t), dim = -1)
    
    with torch.no_grad():
        X_hat, theta_hat = lm_optimize(gold_standard_residuals, X_0, theta_0, observations, dtype=dtype, L_0 = 1e-2, num_steps = 20)
        
    R = axis_angle_to_rotation_matrix(r)
    R_rel, t_rel = relative_camera_motion(R[:1], t[:1,:,None], R[1:], t[1:,:,None])
    r_rel = rotation_matrix_to_axis_angle(R_rel)
    
    r_hat, t_hat = theta_hat.chunk(2, dim=1)
    R_hat = axis_angle_to_rotation_matrix(r_hat)
    R_hat_rel, t_hat_rel = relative_camera_motion(R_hat[:1], t_hat[:1, :, None], R_hat[1:], t_hat[1:, :, None])
    r_hat_rel = rotation_matrix_to_axis_angle(R_hat_rel)
    
    r_error_opt = (r_hat_rel-r_rel).norm()
    
    t_error_opt = (t_hat_rel-t_rel).norm()
    
    X = (X-X.mean(dim=0))
    X_hat = (X_hat-X_hat.mean(dim=0))
    
    X_error_opt = (X-X_hat).norm(dim=-1).mean()
    
    print(f"Residuals: {r_error_opt=} {t_error_opt=} {X_error_opt=}")