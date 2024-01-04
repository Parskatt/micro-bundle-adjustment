import torch
from kornia.geometry import axis_angle_to_rotation_matrix
from micro_bundle_adjustment import bundle_adjust


def projection(X, r, t):
    if len(X.shape) == 1: # should be unlikely
        X = X[None]
    R = axis_angle_to_rotation_matrix(r[None])[0]
    x = (R @ X.mT).mT + t[None]
    return x[...,:2]/x[...,[2]]

def gold_standard_residuals(X, theta, oberservations):
    M = len(oberservations)
    r, t = theta.chunk(2,dim=1)
    residuals = []
    for im in range(len(observations)):
        x_im, inds_im = observations[im]
        r_im = x_im - projection(X[inds_im], r[im], t[im])
        residuals.append(r_im)
    return torch.cat(residuals, dim=1)

if __name__ == "__main__":
    N = 100_000
    dtype = torch.float32
    device = "cuda"
    X = torch.rand(N, 3).to(device=device,dtype=dtype)
    X[...,2] = X[...,2] + 10
    
    r = torch.rand(2,3).to(device=device,dtype=dtype) * 1 # A large rotation
    t = torch.rand(2,3).to(device=device,dtype=dtype) * 0.5 # A small translation
    
    x_a = projection(X, r[0], t[0])
    x_b = projection(X, r[1], t[1])

    observations = [(x_a + 0.001*torch.rand_like(x_a), torch.arange(len(X), device = device)[:,None])] + \
        [(x_b + 0.001*torch.rand_like(x_b), torch.arange(len(X), device = device)[:,None])]
    X_0 = X + 0.1*torch.rand_like(X)
    noisy_r = r + torch.rand_like(r)*0.1
    noisy_t = t + torch.rand_like(t)*0.1
    theta_0 = torch.cat((noisy_r, noisy_t), dim = -1)
    
    X_hat, theta_hat = bundle_adjust(gold_standard_residuals, X_0, theta_0, observations)
    r_hat, t_hat = theta_hat.chunk(2)
    r_error_opt = (r_hat-r).norm()
    r_error_init = (noisy_r-r).norm()
    
    t_error_opt = (t_hat-t).norm()
    t_error_init = (noisy_t-t).norm()
    
    X_error_opt = (X-X_hat).norm(dim=-1).mean()
    X_error_init = (X_0-X).norm(dim=-1).mean()
    
    print(f"Residuals: {r_error_init=} {r_error_opt=} {t_error_init=} {t_error_opt=} {X_error_init=} {X_error_opt=}")