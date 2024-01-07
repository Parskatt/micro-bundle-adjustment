import torch
from torch.func import vmap
from micro_bundle_adjustment.api import projection, optimize_simple_radial

if __name__ == "__main__":
    N = 1_000_000
    dtype = torch.float32
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
    print(f"Residuals opt: {(X_0-X).norm(dim=-1).mean()=}")
    
    noisy_f = f  + torch.randn_like(f).clamp(-1,1)*2
    noisy_principal_point = principal_point + torch.randn_like(principal_point).clamp(-1,1)*2
    noisy_r = r + torch.randn_like(r).clamp(-1,1)*0.01
    noisy_t = t + torch.randn_like(t).clamp(-1,1)*0.5
    noisy_k = torch.randn_like(f).clamp(-1,1)*0.01
        
    with torch.no_grad():
        X_hat, theta_hat = optimize_simple_radial(X_0, noisy_f, noisy_principal_point, noisy_k, noisy_r, noisy_t, 
                                                  observations, 
                                                  dtype=dtype, L_0 = 1e-2, num_steps = 5)
        
    print(theta_hat)
    
    X = (X-X.mean(dim=0))
    X_hat = (X_hat-X_hat.mean(dim=0))
    
    X_error_opt = (X-X_hat).norm(dim=-1).mean()
    
    print(f"Residuals opt: {X_error_opt=}")