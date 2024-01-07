import torch
from kornia.geometry import axis_angle_to_rotation_matrix
from .optimizer import lm_optimize

def projection(X, r, t):
    R = axis_angle_to_rotation_matrix(r[None])[0]
    if len(X.shape) > 1:#TODO: don't want this
        x = (R @ X.mT).mT + t[None]
    else:        
        x = (R @ X) + t
    return x[...,:2]/x[...,[2]]

def calibrated_residuals(X, theta, x_im):
    r, t = theta.chunk(2)
    x_im_hat = projection(X, r, t)
    r_im = x_im_hat - x_im
    return r_im

def simple_pinhole_residuals(X, theta, x_im):
    intrinsics, r, t = theta.chunk(3)
    principal_point = intrinsics[1:]
    f = intrinsics[0]
    r_im = f*projection(X, r, t) + principal_point - x_im
    return r_im

# colmap: https://github.com/colmap/colmap/blob/f5597bf7abc2bdfa3f99b8cbeed89173062bdbfe/src/colmap/sensor/models.h#L775
def simple_radial_residuals(X, theta, x_im):
    intrinsics, r, t = theta[:4], theta[4:7], theta[7:]
    principal_point = intrinsics[1:3]
    f = intrinsics[0]
    k = intrinsics[3]
    x = projection(X, r, t)
    radius_squared = x.pow(2).sum(dim=-1, keepdim = True)
    x = x * (1 + k * radius_squared)
    r_im = f*x + principal_point - x_im
    return r_im



def optimize_calibrated(X_0, r_0, t_0, observations, dtype=torch.float32, L_0 = 1e-2, num_steps = 5):
    theta_0 = torch.cat((r_0, t_0), dim = -1)
    X_hat, theta_hat = lm_optimize(calibrated_residuals, X_0, theta_0, observations, dtype=dtype, L_0 = L_0, num_steps = num_steps)
    return X_hat, theta_hat

def optimize_simple_radial(X_0, f, principal_point, k, r_0, t_0, observations, dtype=torch.float32, L_0 = 1e-2, num_steps = 5):
    theta_0 = torch.cat((f, principal_point, k, r_0, t_0), dim = -1)
    X_hat, theta_hat = lm_optimize(simple_radial_residuals, X_0, theta_0, observations, dtype=dtype, L_0 = L_0, num_steps = num_steps)
    return X_hat, theta_hat