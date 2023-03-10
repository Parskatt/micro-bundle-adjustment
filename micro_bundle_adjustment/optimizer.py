import torch
from functorch import jacrev, vmap, jacfwd
from einops import einsum


def schur_solve(cam_block, cross_block, point_block, g):
    N = point_block.shape[0]
    point_block_inv = torch.linalg.inv(point_block)
    g_camera, g_points = g[:6], g[6:] # TODO: Remove hardcoded number of coordinates
    g_points = g_points.reshape(N,3)
    
    S = cam_block - einsum(cross_block, (point_block_inv @ cross_block.mT), " n p1 x, n x p2 -> p1 p2")
    
    tmp = einsum(point_block_inv, g_points,"n x1 x2, n x2 -> n x1")
    camera_rhs = g_camera + einsum(cross_block, tmp,"n p x, n x -> p")
    camera_params = torch.linalg.solve(S, camera_rhs)
    
    points_rhs = g_points - einsum(cross_block, camera_params, "n p x, p -> n x")
    point_coordinates = einsum(point_block_inv, points_rhs,"n x1 x2, n x2 -> n x1")
    return camera_params, point_coordinates

def lm_optimize(f, X_0, r_0, t_0, image_A_points, image_B_points, num_steps = 10, L_0 = 10):
    
    N, D = X_0.shape
    X = X_0.clone()
    r = r_0.clone()
    t = t_0.clone()
    L = L_0
    
    jacobian_operator_x = vmap(jacrev(f, 0))
    jacobian_operator_rt = jacfwd(f, (1,2))
    
    for step in range(num_steps):
        residuals = f(X, r, t, image_A_points, image_B_points)
        loss = (residuals**2).sum()
        J_x = jacobian_operator_x(X,r,t,image_A_points,image_B_points)[:,0]
        
        J_r, J_t = jacobian_operator_rt(X,r[[0]],t[[0]],image_A_points,image_B_points)
        J_theta = torch.cat((J_r,J_t),dim=-1)[:,:,0]
        damp_x = L * torch.eye(D,device=X.device)[None]
        damp_theta = L * torch.eye(2 * D,device=X.device)
        
        camera_block = einsum(J_theta, J_theta, "n i t, n i p -> t p") + damp_theta
        cross_block = J_theta.mT @ J_x
        points_block = J_x.mT @ J_x + damp_x
        g_theta = -einsum(J_theta,  residuals, "n r p, n r -> p")
        g_points = -einsum(J_x,  residuals, "n r p, n r -> n p").flatten()
        g = torch.cat((g_theta, g_points))
        delta_theta, delta_x = schur_solve(camera_block, cross_block, points_block, g)
        # torch.linalg.solve(JTJ+L*damp, )[..., 0]
        r_new = r + delta_theta[:3]
        t_new = t + delta_theta[3:]
        X_new = X + delta_x
        loss_new = (f(X_new, r_new, t_new, image_A_points, image_B_points)**2).sum()
        # Could we have L for each parameter in x?
        if loss_new < loss:
            L = L/10
            r = r_new
            t = t_new
            X = X_new
        else:
            L = L*10
        print(f"{loss_new=} {loss=} {L=}")
    return X, r[0], t[0]