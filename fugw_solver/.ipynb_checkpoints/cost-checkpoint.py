import torch

from fugw_solver.utils import compute_unnormalized_kl

def cost(P, w_s, w_t, ws_wt, D_s, D_t, F, rho, eps, alpha):

    D_s_sqr, D_t_sqr = D_s**2, D_t**2 
    
    D_s, D_t, D_s_sqr, D_t_sqr = D_s.T, D_t.T, D_s_sqr.T, D_t_sqr.T

    P1, P2 = P.sum(1), P.sum(0)

    cost = torch.zeros_like(P)

    if alpha != 1 and F is not None:
        wasserstein_cost = F / 2
        cost += (1 - alpha) * wasserstein_cost

    if alpha != 0:
        A = D_s_sqr @ P1
        B = D_t_sqr @ P2
        gromov_wasserstein_cost = (
            A[:, None] + B[None, :] - 2 * D_s @ P @ D_t.T
        )
        cost += alpha * gromov_wasserstein_cost

    marginal_cost_dim1 = compute_unnormalized_kl(P1, w_s)
    cost += rho * marginal_cost_dim1
    marginal_cost_dim2 = compute_unnormalized_kl(P2, w_t)
    cost += rho * marginal_cost_dim2

    regularized_cost = compute_unnormalized_kl(P, ws_wt)
    cost += eps * regularized_cost
    
    return cost