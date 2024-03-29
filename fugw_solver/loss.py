import torch

from fugw_solver.utils import compute_quad_kl

def compute_fugw_loss(P, Q, w_s, w_t, ws_wt, D_s, D_t, F, rho, eps, alpha):
        D_s_sqr, D_t_sqr = D_s**2, D_t**2 

        P1, P2 = P.sum(1), P.sum(0)
        Q1, Q2 = Q.sum(1), Q.sum(0)

        loss_wasserstein = torch.zeros(1)
        loss_gromov_wasserstein = torch.zeros(1)
        loss_marginal_constraint_dim1 = torch.zeros(1)
        loss_marginal_constraint_dim2 = torch.zeros(1)
        loss_regularization = torch.zeros(1)
        loss = 0

        if alpha != 1 and F is not None:
            loss_wasserstein = ((F * P).sum() + (F * Q).sum()) / 2
            loss += (1 - alpha) * loss_wasserstein
        if alpha != 0:
            A = (D_s_sqr @ Q1).dot(P1)
            B = (D_t_sqr @ Q2).dot(P2)
            C = (D_s @ Q @ D_t.T) * P
            loss_gromov_wasserstein = A + B - 2 * C.sum()
            loss += alpha * loss_gromov_wasserstein

        loss_marginal_constraint_dim1 = compute_quad_kl(
            P1, Q1, w_s, w_s
        )
        loss += rho * loss_marginal_constraint_dim1
        loss_marginal_constraint_dim2 = compute_quad_kl(
            P2, Q2, w_t, w_t
        )
        loss += rho * loss_marginal_constraint_dim2
    
        if eps != 0:
            loss_regularization = compute_quad_kl(
                P, Q, ws_wt, ws_wt
            )
                
        return {
            "wasserstein": loss_wasserstein.item(),
            "gromov_wasserstein": loss_gromov_wasserstein.item(),
            "marginal_constraint_dim1": loss_marginal_constraint_dim1.item(),
            "marginal_constraint_dim2": loss_marginal_constraint_dim2.item(),
            "regularization": loss_regularization.item(),
            "total": loss.item(),
        }