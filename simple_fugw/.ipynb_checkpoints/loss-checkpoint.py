import torch
from simple_fugw.utils import compute_unnormalized_kl, compute_divergence, compute_quad_kl

def compute_fugw_loss(P, G, w_s, w_t, ws_wt, D_s, D_t, F, rho, eps, alpha):
        D_s_sqr, D_t_sqr = D_s**2, D_t**2 

        P1, P2 = P.sum(1), P.sum(0)
        G1, G2 = G.sum(1), G.sum(0)

        loss_wasserstein = torch.zeros(1)
        loss_gromov_wasserstein = torch.zeros(1)
        loss_marginal_con straint_dim1 = torch.zeros(1)
        loss_marginal_constraint_dim2 = torch.zeros(1)
        loss_regularization = torch.zeros(1)
        loss = 0

        if alpha != 1 and F is not None:
            loss_wasserstein = ((F * P).sum() + (F * G).sum()) / 2
            loss += (1 - alpha) * loss_wasserstein
        if alpha != 0:
            A = (D_s_sqr @ G1).dot(P1)
            B = (D_t_sqr @ G2).dot(P2)
            C = (D_s @ G @ D_t.T) * P
            loss_gromov_wasserstein = A + B - 2 * C.sum()
            loss += alpha * loss_gromov_wasserstein

        loss_marginal_constraint_dim1 = compute_quad_kl(
            P1, G1, w_s, w_s
        )
        loss += rho * loss_marginal_constraint_dim1
        loss_marginal_constraint_dim2 = compute_quad_kl(
            P2, G2, w_t, w_t
        )
        loss += rho * loss_marginal_constraint_dim2
    
        if eps != 0:
            loss_regularization = compute_quad_kl(
                P, G, ws_wt, ws_wt
            )

        print(loss.item())
                
        return {
            "wasserstein": loss_wasserstein.item(),
            "gromov_wasserstein": loss_gromov_wasserstein.item(),
            "marginal_constraint_dim1": loss_marginal_constraint_dim1.item(),
            "marginal_constraint_dim2": loss_marginal_constraint_dim2.item(),
            "regularization": loss_regularization.item(),
            "total": loss.item(),
        }