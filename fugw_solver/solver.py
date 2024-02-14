import torch

from fugw_solver.loss import compute_fugw_loss 
from fugw_solver.cost import cost
from fugw_solver.scaling import scaling 

def solver(source_features, target_features, source_geometry, target_geometry, w_s, w_t, device, nits_bcd=10, nits_uot=1000, alpha=0.5, rho=1, eps=1e-2):
    w_sxw_t = w_s[:, None] * w_t[None, :]
    P = w_sxw_t / w_sxw_t.sum()

    F_s = torch.tensor(source_features.T, device=device)
    F_t = torch.tensor(target_features.T, device=device)
    F = torch.cdist(F_s, F_t, p=2) ** 2
    D_s = torch.tensor(source_geometry, device=device)
    D_t = torch.tensor(target_geometry, device=device)

    Q = P

    loss = []
    loss.append(compute_fugw_loss(P, Q, w_s, w_t, w_sxw_t, D_s, D_t, F, rho, eps, alpha))

    for _ in range(nits_bcd):
        mass_P = P.sum()

        # Update Q    
        c_q = cost(P, w_s, w_t, w_sxw_t, D_s, D_t, F, rho, eps, alpha)

        new_rho = rho * mass_P
        new_eps = eps * mass_P
        Q = scaling(c_q, new_rho, new_eps, w_s, w_t, w_sxw_t, nits_uot)
        # Rescale Q
        Q = (mass_P / Q.sum()).sqrt() * Q


        # Update P
        mass_Q = Q.sum()
        c_p = cost(Q, w_s, w_t, w_sxw_t, D_s, D_t, F, rho, eps, alpha)

        new_rho = rho * mass_Q
        new_eps = eps * mass_Q
        P = scaling(c_p, new_rho, new_eps, w_s, w_t, w_sxw_t, nits_uot)
        # Rescale P
        P = (mass_Q / P.sum()).sqrt() * P

        loss.append(compute_fugw_loss(P, Q, w_s, w_t, w_sxw_t, D_s, D_t, F, rho, eps, alpha))
        
    return P, loss