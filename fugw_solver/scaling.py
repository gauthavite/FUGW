import torch

def scaling(cost, rho, eps, w_s, w_t, ws_wt, niters):
    """Scaling algorithm (ie Sinkhorn algorithm)."""

    log_ws = w_s.log()
    log_wt = w_t.log()
    u, v = torch.zeros_like(w_s), torch.zeros_like(w_t)

    tau = rho / (rho + eps)

    for _ in range(niters):
        if rho != 0:
            v = -tau * ((u + log_ws)[:, None] - cost / eps).logsumexp(
                dim=0
            )
            u = -tau * ((v + log_wt)[None, :] - cost / eps).logsumexp(
                dim=1
            )

    P = ws_wt * (u[:, None] + v[None, :] - cost / eps).exp()

    return P