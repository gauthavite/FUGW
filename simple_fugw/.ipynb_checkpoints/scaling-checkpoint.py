import torch

def scaling(
    cost, init_duals, uot_params, w_s, w_t, ws_wt, train_params, verbose=True
):
    """
    Scaling algorithm (ie Sinkhorn algorithm).
    Code adapted from Séjourné et al 2020:
    https://github.com/thibsej/unbalanced_gromov_wasserstein.
    """

    log_ws = w_s.log()
    log_wt = w_t.log()
    u, v = init_duals
    rho, eps = uot_params
    niters, tol, eval_freq = train_params

    tau = 1 if torch.isinf(rho) else rho / (rho + eps)

    P_diff = None
    idx = 0
    while (P_diff is None or P_diff >= tol) and (
        niters is None or idx < niters
    ):
        u_prev, v_prev = u.detach().clone(), v.detach().clone()
        if rho == 0:
            v = torch.zeros_like(v)
            u = torch.zeros_like(u)
        else:
            v = -tau * ((u + log_ws)[:, None] - cost / eps).logsumexp(
                dim=0
            )

        else:
            u = -tau * ((v + log_wt)[None, :] - cost / eps).logsumexp(
                dim=1
            )


        if tol is not None and idx % eval_freq == 0:
            P_diff = max(
                (u - u_prev).abs().max(), (v - v_prev).abs().max()
            )

        idx += 1

    P = ws_wt * (u[:, None] + v[None, :] - cost / eps).exp()

    return (u, v), P