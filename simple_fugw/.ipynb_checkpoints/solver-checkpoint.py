import torch
import numpy as np

from simple_fugw.utils import _add_dict
from simple_fugw.loss import compute_fugw_loss
from simple_fugw.cost import cost
from simple_fugw.scaling import scaling

class FUGW:
    """
    class computing dense transport plan
    """
    def __init__(
        self,
        n_its_bcd=10,
        nits_uot=1000,
        tol_bcd=None,
        tol_uot=None,
        tol_loss=None,
        eval_bcd=1,
        eval_uot=10,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        
        device='cpu'
    ):
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.pi = None
        self.device = device
        self.nits_bcd = nits_bcd
        self.nits_uot = nits_uot
        self.tol_bcd = tol_bcd
        self.tol_uot = tol_uot
        self.tol_loss = tol_loss
        self.eval_bcd = eval_bcd
        self.eval_uot = eval_uot
        self.loss = {}
        self.loss_val = {}
        self.loss_steps = []
        
    def fit(
        self,
        source_features = None, 
        source_geometry = None, 
        source_weights = None,
        target_features = None,
        target_geometry = None,
        target_weights = None,
        
    ):
        """
        fit function compute P transport plan
        """
        if source_weights is None:
            self.w_s = (
                torch.ones(source_features.shape[1], device=self.device)
                / source_features.shape[1]
            )
        if target_weights is None:
            self.w_t = (
                torch.ones(target_features.shape[1], device=self.device)
                / target_features.shape[1]
            )

        ######## init P #########
        self.ws_wt = self.w_s[:, None] * self.w_t[None, :]
        P = self.ws_wt / self.ws_wt.sum()
        #########################

        print("init ws :", self.w_s)
        print("init wt :", self.w_t)
        print("init ws_wt :", self.ws_wt)
        print("init P :",P)
        
        self.Fs = torch.tensor(source_features.T, device=self.device)
        self.Ft = torch.tensor(target_features.T, device=self.device)
        self.F = torch.cdist(self.Fs, self.Ft, p=2) ** 2
        self.D_s = torch.tensor(source_geometry, device=self.device)
        self.D_t = torch.tensor(target_geometry, device=self.device)

        print("init Ds :", self.D_s)
        print("init Dt :", self.D_t)
        print("init F :", self.F)
        print("init Fs :", self.Fs)
        print("init Ft :", self.Ft)

        G = P
        print("init G :", G)

        duals_P = (
                    torch.zeros_like(self.w_s),
                    torch.zeros_like(self.w_t),
                    )
        duals_G = duals_P

        print("init duals_G :", duals_G)
        print("init duals_P :", duals_P)


        ### Initialize loss
        current_loss = compute_fugw_loss(P, G, self.w_s, self.w_t, self.ws_wt, self.D_s, self.D_t, self.F, self.rho, self.eps, self.alpha)
        current_loss_validation = current_loss

        print("first loss :", current_loss)

        P_diff = None
        loss_diff = None

        loss = _add_dict({}, current_loss)
        loss_val = _add_dict({}, current_loss_validation)
        loss_steps = [0]
        idx = 0
        
        while (
            (P_diff is None or P_diff >= self.tol_bcd)
            and (loss_diff is None or loss_diff >= self.tol_loss)
            and (self.nits_bcd is None or idx < self.nits_bcd)
        ):
            P_prev = P.detach().clone()

            print("P_prev :", P_prev)

            # Update gamma
            mass_P = P.sum()

            print("mass_P :", mass_P)
            
            c_g = cost(P, self.w_s, self.w_t, self.ws_wt, self.D_s, self.D_t, self.F, self.rho, self.eps, self.alpha, transpose=True)

            print("c_g :", c_g)

            new_rho = self.rho * mass_P
            new_eps = mass_P * self.eps

            print("new_eps :", new_eps)
            
            uot_params = (new_rho, new_eps)

            duals_G, G = scaling(
                c_g, duals_G, uot_params, self.w_s, self.w_t, self.ws_wt, train_params=(self.nits_uot, self.tol_uot, self.eval_uot)
            )

            # Rescale gamma
            G = (mass_P / G.sum()).sqrt() * G

            print("G :", G)

            # Update pi
            mass_G = G.sum()
            c_p = cost(G, self.w_s, self.w_t, self.ws_wt, self.D_s, self.D_t, self.F, self.rho, self.eps, self.alpha, transpose=True)

            print("c_p :", c_p)
            
            new_rho = self.rho * mass_G
            new_eps = mass_G * self.eps 
            uot_params = (new_rho, new_eps)

            duals_P, P = scaling(
                c_p, duals_P, uot_params, self.w_s, self.w_t, self.ws_wt, train_params=(self.nits_uot, self.tol_uot, self.eval_uot)
            )

            # Rescale mass
            P = (mass_G / P.sum()).sqrt() * P

            print("P :", P)

            if idx % self.eval_bcd == 0:
                current_loss = compute_fugw_loss(P, G, self.w_s, self.w_t, self.ws_wt, self.D_s, self.D_t, self.F, self.rho, self.eps, self.alpha)

                print("loss :", current_loss)
                
                current_loss_validation = current_loss

                self.loss_steps.append(idx + 1)
                self.loss = _add_dict(self.loss, current_loss)
                self.loss_val = _add_dict(self.loss_val, current_loss_validation)

                print(
                    f"BCD step {idx+1}/{self.nits_bcd}\t"
                    f"FUGW loss:\t{current_loss['total']}\t"
                    f"Validation loss:\t{current_loss_validation['total']}"
                )

                # Update plan difference for potential early stopping
                if self.tol_bcd is not None:
                    P_diff = (P - P_prev).abs().sum().item()

                # Update loss difference for potential early stopping
                if self.tol_loss is not None and len(loss["total"]) >= 2:
                    loss_diff = abs(loss["total"][-2] - loss["total"][-1])

            idx += 1

        if P.isnan().any() or G.isnan().any():
            print("There is NaN in coupling")

        return {
            "P": P,
            "G": G,
            "duals_P": duals_P,
            "duals_G": duals_G,
            "loss": loss,
            "loss_val": loss_val,
            "loss_steps": loss_steps,
        }
        

    
    def transform():
        return None