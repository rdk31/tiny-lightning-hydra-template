from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def make_isb_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    ).numpy()

    betas = np.concatenate(
        [betas[: n_timestep // 2], np.flip(betas[: n_timestep // 2])]
    )
    return betas


class ISBDiffusion(nn.Module):
    def __init__(
        self, n_timestep, temperature: float = 1.0, clip_denoised: bool = True
    ):
        super().__init__()
        self.deterministic = temperature == 0.0
        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.n_timestep = n_timestep

        betas = make_isb_beta_schedule(1000)  # scheduler is initialized with 1000 steps

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("std_fwd", to_torch(std_fwd))
        self.register_buffer("std_bwd", to_torch(std_bwd))
        self.register_buffer("std_sb", to_torch(std_sb))
        self.register_buffer("mu_x0", to_torch(mu_x0))
        self.register_buffer("mu_x1", to_torch(mu_x1))

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """Sample q(x_t | x_0, x_1), i.e. eq 11"""

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt) * self.temperature
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = (
                xt_prev + var.sqrt() * torch.randn_like(xt_prev) * self.temperature
            )

        return xt_prev

    def compute_label(self, step, x0, xt):
        """Eq 12"""
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1.0, 1.0)
        return pred_x0

    def training_losses(self, model, x0, x1):
        step = torch.randint(0, len(self.betas), (x0.shape[0],), device=x0.device)

        xt = self.q_sample(step, x0, x1, ot_ode=self.deterministic)
        label = self.compute_label(step, x0, xt)

        pred = model(xt, step, x_low=x1)

        loss = F.mse_loss(pred, label)

        weights = 1 / self.get_std_fwd(step, xdim=x0.shape[1:])

        return {
            "loss": loss,
            "pred_x0": self.compute_pred_x0(step, xt, pred),
            "weights": weights,
        }

    def p_sample_loop(
        self,
        model,
        x1,
        log_steps=None,
        verbose=False,
    ):
        steps = np.linspace(0, len(self.betas) - 1, self.n_timestep, dtype=int)

        xt = x1.detach().to(x1.device)
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = (
            tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1)
            if verbose
            else pair_steps
        )
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            model_step = torch.full(
                (xt.shape[0],), step, device=x1.device, dtype=torch.long
            )
            out = model(xt, model_step, x_low=x1)
            pred_x0 = self.compute_pred_x0(
                model_step, xt, out, clip_denoise=self.clip_denoised
            )

            xt = self.p_posterior(
                prev_step, step, xt, pred_x0, ot_ode=self.deterministic
            )

        return pred_x0
