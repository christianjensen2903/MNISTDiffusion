import torch
import torch.nn as nn
from ddpm import DDPM
import random


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class NoiseDDPM(DDPM):
    def __init__(self, unet, T, device, n_classes, betas, eta=0, sampling_steps=100):
        super(NoiseDDPM, self).__init__(
            unet=unet, T=T, device=device, n_classes=n_classes
        )
        self.nn_model = unet.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.device = device
        self.sampling_steps = sampling_steps
        self.eta = eta
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.T + 1, (x.shape[0],)).to(
            self.device
        )  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.T, c))

    @torch.no_grad()
    def sample(self, n_sample, size):
        x_i = torch.randn(n_sample, *size).to(
            self.device
        )  # x_T ~ N(0, 1), sample initial noise

        c_i = self.get_ci(n_sample)

        c = self.T // self.sampling_steps

        # sample timestemps
        timesteps = [random.randint(1, self.T) for _ in range(c - 2)]
        timesteps.append(self.T)
        timesteps.append(1)
        timesteps.sort(reverse=True)

        for i in range(c):
            t = timesteps[i]
            t_i_minus_1 = timesteps[i + 1] if i + 1 < c else 1
            t_is = torch.tensor([t / self.T]).repeat(n_sample).to(self.device)
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0

            eps = self.nn_model(x_i, t_is, c_i)
            # x_0 = x_i - eps * self.mab_over_sqrtmab[i]
            # x_0.clamp_(-1, 1)

            # x_i = self.oneover_sqrta[i] * x_0 + self.sqrt_beta_t[i] * z

            x0_t = (x_i - eps * (1 - self.alphabar_t[t]).sqrt()) / self.alphabar_t[
                t
            ].sqrt()
            c1 = (
                self.eta
                * (
                    (1 - self.alphabar_t[t] / self.alphabar_t[t_i_minus_1])
                    * (1 - self.alphabar_t[t_i_minus_1])
                    / (1 - self.alphabar_t[t])
                ).sqrt()
            )
            c2 = ((1 - self.alphabar_t[t_i_minus_1]) - c1**2).sqrt()
            x_i = self.alphabar_t[t_i_minus_1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i, c_i
