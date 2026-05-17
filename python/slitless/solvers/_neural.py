"""Neural network and diffusion model solvers for slitless spectral imaging."""

import glob
import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from slitless.config import config
from slitless.forward import forward_op_torch
from slitless.evaluate import net_loader, predict


def nn_solver(
    imager=None,
    model_path="2023_01_19__17_18_44_NF_64_BS_4_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_35_dssize_full",
):
    foldpath = (
        glob.glob(str(config.model_dir) + "/" + "*" + model_path + "*")[0] + "/"
    )
    net = net_loader(foldpath)
    net.eval()
    recon = predict(net, imager.meas3dar.copy())

    losses = []
    return recon, losses


def diffusion_solver(
    imager=None,
    model_path="model-10.pt",
    grad_scale=[1, 1, 1],
    num_samples=5,
):
    meas = imager.meas3dar.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = len(imager.spectral_orders)
    model = Unet(
        channels=channels,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True,
    ).to(device)
    data = torch.load(
        str(config.diffusion_model_dir) + "/" + model_path,
        map_location=device,
        weights_only=True,
    )
    adapted_dict = {
        k[6:]: v for k, v in data["model"].items() if k.startswith("model.")
    }
    model.load_state_dict(adapted_dict)
    model.eval()

    def forward_op(x, device=None):
        return forward_op_torch(
            true_intensity=x[:, 0],
            true_doppler=x[:, 1],
            true_linewidth=x[:, 2],
            device=device,
        )

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=1000,
        recon=True,
        measurement=torch.tensor(meas).to(device),
        beta_schedule="cosine",
        grad_scale=torch.tensor(grad_scale).to(device),
        forward_op=forward_op,
        device=device,
        mode="all",
    )

    samples, norms, grad_norms, rmses = diffusion.sample(
        batch_size=num_samples
    )

    recon = samples.mean(dim=0).detach().cpu().numpy()

    return recon, norms
