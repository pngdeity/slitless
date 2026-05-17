"""Gradient and optimization-based solvers for slitless spectral imaging."""

import copy
import numpy as np
import torch
from torch import optim
from tqdm.auto import tqdm
from scipy.optimize import minimize
from joblib import Parallel, delayed

from slitless.config import config
from slitless.forward import forward_op, forward_op_tomo_3d
from slitless.measure import tv_loss


def grad_descent_solver(
    imager=None,
    truth=None,
    OPTIMIZER="ADAM",
    USE_TV_LOSS=True,
    DATA_FIDELITY="L2",
    lam_i=1e-2,
    lam_v=1e-2,
    lam_w=1e-2,
    LR=1e-2,
    maxiter=10000,
    savepath=None,
):
    """Solves for intensity, velocity and width using gradient descent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = []
    meas = imager.meas3dar.copy()

    if type(meas) is not torch.Tensor:
        meas = torch.from_numpy(meas).to(device=device, dtype=torch.float)

    mask = imager.mask.copy()
    if type(mask) is not torch.Tensor:
        mask = torch.from_numpy(mask).to(device=device, dtype=torch.float)

    xh_int = copy.deepcopy(meas[..., 0, :, :])
    xh_int = xh_int.requires_grad_()

    if imager is not None:
        rest_wave = (
            imager.srpix.rest_wavelength
            if hasattr(imager, "srpix")
            else config.wavelength
        )
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = config.wavelength
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale

    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale

    xh_vel = vel_pix_0 * torch.ones_like(xh_int, device=device, dtype=torch.float)
    xh_vel = xh_vel.requires_grad_()
    xh_width = width_pix_0 * torch.ones_like(
        xh_int, device=device, dtype=torch.float
    )
    xh_width = xh_width.requires_grad_()

    if OPTIMIZER.upper() == "ADAM":
        optimizer = optim.Adam([xh_int, xh_vel, xh_width], lr=LR)
    if OPTIMIZER.upper() == "SGD":
        optimizer = optim.SGD([xh_int, xh_vel, xh_width], lr=LR)
    xhs = []
    if truth is not None:
        diffs_vel = []
        diffs_width = []

    for i in tqdm(range(maxiter)):
        optimizer.zero_grad()
        if DATA_FIDELITY == "L1":
            loss = torch.mean(
                abs(meas - imager.forward_op(xh_int, xh_vel, xh_width))
            )
        elif DATA_FIDELITY == "L2":
            loss = torch.mean(
                (meas - imager.forward_op(xh_int, xh_vel, xh_width)) ** 2
            )
        if USE_TV_LOSS:
            loss += (
                lam_i * tv_loss(xh_int)
                + lam_v * tv_loss(xh_vel)
                + lam_w * tv_loss(xh_width)
            )
        loss.backward()

        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        if truth is not None:
            diff_vel = torch.sum((truth[:, 1] - xh_vel) ** 2) / torch.sum(
                truth[:, 1] ** 2
            )
            diffs_vel.append(diff_vel.detach().cpu().numpy())

            diff_width = torch.sum(
                (truth[:, 2] - xh_width) ** 2
            ) / torch.sum(truth[:, 2] ** 2)
            diffs_width.append(diff_width.detach().cpu().numpy())

        if (savepath is not None) & (i > 0) & (i % 10000 == 0):
            xhs0 = (
                torch.stack((xh_int, xh_vel, xh_width), axis=-3)
                .detach()
                .cpu()
                .numpy()
            )
            np.save(savepath + f"recons_{i}.npy", xhs0)

    losses = np.array(losses)
    recon = (
        torch.stack((xh_int, xh_vel, xh_width), axis=-3)
        .detach()
        .cpu()
        .numpy()
    )

    return recon, losses


def scipy_solver(
    imager=None,
    OPTIMIZER="L-BFGS-B",
    DATA_FIDELITY="L2",
    lam_i=5e2,
    lam_v=5e2,
    lam_w=1e0,
    maxiter=10000,
):

    def obj_ls(x, meas=None, mask=None, lam_i=1e1, lam_v=1e1, lam_w=1e1):
        aa, bb = meas.shape[1:]
        intensity, doppler, linewidth = np.reshape(x, (3, aa, bb))
        diff = (
            forward_op(
                intensity,
                doppler,
                linewidth,
                pixelated=imager.pixelated,
                mask=mask,
                spectral_orders=imager.spectral_orders,
            )
            - meas
        )
        regu = (
            lam_v * np.sum(np.diff(doppler, axis=0) ** 2)
            + lam_w * np.sum(np.diff(linewidth, axis=0) ** 2)
            + lam_i * np.sum(np.diff(intensity, axis=0) ** 2)
        )

        if DATA_FIDELITY == "L2":
            return np.sum(diff ** 2) + regu
        elif DATA_FIDELITY == "L1":
            return np.sum(abs(diff)) + regu

    meas = imager.meas3dar.copy()
    mask = imager.mask.copy()
    aa, bb = meas[0].shape

    int0 = meas[0].copy()

    if imager is not None:
        rest_wave = (
            imager.srpix.rest_wavelength
            if hasattr(imager, "srpix")
            else config.wavelength
        )
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = config.wavelength
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale

    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale

    vel0 = vel_pix_0 * np.ones_like(int0)
    width0 = width_pix_0 * np.ones_like(int0)
    x0 = np.stack((int0, vel0, width0), axis=0).flatten()

    rec = np.zeros((3, aa, bb))
    for i in tqdm(range(bb)):
        x0 = np.stack((int0[:, i], vel0[:, i], width0[:, i]), axis=0).flatten()
        recon = minimize(
            obj_ls,
            x0,
            args=(meas[:, :, [i]], mask[:, [i]], lam_i, lam_v, lam_w),
            method=OPTIMIZER,
            options={"disp": False, "maxiter": maxiter},
        )
        rec[:, :, i] = recon.x.reshape(3, aa)

    losses = []
    return rec, losses


def _worker_scipy_col(
    x0,
    meas_slice,
    mask_slice,
    lam_i,
    lam_v,
    lam_w,
    pixelated,
    spectral_orders,
    OPTIMIZER,
    maxiter,
    DATA_FIDELITY,
):

    def obj_ls_local(x, meas, mask, lam_i, lam_v, lam_w):
        aa, bb = meas.shape[1:]

        intensity, doppler, linewidth = np.reshape(x, (3, aa, bb))

        proj_gauss = forward_op(
            intensity,
            doppler,
            linewidth,
            pixelated=pixelated,
            mask=mask,
            spectral_orders=spectral_orders,
        )

        diff = proj_gauss - meas

        regu = (
            lam_v * np.sum(np.diff(doppler, axis=0) ** 2)
            + lam_w * np.sum(np.diff(linewidth, axis=0) ** 2)
            + lam_i * np.sum(np.diff(intensity, axis=0) ** 2)
        )

        if DATA_FIDELITY == "L2":
            return np.sum(diff ** 2) + regu
        elif DATA_FIDELITY == "L1":
            return np.sum(abs(diff)) + regu

    res = minimize(
        obj_ls_local,
        x0,
        args=(meas_slice, mask_slice, lam_i, lam_v, lam_w),
        method=OPTIMIZER,
        options={"disp": False, "maxiter": maxiter},
    )
    return res.x


def scipy_solver_parallel(
    imager=None,
    OPTIMIZER="L-BFGS-B",
    DATA_FIDELITY="L2",
    lam_i=5e2,
    lam_v=5e2,
    lam_w=1e0,
    maxiter=10000,
    n_jobs=-1,
):

    meas = imager.meas3dar.copy()
    mask = imager.mask
    if mask is None:
        mask = np.ones_like(meas[0])
    aa, bb = meas[0].shape

    int0 = meas[0].copy()

    if imager is not None:
        rest_wave = (
            imager.srpix.rest_wavelength
            if hasattr(imager, "srpix")
            else config.wavelength
        )
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = config.wavelength
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale

    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale

    vel0 = vel_pix_0 * np.ones_like(int0)
    width0 = width_pix_0 * np.ones_like(int0)

    tasks = []
    for i in range(bb):
        x0 = np.stack((int0[:, i], vel0[:, i], width0[:, i]), axis=0).flatten()
        tasks.append(
            (
                x0,
                meas[:, :, [i]],
                mask[:, [i]],
                lam_i,
                lam_v,
                lam_w,
                imager.pixelated,
                imager.spectral_orders,
                OPTIMIZER,
                maxiter,
                DATA_FIDELITY,
            )
        )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_worker_scipy_col)(*t) for t in tasks
    )

    rec = np.zeros((3, aa, bb))
    for i, res_x in enumerate(results):
        rec[:, :, i] = res_x.reshape(3, aa)

    losses = []
    return rec, losses


def _worker_scipy_col2(
    x0,
    meas_slice,
    mask_slice,
    lam_i,
    lam_v,
    lam_w,
    pixelated,
    spectral_orders,
    OPTIMIZER,
    maxiter,
    DATA_FIDELITY,
    bg_proj_matrix,
    ratio_i2,
    ratio_bkg,
):

    def obj_ls_local(x, meas, mask, lam_i, lam_v, lam_w):
        aa, bb = meas.shape[1:]

        int1, vel1, wid1, int2, vel2, wid2, bkg = np.reshape(x, (7, aa, bb))

        proj1 = forward_op(
            int1,
            vel1,
            wid1,
            pixelated=pixelated,
            mask=mask,
            spectral_orders=spectral_orders,
        )

        proj2 = forward_op(
            int2,
            vel2,
            wid2,
            pixelated=pixelated,
            mask=mask,
            spectral_orders=spectral_orders,
        )

        proj_bg = np.einsum("oij,jk->oik", bg_proj_matrix, bkg * mask)

        diff = (proj1 + proj2 + proj_bg) - meas

        regu = (
            lam_v * np.sum(np.diff(vel1, axis=0) ** 2)
            + lam_w * np.sum(np.diff(wid1, axis=0) ** 2)
            + lam_i * np.sum(np.diff(int1, axis=0) ** 2)
            + lam_v * np.sum(np.diff(vel2, axis=0) ** 2)
            + lam_w * np.sum(np.diff(wid2, axis=0) ** 2)
            + (lam_i * ratio_i2) * np.sum(np.diff(int2, axis=0) ** 2)
            + (lam_i * ratio_bkg) * np.sum(np.diff(bkg, axis=0) ** 2)
        )

        if DATA_FIDELITY == "L2":
            return np.sum(diff ** 2) + regu
        elif DATA_FIDELITY == "L1":
            return np.sum(abs(diff)) + regu

    res = minimize(
        obj_ls_local,
        x0,
        args=(meas_slice, mask_slice, lam_i, lam_v, lam_w),
        method=OPTIMIZER,
        options={"disp": False, "maxiter": maxiter},
    )
    return res.x


def scipy_solver_parallel2(
    imager=None,
    OPTIMIZER="L-BFGS-B",
    DATA_FIDELITY="L2",
    lam_i=5e2,
    lam_v=5e2,
    lam_w=1e0,
    maxiter=10000,
    n_jobs=-1,
    frac1=0.8620,
    frac2=0.0521,
    frac_bg=0.0860,
    cent1=195.11723,
    wid1=0.02981,
    cent2=195.17723,
    wid2=0.02981,
    bg_shape_norm=[0.04762] * 21,
    return_full=False,
):

    meas = imager.meas3dar.copy()
    mask = imager.mask
    if mask is None:
        mask = np.ones_like(meas[0])
    aa, bb = meas[0].shape

    rest_wave = (
        imager.srpix.rest_wavelength
        if hasattr(imager, "srpix")
        else config.wavelength
    )
    mid_wave = (
        imager.mid_wavelength if imager is not None else config.mid_wavelength
    )
    disp_scale = (
        imager.dispersion_scale
        if imager is not None
        else config.dispersion_scale
    )

    if bg_shape_norm is None:
        bg_shape_norm = np.ones(21) / 21.0
    else:
        bg_shape_norm = np.array(bg_shape_norm)

    I = np.eye(aa)
    bg_cube = bg_shape_norm[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
    bg_proj_matrix = forward_op_tomo_3d(bg_cube, orders=imager.spectral_orders)

    ratio_i2 = (frac1 / max(frac2, 1e-6)) ** 2
    ratio_bkg = (frac1 / max(frac_bg, 1e-6)) ** 2 * 100.0

    int0 = meas[0].copy()
    int1_0 = int0 * frac1
    int2_0 = int0 * frac2
    bkg_0 = int0 * frac_bg

    vel1_pix_0 = (cent1 - mid_wave) / disp_scale
    wid1_pix_0 = wid1 / disp_scale

    vel2_pix_0 = (cent2 - mid_wave) / disp_scale
    wid2_pix_0 = wid2 / disp_scale

    v1_0 = vel1_pix_0 * np.ones_like(int0)
    w1_0 = wid1_pix_0 * np.ones_like(int0)

    v2_0 = vel2_pix_0 * np.ones_like(int0)
    w2_0 = wid2_pix_0 * np.ones_like(int0)

    tasks = []
    for i in range(bb):
        x0 = np.stack(
            (
                int1_0[:, i],
                v1_0[:, i],
                w1_0[:, i],
                int2_0[:, i],
                v2_0[:, i],
                w2_0[:, i],
                bkg_0[:, i],
            ),
            axis=0,
        ).flatten()
        tasks.append(
            (
                x0,
                meas[:, :, [i]],
                mask[:, [i]],
                lam_i,
                lam_v,
                lam_w,
                imager.pixelated,
                imager.spectral_orders,
                OPTIMIZER,
                maxiter,
                DATA_FIDELITY,
                bg_proj_matrix,
                ratio_i2,
                ratio_bkg,
            )
        )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_worker_scipy_col2)(*t) for t in tasks
    )

    n_out = 7 if return_full else 3
    rec = np.zeros((n_out, aa, bb))
    for i, res_x in enumerate(results):
        rec[:, :, i] = res_x.reshape(7, aa)[:n_out, :]

    losses = []
    return rec, losses
