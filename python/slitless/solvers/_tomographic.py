"""Tomographic / iterative solvers for slitless spectral imaging."""

import numpy as np
import torch
import eispac
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import convolve

from slitless.config import config
from slitless.forward import (
    forward_op_tomo_3d,
    forward_op_tomo_3d_transpose,
    gauss_pix,
    datacube_generator,
    tomomtx_gen,
)


def gauss_pmf_fitter(line):
    inten = np.sum(line, axis=0)
    line0 = line / inten
    mean = np.sum(np.arange(len(line))[:, None, None] * line0, axis=0)
    std = np.sqrt(
        np.sum((np.arange(len(line)) ** 2)[:, None, None] * line0, axis=0) - mean ** 2
    )
    mean -= len(line) // 2
    return np.stack((inten, mean, std), axis=0)


def gauss_pmf_fitter2(line):
    inten = np.sum(line, axis=0)
    mean = np.zeros_like(inten)
    std = np.ones_like(inten)
    ind0 = np.where(inten <= 0)

    mean = np.sum(np.arange(len(line))[:, None, None] * line, axis=0) / inten
    std = np.sqrt(
        np.sum((np.arange(len(line)) ** 2)[:, None, None] * line, axis=0) / inten
        - mean ** 2
    )
    mean -= len(line) // 2
    inten[ind0] = 0
    std[ind0] = 1.2
    mean[ind0] = 0
    inten = np.clip(inten, 0, 1)
    std = np.clip(std, 0.5, 2.3)
    mean = np.clip(mean, -2, 2)
    inten[np.isnan(inten)] = 0
    mean[np.isnan(mean)] = 0
    std[np.isnan(std)] = 2.3

    return np.stack((inten, mean, std), axis=0)


def gauss_(x, inten, vel, width):
    return inten * gauss_pix(x, vel + len(x) // 2, width)


def gauss_curvefit(dc):
    M, N, R = dc.shape
    param3d = np.zeros((3, N, R))
    for i in range(N):
        for j in range(R):
            par, var = curve_fit(
                gauss_,
                np.arange(M),
                dc[:, i, j],
                p0=[1, 0, 1],
                bounds=((0, -2, 1), (1, 2, 2.3)),
                maxfev=5000,
            )
            param3d[:, i, j] = par
    return param3d


def smart_fit_spectra_joblib(cube, tmplt, wave, errs=None, n_jobs=-1, component=0):
    """
    Fits a reconstructed 3D data cube from SMART using mpfit and an eispac template.

    Args:
        cube: 3D numpy array of shape (lamdim, Y, X)
        tmplt: eispac template object
        wave: 1D array of shape (lamdim,) containing wavelength values
        errs: 3D array of uncertainties. If None, assumes uniform errors.
        n_jobs: Number of parallel jobs
        component: Gaussian component index to extract from the template

    Returns:
        param3d: Stacked array (3, Y, X) containing Intensity, Velocity (km/s),
            and Width (Angstroms).
    """
    import copy
    from joblib import Parallel, delayed
    from slitless.eistools import _worker_fit_chunk
    from eispac.instr import calc_velocity

    safe_data = np.transpose(cube, (1, 2, 0)).astype(np.float64)
    Y, X, lamdim = safe_data.shape

    if errs is None:
        safe_errs = np.ones_like(safe_data)
    else:
        safe_errs = np.transpose(errs, (1, 2, 0)).astype(np.float64)

    if wave.ndim == 1:
        safe_wave = np.tile(wave, (Y, X, 1)).astype(np.float64)
    else:
        safe_wave = np.transpose(wave, (1, 2, 0)).astype(np.float64)

    p_base = tmplt.parinfo
    t_base = tmplt.template

    tasks = []
    for y in range(Y):
        tasks.append(
            (
                safe_wave[y, :, :],
                safe_data[y, :, :],
                safe_errs[y, :, :],
                t_base,
                copy.deepcopy(p_base),
                7,  # min_points
            )
        )

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_worker_fit_chunk)(t) for t in tasks
    )

    full_params = np.stack([r[0] for r in results])
    full_status = np.stack([r[2] for r in results])

    idx_peak = component * 3
    idx_cent = component * 3 + 1
    idx_width = component * 3 + 2

    raw_peak = full_params[:, :, idx_peak]
    raw_cent = full_params[:, :, idx_cent]
    raw_width = full_params[:, :, idx_width]

    intensity = np.sqrt(2 * np.pi) * raw_peak * raw_width
    rest_wave = p_base[idx_cent]["value"]

    velocity = config.speed_of_light * (raw_cent - rest_wave) / rest_wave

    bad_mask = full_status <= 0
    intensity[bad_mask] = 0
    velocity[bad_mask] = 0
    raw_width[bad_mask] = 0

    param3d = np.stack((intensity, velocity, raw_width))
    return param3d


def tomoinv0(meas=None, imager=None, stepsize=1e-1, lam=1e-1, numiter=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gauss_curvefit = gauss_pmf_fitter2
    if imager is not None:
        meas = torch.tensor(imager.meas3dar.copy()).to(device)
    else:
        meas = torch.tensor(meas).to(device)
    NK, N, R = meas.shape
    H = torch.tensor(
        tomomtx_gen((21, N), orders=[0, -1, 1]), dtype=torch.float64
    ).to(device)
    y = meas.view(-1, R)
    gaminv = torch.linalg.inv(H.T @ H + lam * torch.eye(21 * N).to(device))
    Hty = H.T @ y
    r_pinv = gaminv @ Hty
    r_hat = (
        torch.tensor(
            datacube_generator(
                gauss_curvefit(r_pinv.view(21, N, R).cpu().numpy())
            )
        )
        .to(device)
        .view(-1, R)
    )
    for i in tqdm(range(numiter)):
        r_hat = (
            torch.tensor(
                datacube_generator(
                    gauss_curvefit(
                        (r_pinv + lam * gaminv @ r_hat).view(21, N, R).cpu().numpy()
                    )
                )
            )
            .to(device)
            .view(-1, R)
        )
    gauss_curvefit = gauss_pmf_fitter2
    return gauss_curvefit(r_hat.view(21, N, R).cpu().numpy()), []


def tomoinv(
    meas=None,
    imager=None,
    data_step="grad",
    positivity=False,
    proj="gauss",
    init_recon=None,
    stepsize=1e-1,
    lam=1e-1,
    numiter=20,
):
    gauss_curvefit = gauss_pmf_fitter2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if imager is not None:
        meas = torch.tensor(imager.meas3dar.copy()).to(device)
    else:
        meas = torch.tensor(meas).to(device)
    NK, N, R = meas.shape
    H = torch.tensor(
        tomomtx_gen((21, N), orders=[0, -1, 1]), dtype=torch.float64
    ).to(device)
    y = meas.view(-1, R)
    Hty = H.T @ y
    loss = []

    if proj == "gauss":

        def proj0(a):
            return (
                torch.tensor(
                    datacube_generator(
                        gauss_curvefit(a.view(21, N, R).cpu().numpy())
                    )
                )
                .to(device)
                .view(-1, R)
            )

        if positivity:

            def proj(a):
                return proj0((a * (a > 0)).to(a.dtype))
        else:
            proj = proj0
    elif proj == "positivity":
        proj = lambda a: (a * (a > 0)).to(a.dtype)
    if data_step == "inv":
        gaminv = torch.linalg.inv(H.T @ H + lam * torch.eye(21 * N).to(device))
        r_pinv = gaminv @ Hty
        r_hat = proj(r_pinv)
        for i in tqdm(range(numiter)):
            r_hat = proj(r_pinv + lam * gaminv @ r_hat)
            lam *= 0.95
            loss.append(torch.norm(H @ r_hat - y).cpu().numpy())
    elif data_step == "grad":
        if init_recon is not None:
            r_hat = (
                torch.tensor(init_recon, dtype=y.dtype).to(device).view(-1, R)
            )
        else:
            r_hat = torch.zeros((21 * N, R), dtype=torch.float64).to(device)
        for i in tqdm(range(numiter)):
            Hr = H @ r_hat
            r_hat = proj(r_hat - stepsize * (H.T @ Hr - Hty))
            loss.append(torch.norm(H @ r_hat - y).cpu().numpy())
    gauss_curvefit = gauss_pmf_fitter2
    return gauss_curvefit(r_hat.view(21, N, R).cpu().numpy()), loss


def smart(
    meas=None,
    imager=None,
    psi=0.2,
    maxouter=5,
    maxinner=20,
    inf_prior_width=1.38,
    fitter="pmf",
    tmplt=None,
    n_jobs=-1,
):
    if imager is not None:
        meas = imager.meas3dar.copy()
    NK, M, N = meas.shape
    if inf_prior_width is not None:
        infprior = gauss_pix(
            np.outer(np.arange(M), np.ones(M)), M // 2, inf_prior_width
        )
        meas = np.concatenate((meas, infprior[None]), axis=0)
        meas[-1] *= meas[0].mean(axis=0)[None] / meas[-1].mean(axis=0)[None]
        NK += 1

    orders = imager.spectral_orders
    inf = True if inf_prior_width is not None else False
    orders = orders + ["inf"] if inf else orders
    cubes = []
    cors = []
    int0 = meas[0].copy()
    vel0 = np.zeros_like(int0)
    width0 = 1.38 * np.ones_like(vel0)
    cube = datacube_generator(np.stack((int0, vel0, width0), axis=0))
    cubes.append(cube)
    k0 = np.array([0.25, 0.5, 0.25])
    k1 = np.outer(k0, k0)
    kernel = k0[None, None] * k1[:, :, None]

    mtx = tomomtx_gen((M, M), orders=orders)
    mtx_t = np.einsum("ijk->ikj", mtx.reshape(-1, M, M * M))
    mtx_s = (np.sum(mtx_t, axis=2) < 1).astype(int).reshape(-1, M, M)[:, :, :, None]
    mtx_s = np.repeat(mtx_s, N, axis=3)

    for i in range(maxouter):
        print("Outer Iter: {}/{}".format(i + 1, maxouter))
        cube = (cube + cube ** (1 + psi)) * np.sum(cube) / np.sum(
            cube + cube ** (1 + psi)
        )
        cube = convolve(cube, kernel)
        for j in range(maxinner):
            meas2 = (mtx @ cube.reshape(-1, N)).reshape(meas.shape)
            chi = np.mean(((meas - meas2) ** 2) / (meas + 1e-7), axis=(1, 2))
            unconverged = chi > 0.0000000001
            if np.sum(unconverged) == 0:
                continue
            cor = (meas / (meas2 + 1e-5)) ** (2 / (3))
            Cor = np.einsum("ijk,ikm->ijm", mtx_t, cor).reshape(NK, M, M, N)
            Cor[mtx_s == 1] = 1

            Corr = np.prod(Cor[unconverged], axis=0) ** (
                1 / np.sum(unconverged)
            )
            cube *= Corr
        print(f"chi:{chi}")

    if fitter == "mpfit":
        if tmplt is None:
            template_filepath = str(config.template_path)
            tmplt = eispac.read_template(template_filepath)
        lamdim = cube.shape[0]

        wave_cen = (
            imager.mid_wavelength if imager is not None else config.mid_wavelength
        )
        disp_scale = (
            imager.dispersion_scale
            if imager is not None
            else config.dispersion_scale
        )

        wave = wave_cen + disp_scale * (np.arange(lamdim) - lamdim // 2)
        cube = cube / disp_scale * imager.intenscale

        recon = smart_fit_spectra_joblib(cube, tmplt, wave=wave, n_jobs=n_jobs)

        SPEED_OF_LIGHT = config.speed_of_light
        recon[0] /= imager.intenscale
        rest_wave = tmplt.parinfo[1]["value"]
        actual_wave = rest_wave * (1 + recon[1] / SPEED_OF_LIGHT)
        recon[1] = (actual_wave - wave_cen) / disp_scale
        recon[2] = recon[2] / disp_scale
    elif fitter == "pmf":
        recon = gauss_pmf_fitter(cube)

    return recon, cube


def smart2(
    meas=None,
    imager=None,
    psi=0.2,
    maxouter=5,
    maxinner=20,
    prior_weight=1.0,
    fitter="mpfit",
    tmplt=None,
    n_jobs=-1,
    frac1=0.8555,
    frac2=0.0521,
    frac_bg=0.0924,
    cent1=195.11803,
    wid1=0.02907,
    cent2=195.17803,
    wid2=0.02907,
    bg_shape_norm=[0.04762] * 21,
    init_cube=None,
):
    if imager is not None:
        meas = imager.meas3dar.copy()
    NK, M, N = meas.shape
    L = 21

    if tmplt is None and fitter == "mpfit":
        template_filepath = str(config.template_path)
        import os

        if os.path.exists(template_filepath):
            import eispac

            tmplt = eispac.read_template(template_filepath)

    orders = imager.spectral_orders
    orders_list = list(orders)
    if prior_weight > 0:
        orders_list.append("inf")

    meas_list = [meas[k] for k in range(NK)]

    int0 = meas[0].copy()

    if imager is not None:
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale

    vel1_pix_0 = (cent1 - mid_wave) / disp_scale
    wid1_pix_0 = wid1 / disp_scale

    vel2_pix_0 = (cent2 - mid_wave) / disp_scale
    wid2_pix_0 = wid2 / disp_scale

    if init_cube is not None:
        cube = init_cube.copy()
    else:
        v1_0 = vel1_pix_0 * np.ones_like(int0)
        w1_0 = wid1_pix_0 * np.ones_like(int0)
        cube1 = datacube_generator(
            np.stack((int0 * frac1, v1_0, w1_0), axis=0), lamdim=L
        )

        v2_0 = vel2_pix_0 * np.ones_like(int0)
        w2_0 = wid2_pix_0 * np.ones_like(int0)
        cube2 = datacube_generator(
            np.stack((int0 * frac2, v2_0, w2_0), axis=0), lamdim=L
        )

        if bg_shape_norm is None:
            bg_shape_norm = np.ones(L) / L
        bg_cube = (
            np.array(bg_shape_norm)[:, np.newaxis, np.newaxis]
            * (int0 * frac_bg)[np.newaxis, :, :]
        )

        cube = cube1 + cube2 + bg_cube

    if prior_weight > 0:
        infprior = np.sum(cube, axis=1)
        infprior = (
            infprior
            / np.clip(infprior.sum(axis=0), 1e-12, None)
            * meas[0].sum(axis=0)
        )
        meas_list.append(infprior)

    num_projs = len(meas_list)

    weights = np.ones(num_projs)
    if prior_weight > 0:
        weights[-1] = prior_weight

    mtx_list = []
    for order in orders_list:
        mtx_list.append(tomomtx_gen((L, M), orders=[order]))

    mapped_list = []
    mtx_s_list = []
    for k in range(num_projs):
        mapped = mtx_list[k].T @ np.ones((mtx_list[k].shape[0], 1))
        mapped_list.append(mapped)
        mtx_s_list.append((mapped < 0.99).flatten())

    k0 = np.array([0.25, 0.5, 0.25])
    k1 = np.outer(k0, k0)
    kernel = k0[None, None] * k1[:, :, None]

    for i in range(maxouter):
        print("Outer Iter: {}/{}".format(i + 1, maxouter))
        cube = (cube + cube ** (1 + psi)) * np.sum(cube) / np.sum(
            cube + cube ** (1 + psi)
        )
        cube = convolve(cube, kernel, mode="reflect")
        for j in range(maxinner):
            cube_flat = cube.reshape(L * M, N)

            meas2_list = []
            for k in range(num_projs):
                meas2_list.append(mtx_list[k] @ cube_flat)

            chi_list = []
            for k in range(num_projs):
                chi_list.append(
                    np.mean(
                        ((meas_list[k] - meas2_list[k]) ** 2)
                        / (meas_list[k] + 1e-7)
                    )
                )
            chi = np.array(chi_list)
            unconverged = chi > 1e-10
            if np.sum(unconverged) == 0:
                continue

            Cor_list = []
            active_count = 0.0

            for k in range(num_projs):
                cor_k = meas_list[k] / (meas2_list[k] + 1e-2)

                Cor_k_flat = (mtx_list[k].T @ cor_k) / (mapped_list[k] + 1e-2)
                Cor_k = Cor_k_flat.reshape(L, M, N)

                missing_mask = mtx_s_list[k].reshape(L, M, 1)

                fallback = Cor_k[L // 2, :, :][None, :, :]
                Cor_k = np.where(missing_mask, fallback, Cor_k)

                Cor_k = Cor_k ** weights[k]
                Cor_list.append(Cor_k)

                if unconverged[k]:
                    active_count += weights[k]

            Cor = np.stack(Cor_list, axis=0)
            Corr = np.prod(Cor[unconverged], axis=0) ** (
                1.0 / max(active_count, 1.0)
            )
            cube *= Corr

        print(f"chi:{np.mean(chi)}")

    if fitter == "mpfit":
        wave_cen = (
            imager.mid_wavelength if imager is not None else config.mid_wavelength
        )
        disp_scale = (
            imager.dispersion_scale
            if imager is not None
            else config.dispersion_scale
        )
        wave = wave_cen + disp_scale * (np.arange(L) - L // 2)
        cube = cube / disp_scale * imager.intenscale

        recon = smart_fit_spectra_joblib(cube, tmplt, wave=wave, n_jobs=n_jobs)

        SPEED_OF_LIGHT = config.speed_of_light
        recon[0] /= imager.intenscale
        rest_wave = tmplt.parinfo[1]["value"]
        actual_wave = rest_wave * (1 + recon[1] / SPEED_OF_LIGHT)
        recon[1] = (actual_wave - wave_cen) / disp_scale
        recon[2] = recon[2] / disp_scale
    elif fitter == "pmf":
        bg = np.min(cube, axis=0, keepdims=True)
        cube_safe = np.clip(cube - bg, 0.0, None)
        recon = gauss_pmf_fitter2(cube_safe)

    return recon, cube * disp_scale


def smart2_twostage(
    imager=None,
    psi=0.2,
    maxouter=5,
    maxinner=20,
    prior_weight=1.0,
    fitter="mpfit",
    tmplt=None,
    n_jobs=-1,
    frac1=0.8555,
    frac2=0.0521,
    frac_bg=0.0924,
    cent1=195.11803,
    wid1=0.02907,
    cent2=195.17803,
    wid2=0.02907,
    bg_shape_norm=[0.04762] * 21,
    **kwargs,
):
    if prior_weight == 0:
        return smart2(
            imager=imager,
            psi=psi,
            maxouter=maxouter,
            maxinner=maxinner,
            prior_weight=0,
            fitter=fitter,
            tmplt=tmplt,
            n_jobs=n_jobs,
            frac1=frac1,
            frac2=frac2,
            frac_bg=frac_bg,
            cent1=cent1,
            wid1=wid1,
            cent2=cent2,
            wid2=wid2,
            bg_shape_norm=bg_shape_norm,
        )

    recon1, _ = smart2(
        imager=imager,
        psi=psi,
        maxouter=2 * maxouter,
        maxinner=2 * maxinner,
        prior_weight=0,
        fitter=fitter,
        tmplt=tmplt,
        n_jobs=n_jobs,
        frac1=frac1,
        frac2=frac2,
        frac_bg=frac_bg,
        cent1=cent1,
        wid1=wid1,
        cent2=cent2,
        wid2=wid2,
        bg_shape_norm=bg_shape_norm,
    )

    if imager is not None:
        disp_scale = imager.dispersion_scale
        int0 = imager.meas3dar[0].copy()
    else:
        disp_scale = config.dispersion_scale
        int0 = np.zeros((64, 64))

    L = 21
    vel2_offset = (cent2 - cent1) / disp_scale
    wid_pix = wid1 / disp_scale

    v1_0 = recon1[1]
    w1_0 = wid_pix * np.ones_like(int0)
    cube1 = datacube_generator(
        np.stack((int0 * frac1, v1_0, w1_0), axis=0), lamdim=L
    )

    v2_0 = v1_0 + vel2_offset
    w2_0 = wid_pix * np.ones_like(int0)
    cube2 = datacube_generator(
        np.stack((int0 * frac2, v2_0, w2_0), axis=0), lamdim=L
    )

    if bg_shape_norm is None:
        bg_shape_norm = np.ones(L) / L
    bg_cube = (
        np.array(bg_shape_norm)[:, np.newaxis, np.newaxis]
        * (int0 * frac_bg)[np.newaxis, :, :]
    )
    init_cube = cube1 + cube2 + bg_cube

    recon2, cube = smart2(
        imager=imager,
        psi=psi,
        maxouter=maxouter,
        maxinner=maxinner,
        prior_weight=prior_weight,
        fitter=fitter,
        tmplt=tmplt,
        n_jobs=n_jobs,
        init_cube=init_cube,
        cent1=cent1,
        wid1=wid1,
        cent2=cent2,
        wid2=wid2,
        bg_shape_norm=bg_shape_norm,
    )
    return recon2, cube
