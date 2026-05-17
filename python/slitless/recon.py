import copy, time, datetime, pickle, os
import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager, datacube_generator, add_noise
from slitless.measure import compare_ssim
from slitless.data_loader import meas_transform as unet_meas_transform, param_inv_transform

from slitless.solvers._tomographic import (
    smart,
    smart2,
    smart2_twostage,
    tomoinv,
    tomoinv0,
    gauss_pmf_fitter,
    gauss_pmf_fitter2,
    smart_fit_spectra_joblib,
)
from slitless.solvers._optimization import (
    grad_descent_solver,
    scipy_solver,
    scipy_solver_parallel,
    scipy_solver_parallel2,
)
from slitless.solvers._neural import nn_solver, diffusion_solver
from slitless.solvers._prior import prior_solver


class Reconstructor:
    def __init__(
        self,
        *,
        imager=None,
        solver=None,
        intenscale=1,
        simulate_meas=True,
        **solver_params,
    ):
        self.imager = copy.deepcopy(imager)
        self.source = self.imager.srpix
        self.solver = solver
        self.intenscale = intenscale
        self.solver_params = solver_params
        self.simulate_meas = simulate_meas
        self.tomo = (
            True
            if hasattr(self.solver, "__name__")
            and self.solver.__name__ in ["smart", "smart2", "tomoinv"]
            else False
        )

    def solve(self, num_realizations=1):
        self.num_realizations = num_realizations
        if self.simulate_meas:
            _ = self.imager.get_measurements(
                dbsnr=self.imager.dbsnr,
                max_count=self.imager.max_count,
                avg_count=self.imager.avg_count,
                noise_model=self.imager.noise_model,
                tomo=self.tomo,
            )
        recons = []
        losses = []
        times = []
        for i in range(num_realizations):
            self.imager.meas3dar = add_noise(
                self.imager.meas3dar_nn,
                dbsnr=self.imager.dbsnr,
                max_count=self.imager.max_count,
                avg_count=self.imager.avg_count,
                noise_model=self.imager.noise_model,
            )

            if self.simulate_meas is False:
                self.meas_transform()
            t0 = time.time()
            recon, loss = self.solver(imager=self.imager, **self.solver_params)
            t1 = time.time()
            if self.simulate_meas is False:
                recon = self.recon_inv_transform(recon)

            times.append(t1 - t0)
            recons.append(recon)
            losses.append(loss)
        self.recons = Recon(
            recon=np.array(recons),
            losses=np.array(losses),
            times=np.array(times),
            imager=self.imager,
            source=self.source,
            intenscale=self.intenscale,
        )
        self.recons.eval()
        self.times = np.array(times)

        return self.recons

    def meas_transform(self):
        if self.solver.__name__ == "nn_solver":
            self.imager.meas3dar = unet_meas_transform(
                self.imager.meas3dar.copy() * self.intenscale
            )

    def recon_inv_transform(self, recon):
        if self.solver.__name__ == "nn_solver":
            return self.imager.topix(
                Source(
                    param3d=param_inv_transform(recon, w_kms=False),
                    pix=False,
                )
            ).param3d
        else:
            return recon


class Reconstructor_Multi:
    def __init__(
        self,
        *,
        meas4dar=None,
        imager=None,
        param4dar=None,
        solver=None,
        pix=None,
        intenscaling=False,
        **solver_params,
    ):
        self.imager = imager
        self.meas4dar = meas4dar
        self.solver = solver
        self.param4dar = param4dar
        self.solver_params = solver_params
        self.intenscaling = intenscaling
        self.pix = pix
        self.simulate_meas = False if self.meas4dar is not None else True

    def solve(self, num_realizations=1):
        self.num_realizations = num_realizations

        self.recons = []
        self.sources = []
        self.times = []

        self.ssim = []
        self.rmse_pix = []
        self.rmse_phy = []
        self.mae_pix = []
        self.mae_phy = []
        self.bias_pix = []
        self.bias_phy = []
        self.ssim_m = []
        self.rmse_pix_m = []
        self.rmse_phy_m = []
        self.mae_pix_m = []
        self.mae_phy_m = []

        for i in range(self.param4dar.shape[0]):
            Sr = Source(param3d=self.param4dar[i], pix=self.pix)

            self.sources.append(Sr)

            intenscale = Sr.param3d[0].max() if self.intenscaling else 1
            self.imager.intenscale = intenscale

            if self.pix == False:
                self.imager.topix(Sr)
            else:
                self.imager.srpix = Sr

            if self.meas4dar is not None:
                self.imager.meas3dar_nn = self.meas4dar[i] / intenscale
                self.imager.meas3dar = self.meas4dar[i] / intenscale

            Rec = Reconstructor(
                imager=self.imager,
                solver=self.solver,
                intenscale=intenscale,
                simulate_meas=self.simulate_meas,
                **self.solver_params,
            )

            recons = Rec.solve(num_realizations=self.num_realizations)
            self.recons.append(recons)
            self.times.append(Rec.times)

            self.ssim.append(recons.ssim)
            self.rmse_pix.append(recons.rmse_pix)
            self.rmse_phy.append(recons.rmse_phy)
            self.mae_pix.append(recons.mae_pix)
            self.mae_phy.append(recons.mae_phy)
            self.bias_pix.append(recons.bias_pix)
            self.bias_phy.append(recons.bias_phy)
            self.ssim_m.append(recons.ssim_m)
            self.rmse_pix_m.append(recons.rmse_pix_m)
            self.rmse_phy_m.append(recons.rmse_phy_m)
            self.mae_pix_m.append(recons.mae_pix_m)
            self.mae_phy_m.append(recons.mae_phy_m)

        self.times = np.array(self.times)
        self.ssim = np.array(self.ssim)
        self.rmse_pix = np.array(self.rmse_pix)
        self.rmse_phy = np.array(self.rmse_phy)
        self.mae_pix = np.array(self.mae_pix)
        self.mae_phy = np.array(self.mae_phy)
        self.bias_pix = np.array(self.bias_pix)
        self.bias_phy = np.array(self.bias_phy)
        self.ssim_m = np.array(self.ssim_m)
        self.rmse_pix_m = np.array(self.rmse_pix_m)
        self.rmse_phy_m = np.array(self.rmse_phy_m)
        self.mae_pix_m = np.array(self.mae_pix_m)
        self.mae_phy_m = np.array(self.mae_phy_m)

        return self.recons


class Recon:
    def __init__(
        self,
        *,
        recon=None,
        losses=None,
        times=None,
        imager=None,
        source=None,
        intenscale=1,
    ):
        self.recon = recon
        self.losses = losses
        self.times = times
        self.imager = imager
        self.source = source
        self.intenscale = intenscale
        self.losses_avg = np.mean(self.losses, axis=0)
        self.times_avg = np.mean(self.times)

    def plot(self, compare=False, index=0, title=""):
        sr = Source(param3d=self.recon[index], pix=True)
        if compare is True:
            assert self.source is not None, "Source is not given!"
            truth = self.source.param3d

            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            plt.suptitle(title)
            i0 = ax[0, 0].imshow(sr.inten, cmap="hot")
            ax[0, 0].set_title("Intensity")
            fig.colorbar(i0, ax=ax[0, 0])
            i0 = ax[0, 1].imshow(sr.vel, cmap="seismic")
            fig.colorbar(i0, ax=ax[0, 1])
            ax[0, 1].set_title("Velocity [pix]")
            i0 = ax[0, 2].imshow(sr.width, cmap="plasma")
            fig.colorbar(i0, ax=ax[0, 2])
            ax[0, 2].set_title("Linewidth [pix]")

            i0 = ax[1, 0].imshow(truth[0], cmap="hot")
            ax[1, 0].set_title("True Intensity")
            fig.colorbar(i0, ax=ax[1, 0])
            i0 = ax[1, 1].imshow(truth[1], cmap="seismic")
            fig.colorbar(i0, ax=ax[1, 1])
            ax[1, 1].set_title("True Velocity [pix]")
            i0 = ax[1, 2].imshow(truth[2], cmap="plasma")
            fig.colorbar(i0, ax=ax[1, 2])
            ax[1, 2].set_title("True Linewidth [pix]")

            plt.tight_layout()
            plt.show()
        else:
            fig, ax = sr.plot(title=title)

        return fig, ax

    def plot_loss(self):
        plt.figure()
        plt.title("Loss vs Iter")
        plt.plot(self.losses_avg, linewidth=2)
        plt.grid(which="both", axis="both")
        plt.show()

    def eval(self):
        truth_pix = self.source.param3d
        truth_pix_mean = (
            np.ones_like(truth_pix)
            * truth_pix.mean(axis=(1, 2))[:, None, None]
        )
        recon_pix = self.recon
        truth_pix = np.repeat(
            truth_pix[np.newaxis, :], len(recon_pix), axis=0
        )
        truth_pix_mean = np.repeat(
            truth_pix_mean[np.newaxis, :], len(recon_pix), axis=0
        )
        truth_phy = self.imager.frompix(
            truth_pix, width_unit="km/s", array=True
        )
        truth_phy_mean = self.imager.frompix(
            truth_pix_mean, width_unit="km/s", array=True
        )
        recon_phy = self.imager.frompix(
            recon_pix, width_unit="km/s", array=True
        )

        self.ssim = compare_ssim(truth=truth_pix, estimate=recon_pix)
        self.rmse_pix = np.sqrt(
            np.mean((recon_pix - truth_pix) ** 2, axis=(-1, -2))
        )
        self.rmse_phy = np.sqrt(
            np.mean((recon_phy - truth_phy) ** 2, axis=(-1, -2))
        )
        self.mae_pix = np.mean(abs(recon_pix - truth_pix), axis=(-1, -2))
        self.mae_phy = np.mean(abs(recon_phy - truth_phy), axis=(-1, -2))
        self.bias_pix = np.mean(recon_pix - truth_pix, axis=(-1, -2))
        self.bias_phy = np.mean(recon_phy - truth_phy, axis=(-1, -2))
        self.ssim_m = compare_ssim(truth=truth_pix, estimate=truth_pix_mean)
        self.rmse_pix_m = np.sqrt(
            np.mean((truth_pix_mean - truth_pix) ** 2, axis=(-1, -2))
        )
        self.rmse_phy_m = np.sqrt(
            np.mean((truth_phy_mean - truth_phy) ** 2, axis=(-1, -2))
        )
        self.mae_pix_m = np.mean(
            abs(truth_pix_mean - truth_pix), axis=(-1, -2)
        )
        self.mae_phy_m = np.mean(
            abs(truth_phy_mean - truth_phy), axis=(-1, -2)
        )


def comparison_test_multi(
    path_data,
    data,
    savepath,
    single_param4dar=False,
    save=False,
    numdetectors=3,
    dbsnr=50,
    noise_model="poisson",
    solver="scipy",
    **kwargs,
):
    if single_param4dar:
        param4dar = np.load(path_data + data)[[0]]
    else:
        param4dar = np.load(path_data + data)
    if len(param4dar.shape) < 4:
        param4dar = param4dar[np.newaxis]

    M = param4dar.shape[-1]

    def inf_priorer(param4dar):
        means = []
        for i in range(len(param4dar)):
            means.append(datacube_generator(param4dar[i]).mean(axis=(1, 2)))
        return np.array(means).mean(axis=0)

    mask = np.array([[(i + j) % 2 for j in range(M)] for i in range(M)])
    mask = np.ones_like(mask)
    Imgr = Imager(
        pixelated=True,
        mask=mask,
        dbsnr=dbsnr,
        max_count=dbsnr ** 2 / 0.9,
        noise_model=noise_model,
        spectral_orders=[0, -1, 1, -2, 2][:numdetectors],
    )

    solver_func = SOLVERS.get(solver)
    if solver_func is None:
        raise ValueError(
            f"Unknown solver: {solver}. Available: {list(SOLVERS.keys())}"
        )

    Rec = Reconstructor_Multi(
        imager=Imgr,
        param4dar=param4dar,
        pix=True,
        solver=solver_func,
        **kwargs,
    )

    recons = Rec.solve(num_realizations=1)
    Rec.recons[0].plot_loss()
    fig, ax = recons[0].plot(compare=True, title=f"{Rec.solver.__name__}")
    print("mask: {}".format(mask[:2, :2]))
    print("Solver: {}".format(Rec.solver.__name__))
    print("Solver Params: {}".format(Rec.solver_params))
    print("Recon Time Avg: {:.2f} s".format(Rec.times.mean()))
    print("RMSE_phy Avg (per Img): {}".format(Rec.rmse_phy.mean(axis=1)))
    print("RMSE_phy Avg: {}".format(Rec.rmse_phy.mean(axis=(0, 1))))
    print("MAE_phy Avg: {}".format(Rec.mae_phy.mean(axis=(0, 1))))
    print("Bias_phy Avg (per Img): {}".format(Rec.bias_phy.mean(axis=1)))
    print("Bias_phy Avg: {}".format(Rec.bias_phy.mean(axis=(0, 1))))
    print("RMSE_pix Avg (per Img): {}".format(Rec.rmse_pix.mean(axis=1)))
    print("RMSE_pix Avg: {}".format(Rec.rmse_pix.mean(axis=(0, 1))))

    if save == True:
        now = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        name = f"{now}_{Rec.solver.__name__}_{data[:-4]}_K_{numdetectors}_{noise_model}_dbsnr_{dbsnr}"
        savedir = savepath + name
        os.mkdir(savedir)
        recon_summary = [
            "############## Recon Parameters ############## \n",
            "mask: {} \n".format(mask[:2, :2]),
            "Solver: {} \n".format(Rec.solver.__name__),
            "Num Detectors: {} \n".format(numdetectors),
            "Noise Model / dbsnr: {} / {} \n".format(noise_model, dbsnr),
            "Num Realizations: {} \n".format(Rec.num_realizations),
            "Solver Params: {} \n".format(Rec.solver_params),
            "Recon Time Avg: {:.2f} s \n".format(Rec.times.mean()),
            "RMSE_phy Avg (per Img): {} \n".format(
                Rec.rmse_phy.mean(axis=1)
            ),
            "RMSE_phy Avg: {} \n".format(Rec.rmse_phy.mean(axis=(0, 1))),
            "MAE_phy Avg: {} \n".format(Rec.mae_phy.mean(axis=(0, 1))),
            "Bias_phy Avg (per Img): {} \n".format(
                Rec.bias_phy.mean(axis=1)
            ),
            "Bias_phy Avg: {} \n".format(Rec.bias_phy.mean(axis=(0, 1))),
            "RMSE_pix Avg (per Img): {} \n".format(
                Rec.rmse_pix.mean(axis=1)
            ),
            "RMSE_pix Avg: {} \n".format(Rec.rmse_pix.mean(axis=(0, 1))),
        ]

        rec_array_pix = []
        rec_array_phy = []
        truth_array_pix = []
        truth_array_phy = []
        for i in range(len(Rec.recons)):
            fig, ax = recons[i].plot(
                compare=True, title=f"{Rec.solver.__name__}"
            )
            fig.savefig(savedir + f"/recon_{i}.png")
            rec_array_pix.append(Rec.recons[i].recon)
            truth_array_pix.append(Rec.sources[i].param3d[None])
        rec_array_pix = np.array(rec_array_pix)
        truth_array_pix = np.array(truth_array_pix)
        rec_array_phy = Rec.imager.frompix(
            rec_array_pix, width_unit="km/s", array=True
        )
        truth_array_phy = Rec.imager.frompix(
            truth_array_pix, width_unit="km/s", array=True
        )
        diff_pix = rec_array_pix - truth_array_pix
        diff_phy = rec_array_phy - truth_array_phy

        def hist_plotter(diff, unit="phy"):
            unitstr = "km/s" if unit == "phy" else "pixels"
            fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.8))
            ax[0].hist(
                diff[:, :, 0].flatten(),
                bins=20,
                edgecolor="Black",
                color="sandybrown",
            )
            ax[0].set_title(
                (
                    "Intensity RMS Error = {:.4f} \n ".format(
                        np.sqrt(np.mean(diff[:, :, 0] ** 2))
                    )
                    + "Intensity Bias = {:.4f}".format(
                        np.mean(diff[:, :, 0])
                    )
                )
            )
            ax[0].set_xlabel("Error")
            ax[0].set_ylabel("Number of Occurances")
            ax[1].hist(
                diff[:, :, 1].flatten(),
                bins=20,
                edgecolor="Black",
                color="sandybrown",
            )
            ax[1].set_title(
                (
                    "Doppler Velocity RMS Error = {:.3f} {} \n".format(
                        np.sqrt(np.mean(diff[:, :, 1] ** 2)), unitstr
                    )
                    + "Doppler Velocity Bias = {:.3f} {}".format(
                        np.mean(diff[:, :, 1]), unitstr
                    )
                )
            )
            ax[1].set_xlabel(f"Error [{unitstr}]")
            ax[1].set_ylabel("Number of Occurances")
            ax[2].hist(
                diff[:, :, 2].flatten(),
                bins=20,
                edgecolor="Black",
                color="sandybrown",
            )
            ax[2].set_title(
                (
                    "Line Width RMS Error = {:.3f} {} \n".format(
                        np.sqrt(np.mean(diff[:, :, 2] ** 2)), unitstr
                    )
                    + "Line Width Bias = {:.3f} {}".format(
                        np.mean(diff[:, :, 2]), unitstr
                    )
                )
            )
            ax[2].set_xlabel(f"Error [{unitstr}]")
            ax[2].set_ylabel("Number of Occurances")
            plt.tight_layout()
            plt.show()
            return fig, ax

        fig, ax = hist_plotter(diff_phy, "phy")
        fig.savefig(savedir + "/error_hist_phy.png")
        fig, ax = hist_plotter(diff_pix, "pix")
        fig.savefig(savedir + "/error_hist_pix.png")

        with open(f"{savedir}/summary.txt", "w") as file:
            for line in recon_summary:
                file.write(line)

        with open(f"{savedir}/Rec.pickle", "wb") as file:
            pickle.dump(Rec, file)

    return Rec


SOLVERS = {
    "scipy": scipy_solver,
    "scipy_parallel": scipy_solver_parallel,
    "scipy_parallel2": scipy_solver_parallel2,
    "smart": smart,
    "smart2": smart2,
    "tomoinv": tomoinv,
    "gd": grad_descent_solver,
    "nn": nn_solver,
    "diffusion": diffusion_solver,
    "prior": prior_solver,
}
