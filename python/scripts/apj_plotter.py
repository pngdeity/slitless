import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scienceplots  # noqa: F401  — registers 'science' style with matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from slitless.forward import gauss
from slitless.config import config

plt.style.use(['science', 'no-latex'])

savepath = str(config.data_root.parent / 'figures/apj2024_plots')
SPEED_OF_LIGHT = 299792.458

def get_recon_comp_figures_final(idx_im=13, config='K_3_dbsnr_10', save=False):
    SPEED_OF_LIGHT = 299792.458
    base = str(config.results_root / 'recons')

    # Translate K_X_dbsnr_Y -> actual folder name (K_X_poisson_dbsnr_Y or K_X_None_dbsnr_None)
    K_str    = config[2 : config.index('_dbsnr_')]
    dbsnr_str = config[config.index('_dbsnr_') + 7:]
    noise_seg = 'None' if dbsnr_str == 'None' else 'poisson'
    folder = f'K_{K_str}_{noise_seg}_dbsnr_{dbsnr_str}'

    Rec_unet  = np.load(base + f'2026_05_13__14_19_03_final_runner_UNET/{folder}/Rec.pickle',  allow_pickle=True)
    Rec_map1d = np.load(base + f'2026_05_13__22_56_48_final_runner_1DMAP/{folder}/Rec.pickle', allow_pickle=True)
    Rec_mart  = np.load(base + f'2026_05_14__16_11_47_final_runner_MART/{folder}/Rec.pickle',  allow_pickle=True)

    idx_n = 0
    rec_unet  = Rec_unet.imager.frompix(Rec_unet.recons[idx_im].recon[idx_n],   width_unit='km/s', array=True)
    rec_map1d = Rec_map1d.imager.frompix(Rec_map1d.recons[idx_im].recon[idx_n], width_unit='km/s', array=True)
    rec_mart  = Rec_mart.imager.frompix(Rec_mart.recons[idx_im].recon[idx_n],   width_unit='km/s', array=True)

    true = Rec_unet.sources[idx_im].param3d.copy()
    rest_wl = getattr(Rec_unet.sources[idx_im], 'rest_wavelength', 195.117937907451)
    true[2] = true[2] * SPEED_OF_LIGHT / rest_wl  # Angstroms -> km/s

    recs         = [true, rec_unet, rec_map1d, rec_mart]
    titles       = ['Ground Truth', 'U-Net', '1D MAP', 'MART']
    cmaps        = ['hot', 'seismic', 'plasma']
    param_names  = ['Intensity', 'Velocity', 'Line Width']
    param_units  = ['[erg/cm²/s/sr]', '[km/s]', '[km/s]']

    fig, ax = plt.subplots(3, 4, figsize=(10, 7))
    for row in range(3):
        vmin, vmax = true[row].min(), true[row].max()
        for col in range(4):
            ax[row, col].imshow(recs[col][row], vmin=vmin, vmax=vmax, cmap=cmaps[row])
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            if row == 0:
                ax[row, col].set_title(titles[col], fontsize=15)
        # Parameter name (large) in the upper half, units (small) in the lower half
        ax[row, 0].set_ylabel(param_names[row], fontsize=15)
        ax[row, 0].yaxis.set_label_coords(-0.11, 0.53)
        ax[row, 0].text(-0.06, 0.53, param_units[row],
            transform=ax[row, 0].transAxes,
            fontsize=9, ha='center', va='center', rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.canvas.draw()  # needed so get_position() returns post-layout values

    cbar_width = 0.02
    cbar_pad   = 0.01
    for row in range(3):
        pos = ax[row, -1].get_position()
        cbar_ax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height])
        fig.colorbar(ax[row, 0].images[0], cax=cbar_ax, orientation='vertical')
    plt.show()
    if save:
        fig.savefig(savepath + 'recons_final.png', transparent=True, dpi=300)
    return fig

def loaderr(x='rec_d_10'):
    file = str(Path(__file__).parent / f'basp25figs/{x}.pickle')
    return np.load(file, allow_pickle=True)

def fig2(save=False):
    rec_d_1000 = loaderr('rec_d_1000')
    rec_d_10 = loaderr('rec_d_10')
    rec_d_m10 = loaderr('rec_d_m10')
    rec_u_1000 = loaderr('rec_u_1000')
    rec_u_10 = loaderr('rec_u_10')
    rec_u_m10 = loaderr('rec_u_m10')

    # def get_recon_comp_figures_basp(save=False):
    true = rec_d_10.sources[0].param3d
    meas = rec_d_10.imager.meas3dar_nn[2]
    meas_m10 = rec_d_m10.imager.meas3dar[2]
    meas_10 = rec_d_10.imager.meas3dar[2]
    meas_1000 = rec_d_1000.imager.meas3dar[2]

    rec_d1000 = rec_d_1000.recons[0].recon[0]
    rec_d10 = rec_d_10.recons[0].recon[0]
    rec_dm10 = rec_d_m10.recons[0].recon[0]
    rec_u1000 = rec_u_1000.recons[0].recon[0]
    rec_u10 = rec_u_10.recons[0].recon[0]
    rec_um10 = rec_u_m10.recons[0].recon[0]

    recs = [true, rec_d1000, rec_d10, rec_dm10, rec_u1000, rec_u10, rec_um10]
    meass = [meas, meas_1000, meas_10, meas_m10, meas_1000, meas_10, meas_m10]
    # titles = ['True', 'U-Net', '1D MAP', 'MART']

    fig, ax = plt.subplots(4, 7, figsize=(14,7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1]})
    for i in range(7):
        im = ax[0,i].imshow(recs[i][0], vmin=true[0].min(), vmax=true[0].max(), cmap='hot')
        ax[0,i].axes.get_xaxis().set_visible(False)
        ax[0,i].axes.get_yaxis().set_visible(False)

        im = ax[1,i].imshow(recs[i][1], vmin=true[1].min(), vmax=true[1].max(), cmap='seismic')
        ax[1,i].axes.get_xaxis().set_visible(False)
        ax[1,i].axes.get_yaxis().set_visible(False)

        im = ax[2,i].imshow(recs[i][2], vmin=true[2].min(), vmax=true[2].max(), cmap='plasma')
        ax[2,i].axes.get_xaxis().set_visible(False)
        ax[2,i].axes.get_yaxis().set_visible(False)

        im = ax[3,i].imshow(meass[i], vmin=meass[0].min(), vmax=meass[0].max(), cmap='hot')
        ax[3,i].axes.get_xaxis().set_visible(False)
        ax[3,i].axes.get_yaxis().set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbars
    plt.show()

    if save==True:
        fig.savefig(savepath+'recons.png', transparent=True, dpi=300)
    
# fig2(save=False)

def fig3(save=False):
    rec_d_1000 = loaderr('rec_d_1000')
    rec_d_10 = loaderr('rec_d_10')
    rec_d_m10 = loaderr('rec_d_m10')
    rec_u_1000 = loaderr('rec_u_1000')
    rec_u_10 = loaderr('rec_u_10')
    rec_u_m10 = loaderr('rec_u_m10')

    recs_ = [rec_d_1000, rec_d_10, rec_d_m10, rec_u_1000, rec_u_10, rec_u_m10]


    # def get_recon_comp_figures_basp(save=False):
    true = rec_d_10.sources[0].param3d
    meas = rec_d_10.imager.meas3dar_nn[2]
    meas_m10 = rec_d_m10.imager.meas3dar[2]
    meas_10 = rec_d_10.imager.meas3dar[2]
    meas_1000 = rec_d_1000.imager.meas3dar[2]

    rec_d1000 = rec_d_1000.recons[0].recon[0]
    rec_d10 = rec_d_10.recons[0].recon[0]
    rec_dm10 = rec_d_m10.recons[0].recon[0]
    rec_u1000 = rec_u_1000.recons[0].recon[0]
    rec_u10 = rec_u_10.recons[0].recon[0]
    rec_um10 = rec_u_m10.recons[0].recon[0]

    recs = [true, rec_d1000, rec_d10, rec_dm10, rec_u1000, rec_u10, rec_um10]
    meass = [meas, meas_1000, meas_10, meas_m10, meas_1000, meas_10, meas_m10]

    fig, ax = plt.subplots(4, 7, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1]})

    for i in range(7):  # Loop over the 7 columns
        # First row
        im = ax[0, i].imshow(recs[i][0], cmap='hot')
        ax[0, i].axes.get_xaxis().set_visible(False)
        ax[0, i].axes.get_yaxis().set_visible(False)

        # Second row
        im = ax[1, i].imshow(recs[i][1], cmap='seismic')
        ax[1, i].axes.get_xaxis().set_visible(False)
        ax[1, i].axes.get_yaxis().set_visible(False)

        # Third row
        im = ax[2, i].imshow(recs[i][2], cmap='plasma')
        ax[2, i].axes.get_xaxis().set_visible(False)
        ax[2, i].axes.get_yaxis().set_visible(False)

        # Fourth row
        im = ax[3, i].imshow(meass[i], cmap='hot')
        ax[3, i].axes.get_xaxis().set_visible(False)
        ax[3, i].axes.get_yaxis().set_visible(False)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 1])  # Leave space for colorbars
    plt.show()

    if save==True:
        fig.savefig(savepath+'recons.png', transparent=True, dpi=300)

    return recs_



def get_spectral_comp_figure(idx_im=4, config='K_3_dbsnr_None', pixel=None, save=False):
    base = str(config.results_root / 'recons')

    K_str     = config[2 : config.index('_dbsnr_')]
    dbsnr_str = config[config.index('_dbsnr_') + 7:]
    noise_seg = 'None' if dbsnr_str == 'None' else 'poisson'
    folder    = f'K_{K_str}_{noise_seg}_dbsnr_{dbsnr_str}'

    dir_unet  = base + f'2026_05_13__14_19_03_final_runner_UNET/{folder}/'
    dir_map1d = base + f'2026_05_13__22_56_48_final_runner_1DMAP/{folder}/'
    dir_mart  = base + f'2026_05_14__16_11_47_final_runner_MART/{folder}/'

    Rec_unet  = np.load(dir_unet  + 'Rec.pickle', allow_pickle=True)
    Rec_map1d = np.load(dir_map1d + 'Rec.pickle', allow_pickle=True)
    Rec_mart  = np.load(dir_mart  + 'Rec.pickle', allow_pickle=True)

    true    = Rec_unet.sources[idx_im].param3d
    rest_wl = getattr(Rec_unet.sources[idx_im], 'rest_wavelength', 195.117937907451)

    mid_wl = Rec_unet.imager.mid_wavelength
    disp   = Rec_unet.imager.dispersion_scale
    wave_fine = np.linspace(195.0, 195.24, 500)

    if pixel is not None:
        y_star, x_star = pixel
    else:
        y_star, x_star = np.unravel_index(
            np.abs(true[0] - 0.9 * true[0].max()).argmin(), true[0].shape
        )

    # ── Ground truth ──────────────────────────────────────────────────────────
    int_gt = true[0, y_star, x_star]
    vel_gt = true[1, y_star, x_star]
    wid_gt = true[2, y_star, x_star]
    I_gt   = int_gt * gauss(wave_fine, rest_wl * (1 + vel_gt / SPEED_OF_LIGHT), wid_gt)

    # ── U-Net ─────────────────────────────────────────────────────────────────
    rec_unet_pix = Rec_unet.recons[idx_im].recon[0]
    I_u = (rec_unet_pix[0, y_star, x_star] *
           gauss(wave_fine, mid_wl + rec_unet_pix[1, y_star, x_star] * disp,
                            rec_unet_pix[2, y_star, x_star] * disp))

    # ── 1D MAP: primary Gaussian ──────────────────────────────────────────────
    rec_map1d_pix = Rec_map1d.recons[idx_im].recon[0]
    I1_m = (rec_map1d_pix[0, y_star, x_star] *
            gauss(wave_fine, mid_wl + rec_map1d_pix[1, y_star, x_star] * disp,
                             rec_map1d_pix[2, y_star, x_star] * disp))

    # ── MART: primary Gaussian ────────────────────────────────────────────────
    rec_mart_pix = Rec_mart.recons[idx_im].recon[0]
    I_r = (rec_mart_pix[0, y_star, x_star] *
           gauss(wave_fine, mid_wl + rec_mart_pix[1, y_star, x_star] * disp,
                            rec_mart_pix[2, y_star, x_star] * disp))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(wave_fine, I_gt,  'k-',  lw=2.0, label='Ground Truth')
    ax.plot(wave_fine, I_u,   'C0-', lw=1.5, label='U-Net')
    ax.plot(wave_fine, I_r,   'C1-', lw=1.5, label='MART')
    ax.plot(wave_fine, I1_m,  'C2-', lw=1.5, label='1D MAP')

    ax.set_xlabel('Wavelength [Å]', fontsize=12)
    ax.set_ylabel('Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$ Å$^{-1}$]', fontsize=11)
    ax.axvline(rest_wl, color='red', lw=1.0, ls='--')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(axis='both', which='both', alpha=0.4)
    ax.set_title(f'Reconstructed spectra at reference pixel ({y_star}, {x_star})', fontsize=10)

    def _vel_kms(rec):
        return (mid_wl + rec[1, y_star, x_star] * disp - rest_wl) / rest_wl * SPEED_OF_LIGHT
    def _wid_kms(rec):
        return rec[2, y_star, x_star] * disp * SPEED_OF_LIGHT / rest_wl

    ann = [
        ('GT',    int_gt,                          vel_gt,               wid_gt * SPEED_OF_LIGHT / rest_wl),
        ('U-Net', rec_unet_pix[0, y_star, x_star], _vel_kms(rec_unet_pix),  _wid_kms(rec_unet_pix)),
        ('1D MAP',rec_map1d_pix[0, y_star, x_star],_vel_kms(rec_map1d_pix), _wid_kms(rec_map1d_pix)),
        ('MART',  rec_mart_pix[0, y_star, x_star], _vel_kms(rec_mart_pix),  _wid_kms(rec_mart_pix)),
    ]
    hdr  = f"{'':6s}  {'Int[cgs]':>9s}  {'Vel[km/s]':>9s}  {'Wid[km/s]':>9s}"
    body = '\n'.join(f'{n:6s}  {i:9.2e}  {v:+9.1f}  {w:9.1f}' for n, i, v, w in ann)
    ax.text(0.99, 0.97, hdr + '\n' + body, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(savepath + 'spectra_comp.png', transparent=True, dpi=300)
    return fig


def get_combined_figure(idx_im=4, config='K_3_dbsnr_None', pixel=None, save=False):
    from matplotlib.gridspec import GridSpec

    base = str(config.results_root / 'recons')
    K_str     = config[2 : config.index('_dbsnr_')]
    dbsnr_str = config[config.index('_dbsnr_') + 7:]
    noise_seg = 'None' if dbsnr_str == 'None' else 'poisson'
    folder    = f'K_{K_str}_{noise_seg}_dbsnr_{dbsnr_str}'

    Rec_unet  = np.load(base + f'2026_05_13__14_19_03_final_runner_UNET/{folder}/Rec.pickle',  allow_pickle=True)
    Rec_map1d = np.load(base + f'2026_05_13__22_56_48_final_runner_1DMAP/{folder}/Rec.pickle', allow_pickle=True)
    Rec_mart  = np.load(base + f'2026_05_14__16_11_47_final_runner_MART/{folder}/Rec.pickle',  allow_pickle=True)

    rec_unet  = Rec_unet.imager.frompix(Rec_unet.recons[idx_im].recon[0],   width_unit='km/s', array=True)
    rec_map1d = Rec_map1d.imager.frompix(Rec_map1d.recons[idx_im].recon[0], width_unit='km/s', array=True)
    rec_mart  = Rec_mart.imager.frompix(Rec_mart.recons[idx_im].recon[0],   width_unit='km/s', array=True)

    src     = Rec_unet.sources[idx_im]
    rest_wl = getattr(src, 'rest_wavelength', 195.117937907451)
    true    = src.param3d.copy()
    true[2] = true[2] * SPEED_OF_LIGHT / rest_wl  # Å → km/s for spatial maps

    recs        = [true, rec_unet, rec_map1d, rec_mart]
    titles      = ['Ground Truth', 'U-Net', '1D MAP', 'MART']
    cmaps       = ['hot', 'seismic', 'plasma']
    param_names = ['Intensity', 'Velocity', 'Line Width']
    param_units = ['[erg/cm²/s/sr]', '[km/s]', '[km/s]']

    if pixel is not None:
        y_star, x_star = pixel
    else:
        y_star, x_star = np.unravel_index(
            np.abs(true[0] - 0.9 * true[0].max()).argmin(), true[0].shape
        )

    # Spectral data (pixel-space params; original source for GT, width still in Å)
    mid_wl    = Rec_unet.imager.mid_wavelength
    disp      = Rec_unet.imager.dispersion_scale
    wave_fine = np.linspace(195.0, 195.24, 500)
    p         = src.param3d  # unmodified: [2] in Å

    I_gt = (p[0, y_star, x_star] *
            gauss(wave_fine, rest_wl * (1 + p[1, y_star, x_star] / SPEED_OF_LIGHT),
                             p[2, y_star, x_star]))

    rec_unet_pix  = Rec_unet.recons[idx_im].recon[0]
    rec_map1d_pix = Rec_map1d.recons[idx_im].recon[0]
    rec_mart_pix  = Rec_mart.recons[idx_im].recon[0]

    def _gauss_rec(rec):
        return (rec[0, y_star, x_star] *
                gauss(wave_fine, mid_wl + rec[1, y_star, x_star] * disp,
                                 rec[2, y_star, x_star] * disp))

    I_u  = _gauss_rec(rec_unet_pix)
    I1_m = _gauss_rec(rec_map1d_pix)
    I_r  = _gauss_rec(rec_mart_pix)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 7))
    gs_l = GridSpec(3, 4, figure=fig, left=0.06, right=0.5,
                    hspace=0.08, wspace=0.05, bottom=0.10, top=0.93)
    gs_r = GridSpec(1, 1, figure=fig, left=0.58, right=0.97,
                    bottom=0.10, top=0.93)

    ax      = np.array([[fig.add_subplot(gs_l[r, c]) for c in range(4)] for r in range(3)])
    ax_spec = fig.add_subplot(gs_r[0, 0])

    # ── Spatial maps ──────────────────────────────────────────────────────────
    for row in range(3):
        vmin, vmax   = true[row].min(), true[row].max()
        marker_color = 'white' if row == 2 else 'black'
        for col in range(4):
            ax[row, col].imshow(recs[col][row], vmin=vmin, vmax=vmax, cmap=cmaps[row])
            ax[row, col].scatter(x_star, y_star, marker='x', color=marker_color,
                                 s=60, linewidths=1.5, zorder=5)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            if row == 0:
                ax[row, col].set_title(titles[col], fontsize=15)
        ax[row, 0].set_ylabel(param_names[row], fontsize=15)
        ax[row, 0].yaxis.set_label_coords(-0.11, 0.53)
        ax[row, 0].text(-0.06, 0.53, param_units[row],
                        transform=ax[row, 0].transAxes,
                        fontsize=9, ha='center', va='center', rotation=90)

    # ── Colorbars ─────────────────────────────────────────────────────────────
    fig.canvas.draw()
    for row in range(3):
        pos = ax[row, -1].get_position()
        cbar_ax = fig.add_axes([0.505, pos.y0, 0.015, pos.height])
        fig.colorbar(ax[row, 0].images[0], cax=cbar_ax, orientation='vertical')

    # ── Spectral panel ────────────────────────────────────────────────────────
    ax_spec.plot(wave_fine, I_gt,  'k-',  lw=2.0, label='Ground Truth')
    ax_spec.plot(wave_fine, I_u,   'C0-', lw=1.5, label='U-Net')
    ax_spec.plot(wave_fine, I_r,   'C1-', lw=1.5, label='MART')
    ax_spec.plot(wave_fine, I1_m,  'C2-', lw=1.5, label='1D MAP')
    ax_spec.axvline(rest_wl, color='red', lw=1.0, ls='--')
    ax_spec.set_xlabel('Wavelength [Å]', fontsize=12)
    ax_spec.set_ylabel('Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$ Å$^{-1}$]', fontsize=11)
    ax_spec.legend(fontsize=8, frameon=False, loc='upper left')
    ax_spec.grid(axis='both', which='both', alpha=0.4)

    def _vel_kms(rec):
        return (mid_wl + rec[1, y_star, x_star] * disp - rest_wl) / rest_wl * SPEED_OF_LIGHT
    def _wid_kms(rec):
        return rec[2, y_star, x_star] * disp * SPEED_OF_LIGHT / rest_wl

    ann = [
        ('GT',    p[0, y_star, x_star],             p[1, y_star, x_star],    p[2, y_star, x_star] * SPEED_OF_LIGHT / rest_wl),
        ('U-Net', rec_unet_pix[0, y_star, x_star],  _vel_kms(rec_unet_pix),  _wid_kms(rec_unet_pix)),
        ('1D MAP',rec_map1d_pix[0, y_star, x_star], _vel_kms(rec_map1d_pix), _wid_kms(rec_map1d_pix)),
        ('MART',  rec_mart_pix[0, y_star, x_star],  _vel_kms(rec_mart_pix),  _wid_kms(rec_mart_pix)),
    ]
    hdr  = f"{'':6s}  {'Int[cgs]':>9s}  {'Vel[km/s]':>9s}  {'Wid[km/s]':>9s}"
    body = '\n'.join(f'{n:6s}  {i:9.2e}  {v:+9.1f}  {w:9.1f}' for n, i, v, w in ann)
    ax_spec.text(0.99, 0.97, hdr + '\n' + body, transform=ax_spec.transAxes,
                 fontsize=7, va='top', ha='right', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show()
    if save:
        fig.savefig(savepath + 'recon_spectra_combined.png', transparent=True, dpi=300)
    return fig


def get_bar_chart_figure(save=False):
    K_vals     = [2, 3, 4, 5]
    gamma_vals = ['10', '20', '30', r'$\infty$']
    params     = ['Intensity', 'Velocity', 'Line Width']
    methods    = ['U-Net', 'MART', '1D MAP']

    # APJ-friendly, high-contrast, colorblind-safe palette (Okabe-Ito)
    # colors = {
    #     'U-Net': '#0072B2',  # Strong Blue
    #     'MART': '#D55E00',   # Vermillion / Deep Orange
    #     '1D MAP': '#009E73'  # Teal / Strong Green
    # }
    # scienceplots Bright cycle
    _cyc   = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {'U-Net': _cyc[0], 'MART': _cyc[1], '1D MAP': _cyc[2]}

    with open(savepath + 'bar_data.pickle', 'rb') as f:
        data = pickle.load(f)

    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    x_k    = np.arange(len(K_vals))
    x_g    = np.arange(len(gamma_vals))
    w_rmse = 0.25
    w_bias = 0.12

    configs = [
        {'row': 0, 'sweep': 'K_sweep',     'x_arr': x_k, 'x_labels': K_vals,
         'x_title': 'Number of Projections (K)'},
        {'row': 1, 'sweep': 'Gamma_sweep', 'x_arr': x_g, 'x_labels': gamma_vals,
         'x_title': r'SNR ($\gamma$)'},
    ]
    y_labels = [
        r'Intensity Error ($\mathrm{erg \ cm^{-2} \ s^{-1} \ sr^{-1}}$)',
        r'Velocity Error ($\mathrm{km \ s^{-1}}$)',
        r'Line Width Error ($\mathrm{km \ s^{-1}}$)',
    ]

    for config in configs:
        row, sweep = config['row'], config['sweep']
        for col, param in enumerate(params):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(param, fontsize=15, pad=15, fontweight='bold')
            ax.set_xlabel(config['x_title'], labelpad=2)
            ax.set_ylabel(y_labels[col], labelpad=10)
            for i, method in enumerate(methods):
                offset = (i - 1) * w_rmse
                color  = colors[method]
                ax.bar(config['x_arr'] + offset, data[sweep][method]['RMSE'][col], w_rmse,
                       color=color, alpha=0.25, edgecolor=color, linewidth=1.5, zorder=2)
                ax.bar(config['x_arr'] + offset, data[sweep][method]['Bias'][col], w_bias,
                       color=color, alpha=1.0, zorder=3)
            ax.set_xticks(config['x_arr'])
            ax.set_xticklabels(config['x_labels'])
            ax.axhline(0, color='black', linewidth=1.2, zorder=1)
            ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    method_handles = [Patch(facecolor=colors[m], edgecolor=colors[m], label=m) for m in methods]
    metric_handles = [
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=14,
               solid_capstyle='butt', label='RMSE (Wide Bar)'),
        Line2D([0], [0], color='gray', alpha=1.0, linewidth=5,
               solid_capstyle='butt', label='Bias (Inner Bar)'),
    ]
    leg1 = fig.legend(handles=method_handles, loc='upper center',
                      bbox_to_anchor=(0.35, 0.98), ncol=3, frameon=False, fontsize=13)
    fig.legend(handles=metric_handles, loc='upper center',
               bbox_to_anchor=(0.75, 0.98), ncol=2, frameon=False, fontsize=13)
    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=3.0)
    plt.show()
    if save:
        fig.savefig(savepath + 'bar_charts.png', transparent=True, dpi=300)
    return fig


if __name__ == '__main__':
    savepath = str(config.data_root.parent / 'figures/apj26_post_revision')
    get_recon_comp_figures_final(idx_im=4, config='K_3_dbsnr_None', save=True)
    # get_combined_figure(idx_im=4, config='K_3_dbsnr_None', pixel=(36, 59), save=True)
    get_bar_chart_figure(save=True)
