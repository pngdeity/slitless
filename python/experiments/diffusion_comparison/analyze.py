import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from slitless.forward import gauss, Source

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR  = '/home/kamo/resources/slitless/python/experiments/diffusion_comparison/output/'
SPEED_OF_LIGHT = 299792.458

CONFIGS = [
    (None, None),
    (20,  'gaussian'),
    (30,  'gaussian'),
]

SNR_LABELS = {
    (None, None):      r'$\infty$',
    (20,  'gaussian'): '20',
    (30,  'gaussian'): '30',
}

METHOD_KEYS   = ['unet', 'dps', 'cond']
METHOD_NAMES  = {'unet': 'U-Net', 'dps': 'DPS', 'cond': 'CondDiff'}
METHOD_COLORS = {'unet': '#0072B2', 'dps': '#D55E00', 'cond': '#009E73'}

# ==============================================================================
# HELPERS
# ==============================================================================
def pkl_path(method, dbsnr, noise_model):
    return os.path.join(OUTPUT_DIR, f'Rec_{method}_dbsnr_{dbsnr}_{noise_model}.pickle')


def load_rec(method, dbsnr, noise_model):
    return np.load(pkl_path(method, dbsnr, noise_model), allow_pickle=True)


def load_all():
    recs = {}
    for method in METHOD_KEYS:
        recs[method] = {}
        for cfg in CONFIGS:
            dbsnr, nm = cfg
            p = pkl_path(method, dbsnr, nm)
            if os.path.exists(p):
                recs[method][cfg] = np.load(p, allow_pickle=True)
            else:
                print(f'  WARNING: missing {p}')
    return recs


def extract_metrics(Rec):
    rmse = Rec.rmse_phy.mean(axis=(0, 1))
    bias = Rec.bias_phy.mean(axis=(0, 1))
    return rmse, bias


# ==============================================================================
# FIGURE 1 — spatial reconstruction comparison (like get_recon_comp_figures_final)
# ==============================================================================
def plot_recon_comparison(recs, cfg=(None, None), idx_im=4, save=True):
    dbsnr, nm = cfg
    Rec_unet = recs['unet'][cfg]
    Rec_dps  = recs['dps'][cfg]
    Rec_cond = recs['cond'][cfg]

    idx_n   = 0
    rest_wl = getattr(Rec_unet.sources[idx_im], 'rest_wavelength', 195.117937907451)

    rec_unet = Rec_unet.imager.frompix(Rec_unet.recons[idx_im].recon[idx_n], width_unit='km/s', array=True)
    rec_dps  = Rec_dps.imager.frompix(Rec_dps.recons[idx_im].recon[idx_n],   width_unit='km/s', array=True)
    rec_cond = Rec_cond.imager.frompix(Rec_cond.recons[idx_im].recon[idx_n], width_unit='km/s', array=True)

    true = Rec_unet.sources[idx_im].param3d.copy()
    true[2] = true[2] * SPEED_OF_LIGHT / rest_wl  # Å → km/s

    recs_plot   = [true, rec_unet, rec_dps, rec_cond]
    titles      = ['Ground Truth', 'U-Net', 'DPS', 'CondDiff']
    cmaps       = ['hot', 'seismic', 'plasma']
    param_names = ['Intensity', 'Velocity', 'Line Width']
    param_units = [r'[erg/cm²/s/sr]', '[km/s]', '[km/s]']

    fig, ax = plt.subplots(3, 4, figsize=(11, 7))
    for row in range(3):
        vmin, vmax = true[row].min(), true[row].max()
        for col in range(4):
            ax[row, col].imshow(recs_plot[col][row], vmin=vmin, vmax=vmax, cmap=cmaps[row])
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            if row == 0:
                ax[row, col].set_title(titles[col], fontsize=13)
        ax[row, 0].set_ylabel(param_names[row], fontsize=13)
        ax[row, 0].yaxis.set_label_coords(-0.11, 0.53)
        ax[row, 0].text(-0.06, 0.53, param_units[row],
                        transform=ax[row, 0].transAxes,
                        fontsize=8, ha='center', va='center', rotation=90)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fig.canvas.draw()

    cbar_width, cbar_pad = 0.018, 0.01
    for row in range(3):
        pos     = ax[row, -1].get_position()
        cbar_ax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height])
        fig.colorbar(ax[row, 0].images[0], cax=cbar_ax, orientation='vertical')

    snr_str = SNR_LABELS[cfg]
    fig.suptitle(f'K=3, SNR={snr_str} dB', fontsize=12, y=1.01)
    if save:
        fname = os.path.join(OUTPUT_DIR, f'recon_comp_dbsnr_{dbsnr}_{nm}.png')
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'  saved -> {fname}')
    plt.close(fig)
    return fig


# ==============================================================================
# FIGURE 2 — joint scatter (true vs estimated) like get_joint_scatter_figure
# ==============================================================================
def plot_joint_scatter(recs, cfg=(None, None), save=True):
    dbsnr, nm = cfg

    def collect(method):
        Rec = recs[method][cfg]
        truths, estimates = [], []
        for idx_im in range(len(Rec.sources)):
            rest_wl = getattr(Rec.sources[idx_im], 'rest_wavelength', 195.117937907451)
            true = Rec.sources[idx_im].param3d.copy()
            true[2] = true[2] * SPEED_OF_LIGHT / rest_wl
            rec  = Rec.imager.frompix(Rec.recons[idx_im].recon[0], width_unit='km/s', array=True)
            truths.append(true.reshape(3, -1))
            estimates.append(rec.reshape(3, -1))
        return np.concatenate(truths, axis=1), np.concatenate(estimates, axis=1)

    methods_data = {m: collect(m) for m in METHOD_KEYS}

    param_names = ['Intensity', 'Velocity', 'Line Width']
    param_units = [r'erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$', r'km s$^{-1}$', r'km s$^{-1}$']

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for col, method in enumerate(METHOD_KEYS):
        truth, rec = methods_data[method]
        for row in range(3):
            ax  = axes[row, col]
            t   = truth[row]
            r   = rec[row]
            lo  = np.percentile(t, 0.1)
            hi  = np.percentile(t, 99.9)
            hb  = ax.hexbin(t, r, gridsize=70, mincnt=1, cmap='viridis',
                            extent=[lo, hi, lo, hi])
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            r_val = np.corrcoef(t, r)[0, 1]
            ax.text(0.04, 0.95, f'r = {r_val:.3f}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            if row == 0:
                ax.set_title(METHOD_NAMES[method], fontsize=13, fontweight='bold', pad=8)
            if col == 0:
                ax.annotate(param_names[row], xy=(-0.38, 0.5), xycoords='axes fraction',
                            fontsize=13, fontweight='bold', ha='center', va='center',
                            rotation=90, annotation_clip=False)
            ax.set_ylabel(f'Estimated [{param_units[row]}]', fontsize=9)
            ax.set_xlabel(f'True [{param_units[row]}]', fontsize=9)
            fig.colorbar(hb, ax=ax, pad=0.02)

    snr_str = SNR_LABELS[cfg]
    plt.suptitle(f'True vs Estimated — K=3, SNR={snr_str} dB', fontsize=12, y=1.01)
    plt.tight_layout()
    if save:
        fname = os.path.join(OUTPUT_DIR, f'scatter_dbsnr_{dbsnr}_{nm}.png')
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'  saved -> {fname}')
    plt.close(fig)
    return fig


# ==============================================================================
# FIGURE 3 — bar chart (RMSE + bias across SNR sweep)
# ==============================================================================
def plot_bar_chart(results_summary, save=True):
    snr_vals   = [(None, None), (30, 'gaussian'), (20, 'gaussian')]
    x_labels   = [SNR_LABELS[c] for c in snr_vals]
    params     = ['Intensity', 'Velocity', 'Line Width']

    fig, axes  = plt.subplots(1, 3, figsize=(14, 5))
    x_arr  = np.arange(len(snr_vals))
    w_rmse = 0.22
    w_bias = 0.10

    y_labels = [
        r'Intensity Error (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)',
        r'Velocity Error (km s$^{-1}$)',
        r'Line Width Error (km s$^{-1}$)',
    ]

    for col, param in enumerate(params):
        ax = axes[col]
        ax.set_title(param, fontsize=13, fontweight='bold')
        ax.set_ylabel(y_labels[col])
        ax.set_xlabel(r'SNR ($\gamma$)')
        for i, method in enumerate(METHOD_KEYS):
            offset = (i - 1) * w_rmse
            color  = METHOD_COLORS[method]
            rmse_vals = [results_summary[c][method]['rmse'][col] for c in snr_vals]
            bias_vals = [results_summary[c][method]['bias'][col] for c in snr_vals]
            ax.bar(x_arr + offset, rmse_vals, w_rmse,
                   color=color, alpha=0.25, edgecolor=color, linewidth=1.5,
                   zorder=2, label=METHOD_NAMES[method] if col == 0 else '')
            ax.bar(x_arr + offset, bias_vals, w_bias,
                   color=color, alpha=1.0, zorder=3)
        ax.set_xticks(x_arr)
        ax.set_xticklabels(x_labels)
        ax.axhline(0, color='black', linewidth=1.2, zorder=1)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    method_handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor=METHOD_COLORS[m],
                            label=METHOD_NAMES[m]) for m in METHOD_KEYS]
    metric_handles = [
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=12, solid_capstyle='butt',
               label='RMSE'),
        Line2D([0], [0], color='gray', alpha=1.0, linewidth=5, solid_capstyle='butt',
               label='Bias'),
    ]
    fig.legend(handles=method_handles + metric_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fname = os.path.join(OUTPUT_DIR, 'bar_charts.png')
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'  saved -> {fname}')
    plt.close(fig)
    return fig


# ==============================================================================
# TABLE — RMSE / Bias printed as LaTeX and plain text
# ==============================================================================
def print_and_save_table(results_summary):
    lines = []

    header = (
        '\\hline\n'
        '\\textbf{$\\gamma$} & \\textbf{Method} & '
        '\\multicolumn{2}{c}{\\textbf{Intensity}} & '
        '\\multicolumn{2}{c}{\\textbf{Velocity}} & '
        '\\multicolumn{2}{c}{\\textbf{Line Width}} \\\\\n'
        '\\hline\n'
        ' &  & RMSE & Bias & RMSE & Bias & RMSE & Bias \\\\\n'
        ' &  & \\multicolumn{2}{c}{(erg/cm²/s/sr)} & '
        '\\multicolumn{2}{c}{(km/s)} & \\multicolumn{2}{c}{(km/s)} \\\\\n'
        '\\hline\\hline'
    )
    lines.append(header)

    for cfg in CONFIGS:
        snr_label = SNR_LABELS[cfg]
        for j, method in enumerate(METHOD_KEYS):
            res  = results_summary[cfg][method]
            rmse, bias = res['rmse'], res['bias']
            prefix = f'\\multirow{{3}}{{*}}{{{snr_label}}}' if j == 0 else '{}'
            lines.append(
                f'{prefix} & {METHOD_NAMES[method]} & '
                f'{rmse[0]:.1f} & {bias[0]:.1f} & '
                f'{rmse[1]:.3f} & {bias[1]:.3f} & '
                f'{rmse[2]:.3f} & {bias[2]:.3f} \\\\'
            )
        lines.append('\\hline')

    latex_text = '\n'.join(lines)
    print('\n' + '='*70)
    print(' LATEX TABLE')
    print('='*70)
    print(latex_text)

    tex_path = os.path.join(OUTPUT_DIR, 'table.tex')
    with open(tex_path, 'w') as f:
        f.write(latex_text + '\n')
    print(f'  saved -> {tex_path}')

    # plain-text summary
    txt_path = os.path.join(OUTPUT_DIR, 'metrics_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('Diffusion Comparison — RMSE / Bias (K=3)\n')
        f.write('='*60 + '\n\n')
        for cfg in CONFIGS:
            snr_str = SNR_LABELS[cfg]
            f.write(f'SNR = {snr_str} dB\n')
            f.write('-'*40 + '\n')
            for method in METHOD_KEYS:
                res = results_summary[cfg][method]
                rmse, bias = res['rmse'], res['bias']
                f.write(f'  {METHOD_NAMES[method]}:\n')
                f.write(f'    Int  RMSE={rmse[0]:.2f}  Bias={bias[0]:.2f}\n')
                f.write(f'    Vel  RMSE={rmse[1]:.3f}  Bias={bias[1]:.3f} km/s\n')
                f.write(f'    Wid  RMSE={rmse[2]:.3f}  Bias={bias[2]:.3f} km/s\n')
            f.write('\n')
    print(f'  saved -> {txt_path}')


# ==============================================================================
# MAIN
# ==============================================================================
def analyze_all():
    print('Loading pickle files...')
    recs = load_all()

    # Build results summary from the loaded Rec objects
    results_summary = {}
    for cfg in CONFIGS:
        results_summary[cfg] = {}
        for method in METHOD_KEYS:
            if cfg in recs[method]:
                rmse, bias = extract_metrics(recs[method][cfg])
                results_summary[cfg][method] = {
                    'rmse': rmse, 'bias': bias,
                    'time': recs[method][cfg].times.mean(),
                }

    print('\nGenerating spatial comparison figures...')
    for cfg in CONFIGS:
        try:
            plot_recon_comparison(recs, cfg=cfg, idx_im=4, save=True)
        except Exception as e:
            print(f'  skipped recon_comp for {cfg}: {e}')

    print('\nGenerating scatter figures...')
    for cfg in CONFIGS:
        try:
            plot_joint_scatter(recs, cfg=cfg, save=True)
        except Exception as e:
            print(f'  skipped scatter for {cfg}: {e}')

    print('\nGenerating bar chart...')
    try:
        plot_bar_chart(results_summary, save=True)
    except Exception as e:
        print(f'  skipped bar chart: {e}')

    print('\nGenerating tables...')
    print_and_save_table(results_summary)

    print('\nDone. All outputs in:', OUTPUT_DIR)


if __name__ == '__main__':
    analyze_all()
