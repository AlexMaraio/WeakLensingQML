"""
File to compare the numerical Fisher matrix as produced from my conjugate-gradient method to that obtained using the
"analytic" approach using the ECLIPSE formalism
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


# Enable LaTeX support in plots
plt.rcParams['text.usetex'] = True


if __name__ == '__main__':
    # Where to save the figures
    folder_path = '/home/maraio/Codes/WeakLensingQML_Git_Cuillin/data/Plots'

    # The N_side parameter for the Fisher matrix under consideration
    n_side = 64

    # Create an ell range that goes from ell=2 to ell=3*nside - 1
    full_ells = np.arange(2, 3 * n_side)
    n_ell_full = len(full_ells)

    # We only want to plot ells from ell=2 to ell=2*nside
    ells = np.arange(2, 2 * n_side + 1)
    n_ell = len(ells)

    # Load in the "analytic" Fisher matrix, as computed using ECLIPSE formalism
    analytic_fisher = np.load(f'/home/maraio/Codes/WeakLensingQML/data/Fishers/F_N{n_side}.npy')
    analytic_fisher = (analytic_fisher.copy() + analytic_fisher.copy().T) / 2

    # Load in our low-resolution numerical conjugate-gradient Fisher
    numeric_fisher_5maps = np.loadtxt(
        f'/home/maraio/Codes/WeakLensingQML/data/numerical_fishers/ConjGrad_Fisher_N{n_side}_nummaps5_noise30_EB_nostars_new.dat')

    # Load in the high-resolution Fisher
    numeric_fisher_100maps = np.loadtxt(
        f'/home/maraio/Codes/WeakLensingQML/data/numerical_fishers/ConjGrad_Fisher_N{n_side}_nummaps100_noise30_EB_nostars_new.dat')

    # Take the ratio of numerical to analytic Fisher for our two matrices
    ratio = numeric_fisher_5maps / analytic_fisher
    ratio2 = numeric_fisher_100maps / analytic_fisher

    # Extract the EE-EE and BB-BB parts
    ratio_EE_EE = ratio[0 * n_ell_full: 0 * n_ell_full + n_ell, 0 * n_ell_full: 0 * n_ell_full + n_ell]
    ratio_EE_BB = ratio[2 * n_ell_full: 2 * n_ell_full + n_ell, 0 * n_ell_full: 0 * n_ell_full + n_ell]
    ratio_BB_BB = ratio[2 * n_ell_full: 2 * n_ell_full + n_ell, 2 * n_ell_full: 2 * n_ell_full + n_ell]

    ratio_EE_EE_2 = ratio2[0 * n_ell_full: 0 * n_ell_full + n_ell, 0 * n_ell_full: 0 * n_ell_full + n_ell]
    ratio_EE_BB_2 = ratio2[0 * n_ell_full: 0 * n_ell_full + n_ell, 2 * n_ell_full: 2 * n_ell_full + n_ell]
    ratio_BB_BB_2 = ratio2[2 * n_ell_full: 2 * n_ell_full + n_ell, 2 * n_ell_full: 2 * n_ell_full + n_ell]

    # Plot the diagonal of this ratio on two sub-plots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    ax1.plot(ells, np.abs(np.diag(ratio_EE_EE)), lw=2, c='cornflowerblue', label='$EE$-$EE$')
    ax1.plot(ells[1:], np.abs(np.diag(ratio_EE_BB, k=1)), lw=2, c='orange', label='$EE$-$BB$')
    ax1.plot(ells, np.abs(np.diag(ratio_BB_BB)), lw=2, c='mediumseagreen', label='$BB$-$BB$')

    ax2.plot(ells, np.abs(np.diag(ratio_EE_EE_2)), lw=2, c='cornflowerblue', label='$EE$-$EE$')
    ax2.plot(ells[1:], np.abs(np.diag(ratio_EE_BB_2, k=1)), lw=2, c='orange', label='$EE$-$BB$')
    ax2.plot(ells, np.abs(np.diag(ratio_BB_BB_2)), lw=2, c='mediumseagreen', label='$BB$-$BB$')

    ax2.set_xlabel(r'$\ell$')
    ax1.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell} / \mathbf{F}_{\ell \ell}$')
    ax2.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell} / \mathbf{F}_{\ell \ell}$')

    ax1.set_title('5 map average')
    ax2.set_title('100 map average')

    ax1.legend(ncol=3)
    ax1.set_xlim(left=0, right=130)
    fig.tight_layout()
    fig.savefig(f'{folder_path}/Numerical_to_analytic_Fisher_N64_diag_EB_new.pdf')

    # Plot the diagonal of this ratio on two sub-plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 5), facecolor='#013035')

    k = 0

    ax1.plot(ells[:], np.abs(np.diag(ratio_EE_EE, k=k)), lw=2, c='dodgerblue', label=f'5 maps')
    ax1.plot(ells[:], np.abs(np.diag(ratio_EE_EE_2, k=k)), lw=2, c='orange', label=f'100 maps')

    ax2.plot(ells[:-(1+k)], np.abs(np.diag(ratio_EE_BB, k=1+k)), lw=2, c='dodgerblue')
    ax2.plot(ells[:-(1+k)], np.abs(np.diag(ratio_EE_BB_2, k=1+k)), lw=2, c='orange')

    ax3.plot(ells[:], np.abs(np.diag(ratio_BB_BB, k=k)), lw=2, c='dodgerblue')
    ax3.plot(ells[:], np.abs(np.diag(ratio_BB_BB_2, k=k)), lw=2, c='orange')

    ax3.set_xlabel(r'$\ell$', color='white')
    ax1.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell}^{EE\,EE} / \mathbf{F}_{\ell \ell}^{EE\,EE}$', color='white')
    ax2.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell}^{EE\,BB} / \mathbf{F}_{\ell \ell}^{EE\,BB}$', color='white')
    ax3.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell}^{BB\,BB} / \mathbf{F}_{\ell \ell}^{BB\,BB}$', color='white')

    ax1.legend(ncol=3)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.tick_params(axis='both', colors='white')
    ax2.tick_params(axis='both', colors='white')
    ax3.tick_params(axis='both', colors='white')

    ax1.set_xlim(left=0, right=130)

    fig.tight_layout()
    fig.savefig(f'{folder_path}/Numerical_to_analytic_Fisher_N64_diag_EB2_presentation.pdf')

    # Paper version of the above plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 7))

    k = 0

    ax1.plot(ells[:], np.abs(np.diag(ratio_EE_EE, k=k)), lw=2, c='dodgerblue', label=f'5 maps')
    ax1.plot(ells[:], np.abs(np.diag(ratio_EE_EE_2, k=k)), lw=2, c='orange', label=f'100 maps')

    ax2.plot(ells[:-(1+k)], np.abs(np.diag(ratio_EE_BB, k=1+k)), lw=2, c='dodgerblue')
    ax2.plot(ells[:-(1+k)], np.abs(np.diag(ratio_EE_BB_2, k=1+k)), lw=2, c='orange')

    ax3.plot(ells[:], np.abs(np.diag(ratio_BB_BB, k=k)), lw=2, c='dodgerblue')
    ax3.plot(ells[:], np.abs(np.diag(ratio_BB_BB_2, k=k)), lw=2, c='orange')

    ax3.set_xlabel(r'$\ell$', size=16)
    ax1.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell}^{EE\,EE} / \mathbf{F}_{\ell \ell}^{EE\,EE}$', size=16)
    ax2.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell}^{EE\,BB} / \mathbf{F}_{\ell \ell}^{EE\,BB}$', size=16)
    ax3.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell}^{BB\,BB} / \mathbf{F}_{\ell \ell}^{BB\,BB}$', size=16)

    ax1.legend(ncol=3, fontsize=13)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:.2f}$'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:.2f}$'))
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:.2f}$'))

    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)

    ax1.set_xlim(left=0, right=130)
    fig.tight_layout()
    fig.subplots_adjust(top=0.975, hspace=0.175)
    fig.savefig(f'{folder_path}/Numerical_to_analytic_Fisher_N64_diag.pdf')

    # Plot on a grid
    fig, axs = plt.subplots(3, 4, sharex='col', sharey='row', figsize=(8, 8))

    ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = axs

    k = 0
    ax1.plot(ells[:], np.abs(np.diag(ratio_EE_EE, k=k)), lw=2, c='cornflowerblue')
    ax1.plot(ells[:], np.abs(np.diag(ratio_EE_EE_2, k=k)), lw=2, c='orange')

    ax5.plot(ells[:-(1+k)], np.abs(np.diag(ratio_EE_BB, k=1+k)), lw=2, c='cornflowerblue')
    ax5.plot(ells[:-(1+k)], np.abs(np.diag(ratio_EE_BB_2, k=1+k)), lw=2, c='orange')

    ax9.plot(ells[:], np.abs(np.diag(ratio_BB_BB, k=k)), lw=2, c='cornflowerblue')
    ax9.plot(ells[:], np.abs(np.diag(ratio_BB_BB_2, k=k)), lw=2, c='orange')

    k = 2
    ax2.plot(ells[:-k], np.abs(np.diag(ratio_EE_EE, k=k)), lw=2, c='cornflowerblue')
    ax2.plot(ells[:-k], np.abs(np.diag(ratio_EE_EE_2, k=k)), lw=2, c='orange')

    ax6.plot(ells[:-(1 + k)], np.abs(np.diag(ratio_EE_BB, k=(1 + k))), lw=2, c='cornflowerblue')
    ax6.plot(ells[:-(1 + k)], np.abs(np.diag(ratio_EE_BB_2, k=(1 + k))), lw=2, c='orange')

    ax10.plot(ells[:-k], np.abs(np.diag(ratio_BB_BB, k=k)), lw=2, c='cornflowerblue')
    ax10.plot(ells[:-k], np.abs(np.diag(ratio_BB_BB_2, k=k)), lw=2, c='orange')

    k = 8
    ax3.plot(ells[:-k], np.abs(np.diag(ratio_EE_EE, k=k)), lw=2, c='cornflowerblue')
    ax3.plot(ells[:-k], np.abs(np.diag(ratio_EE_EE_2, k=k)), lw=2, c='orange')

    ax7.plot(ells[:-(1 + k)], np.abs(np.diag(ratio_EE_BB, k=(1 + k))), lw=2, c='cornflowerblue')
    ax7.plot(ells[:-(1 + k)], np.abs(np.diag(ratio_EE_BB_2, k=(1 + k))), lw=2, c='orange')

    ax11.plot(ells[:-k], np.abs(np.diag(ratio_BB_BB, k=k)), lw=2, c='cornflowerblue')
    ax11.plot(ells[:-k], np.abs(np.diag(ratio_BB_BB_2, k=k)), lw=2, c='orange')

    k = 32
    ax4.plot(ells[:-k], np.abs(np.diag(ratio_EE_EE, k=k)), lw=2, c='cornflowerblue', label=f'5 maps')
    ax4.plot(ells[:-k], np.abs(np.diag(ratio_EE_EE_2, k=k)), lw=2, c='orange', label=f'100 maps')

    ax8.plot(ells[:-(1 + k)], np.abs(np.diag(ratio_EE_BB, k=(1 + k))), lw=2, c='cornflowerblue')
    ax8.plot(ells[:-(1 + k)], np.abs(np.diag(ratio_EE_BB_2, k=(1 + k))), lw=2, c='orange')

    ax12.plot(ells[:-k], np.abs(np.diag(ratio_BB_BB, k=k)), lw=2, c='cornflowerblue')
    ax12.plot(ells[:-k], np.abs(np.diag(ratio_BB_BB_2, k=k)), lw=2, c='orange')

    ax9.set_xlabel(r'$\ell$', size=16)
    ax10.set_xlabel(r'$\ell$', size=16)
    ax11.set_xlabel(r'$\ell$', size=16)
    ax12.set_xlabel(r'$\ell$', size=16)

    ax1.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell\, \ell^{\prime}}^{EE\,EE} / \mathbf{F}_{\ell\, \ell^{\prime}}^{EE\,EE}$', size=16)
    ax5.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell\, \ell^{\prime}}^{EE\,BB} / \mathbf{F}_{\ell\, \ell^{\prime}}^{EE\,BB}$', size=16)
    ax9.set_ylabel(r'$\tilde{\mathbf{F}}_{\ell\, \ell^{\prime}}^{BB\,BB} / \mathbf{F}_{\ell\, \ell^{\prime}}^{BB\,BB}$', size=16)

    ax1.set_title(r'$\Delta \ell = 0$', size=16)
    ax2.set_title(r'$\Delta \ell = 2$', size=16)
    ax3.set_title(r'$\Delta \ell = 8$', size=16)
    ax4.set_title(r'$\Delta \ell = 32$', size=16)

    [[ax.tick_params(axis='x', labelsize=14) for ax in row] for row in axs]
    [[ax.tick_params(axis='y', labelsize=14) for ax in row] for row in axs]

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:.2f}$'))
    ax5.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:.2f}$'))
    ax9.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:.2f}$'))

    ax4.legend(ncol=1, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{folder_path}/Numerical_to_analytic_Fisher_N64_grid.pdf')

    import sys
    sys.exit()

    # * Now plotting the raw covariance matrices
    plt.rcParams['text.usetex'] = True
    offset = n_ell // 2

    # To do so, we need to cut out the EB parts of the matrix
    analytic_fisher = np.delete(analytic_fisher, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
    analytic_fisher = np.delete(analytic_fisher, np.s_[n_ell_full: 2 * n_ell_full], axis=1)

    # Now cut out the ell > 2 * n_side modes for BB
    analytic_fisher = np.delete(analytic_fisher, np.s_[n_ell_full + n_ell:], axis=0)
    analytic_fisher = np.delete(analytic_fisher, np.s_[n_ell_full + n_ell:], axis=1)

    # Now cut out the ell > 2 * n_side modes for EE
    analytic_fisher = np.delete(analytic_fisher, np.s_[n_ell: n_ell_full], axis=0)
    analytic_fisher = np.delete(analytic_fisher, np.s_[n_ell: n_ell_full], axis=1)

    analytic_fisher = np.insert(analytic_fisher, n_ell, values=np.nan, axis=0)
    analytic_fisher = np.insert(analytic_fisher, n_ell, values=np.nan, axis=1)

    # Plot the "analytic" result
    plt.figure(figsize=(9, 7))

    plt.imshow(np.log10(np.abs(analytic_fisher)), cmap='inferno')
    plt.colorbar(label=r'$\textrm{Log}_{10}(|\mathbf{F}_{\ell_1 \, \ell_2}|)$')

    plt.title(f'Plot of the analytic Fisher matrix for $N={n_side}$')

    # Dummy white text for consistency between the numeric and analytic plots
    plt.xlabel('Power injected into these modes', labelpad=40, color='white')
    plt.ylabel('Response measured in these modes', labelpad=30, color='white')

    plt.text(0 * n_ell + offset, 2 * n_ell + 15, r'$\ell_{1}^{EE}$', fontsize=15)
    plt.text(1 * n_ell + offset, 2 * n_ell + 15, r'$\ell_{1}^{BB}$', fontsize=15)

    plt.text(-20, 0 * n_ell + offset, r'$\ell_{2}^{EE}$', rotation='vertical', fontsize=15)
    plt.text(-20, 1 * n_ell + offset, r'$\ell_{2}^{BB}$', rotation='vertical', fontsize=15)

    ell_tick_pos = np.arange(2, 2 * n_side - 10, 16) - 2
    plt.xticks(np.concatenate([ell_tick_pos, ell_tick_pos + n_ell + 1]),
               np.concatenate([ell_tick_pos + 2, ell_tick_pos + 2]))
    plt.yticks(np.concatenate([ell_tick_pos, ell_tick_pos + n_ell + 1]),
               np.concatenate([ell_tick_pos + 2, ell_tick_pos + 2]))

    plt.tight_layout()
    plt.savefig(f'{folder_path}/Fisher_N64_Analytic.pdf')

    # Numerical Fisher - 5 maps
    numeric_fisher_5maps = np.delete(numeric_fisher_5maps, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
    numeric_fisher_5maps = np.delete(numeric_fisher_5maps, np.s_[n_ell_full: 2 * n_ell_full], axis=1)

    # Now cut out the ell > 2 * n_side modes for BB
    numeric_fisher_5maps = np.delete(numeric_fisher_5maps, np.s_[n_ell_full + n_ell:], axis=0)
    numeric_fisher_5maps = np.delete(numeric_fisher_5maps, np.s_[n_ell_full + n_ell:], axis=1)

    # Now cut out the ell > 2 * n_side modes for EE
    numeric_fisher_5maps = np.delete(numeric_fisher_5maps, np.s_[n_ell: n_ell_full], axis=0)
    numeric_fisher_5maps = np.delete(numeric_fisher_5maps, np.s_[n_ell: n_ell_full], axis=1)

    numeric_fisher_5maps = np.insert(numeric_fisher_5maps, n_ell, values=np.nan, axis=0)
    numeric_fisher_5maps = np.insert(numeric_fisher_5maps, n_ell, values=np.nan, axis=1)

    # Plot the numerical result for five maps
    plt.figure(figsize=(9, 7))

    plt.imshow(np.log10(np.abs(numeric_fisher_5maps)), cmap='inferno')
    plt.colorbar(label=r'$\textrm{Log}_{10}(|\tilde{\mathbf{F}}_{\ell_1 \, \ell_2}|)$')

    plt.title(f'Raw Fisher matrix estimated using 5 maps ')
    plt.xlabel('Power injected into these modes', labelpad=40)
    plt.ylabel('Response measured in these modes', labelpad=30)

    plt.text(0 * n_ell + offset, 2 * n_ell + 15, r'$\ell_{1}^{EE}$', fontsize=15)
    plt.text(1 * n_ell + offset, 2 * n_ell + 15, r'$\ell_{1}^{BB}$', fontsize=15)

    plt.text(-20, 0 * n_ell + offset, r'$\ell_{2}^{EE}$', rotation='vertical', fontsize=15)
    plt.text(-20, 1 * n_ell + offset, r'$\ell_{2}^{BB}$', rotation='vertical', fontsize=15)

    ell_tick_pos = np.arange(2, 2 * n_side - 10, 16) - 2
    plt.xticks(np.concatenate([ell_tick_pos, ell_tick_pos + n_ell + 1]),
               np.concatenate([ell_tick_pos + 2, ell_tick_pos + 2]))
    plt.yticks(np.concatenate([ell_tick_pos, ell_tick_pos + n_ell + 1]),
               np.concatenate([ell_tick_pos + 2, ell_tick_pos + 2]))

    plt.tight_layout()
    plt.savefig(f'{folder_path}/Fisher_N64_Numeric_5maps.pdf')

    # Numerical Fisher - 100 maps
    numeric_fisher_100maps = np.delete(numeric_fisher_100maps, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
    numeric_fisher_100maps = np.delete(numeric_fisher_100maps, np.s_[n_ell_full: 2 * n_ell_full], axis=1)

    # Now cut out the ell > 2 * n_side modes for BB
    numeric_fisher_100maps = np.delete(numeric_fisher_100maps, np.s_[n_ell_full + n_ell:], axis=0)
    numeric_fisher_100maps = np.delete(numeric_fisher_100maps, np.s_[n_ell_full + n_ell:], axis=1)

    # Now cut out the ell > 2 * n_side modes for EE
    numeric_fisher_100maps = np.delete(numeric_fisher_100maps, np.s_[n_ell: n_ell_full], axis=0)
    numeric_fisher_100maps = np.delete(numeric_fisher_100maps, np.s_[n_ell: n_ell_full], axis=1)

    numeric_fisher_100maps = np.insert(numeric_fisher_100maps, n_ell, values=np.nan, axis=0)
    numeric_fisher_100maps = np.insert(numeric_fisher_100maps, n_ell, values=np.nan, axis=1)

    # Plot the numerical result for one hundred maps
    plt.figure(figsize=(9, 7))

    plt.imshow(np.log10(np.abs(numeric_fisher_100maps)), cmap='inferno')
    plt.colorbar(label=r'$\textrm{Log}_{10}(|\tilde{\mathbf{F}}_{\ell_1 \, \ell_2}|)$')

    plt.title(f'Raw Fisher matrix estimated using 100 maps ')
    plt.xlabel('Power injected into these modes', labelpad=40)
    plt.ylabel('Response measured in these modes', labelpad=30)

    plt.text(0 * n_ell + offset, 2 * n_ell + 25, r'$\ell_{1}^{EE}$', fontsize=15)
    plt.text(1 * n_ell + offset, 2 * n_ell + 25, r'$\ell_{1}^{BB}$', fontsize=15)

    plt.text(-30, 0 * n_ell + offset, r'$\ell_{2}^{EE}$', rotation='vertical', fontsize=15)
    plt.text(-30, 1 * n_ell + offset, r'$\ell_{2}^{BB}$', rotation='vertical', fontsize=15)

    ell_tick_pos = np.arange(2, 2 * n_side - 10, 16) - 2
    plt.xticks(np.concatenate([ell_tick_pos, ell_tick_pos + n_ell + 1]),
               np.concatenate([ell_tick_pos + 2, ell_tick_pos + 2]))
    plt.yticks(np.concatenate([ell_tick_pos, ell_tick_pos + n_ell + 1]),
               np.concatenate([ell_tick_pos + 2, ell_tick_pos + 2]))

    plt.tight_layout()
    plt.savefig(f'{folder_path}/Fisher_N64_Numeric_100maps.pdf')

    plt.close('all')
