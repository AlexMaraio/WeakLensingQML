"""
File to compare the numerical Fisher matrix as produced from my conjugate-gradient method to that obtained using the
"analytic" approach using the ECLIPSE formalism
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.25, rc={'text.usetex': True})


if __name__ == '__main__':
    # The N_side parameter for the Fisher matrix under consideration
    n_side = 64

    # Create an ell range that goes from ell=2 to ell=3*nside - 1
    full_ells = np.arange(2, 3 * n_side)
    n_ell_full = len(full_ells)

    # We only want to plot ells from ell=2 to ell=2*nside
    ells = np.arange(2, 2 * n_side + 1)
    n_ell = len(ells)

    # Load in the "analytic" Fisher matrix, as computed using ECLIPSE formalism
    analytic_fisher = np.load(f'../data/Fishers/F_N{n_side}.npy')

    # Load in our low-resolution numerical conjugate-gradient Fisher
    numeric_fisher = np.loadtxt(f'../data/Fishers/ConjGrad_Fisher_N{n_side}_nummaps5_noise30_EB.dat')

    # Load in the high-resolution Fisher
    numeric_fisher2 = np.loadtxt(f'../data/Fishers/ConjGrad_Fisher_N{n_side}_nummaps75_noise30_EB.dat')

    # Take the ratio of numerical to analytic Fisher for our two matrices
    ratio = numeric_fisher / analytic_fisher
    ratio2 = numeric_fisher2 / analytic_fisher

    # Extract the EE-EE and BB-BB parts
    ratio_EE_EE = ratio[0 * n_ell_full: 0 * n_ell_full + n_ell, 0 * n_ell_full: 0 * n_ell_full + n_ell]
    ratio_BB_BB = ratio[2 * n_ell_full: 2 * n_ell_full + n_ell, 2 * n_ell_full: 2 * n_ell_full + n_ell]

    ratio_EE_EE_2 = ratio2[0 * n_ell_full: 0 * n_ell_full + n_ell, 0 * n_ell_full: 0 * n_ell_full + n_ell]
    ratio_BB_BB_2 = ratio2[2 * n_ell_full: 2 * n_ell_full + n_ell, 2 * n_ell_full: 2 * n_ell_full + n_ell]

    # Plot the diagonal of this ratio
    plt.figure(figsize=(8, 5))

    plt.plot(ells, np.diag(ratio_EE_EE), lw=2, c='purple', label='$EE$ - 5 maps')
    plt.plot(ells, np.diag(ratio_BB_BB), lw=2, c='mediumseagreen', label='$BB$ - 5 maps')

    plt.plot(ells, np.diag(ratio_EE_EE_2), lw=2, c='cornflowerblue', label='$EE$ - 75 maps')
    plt.plot(ells, np.diag(ratio_BB_BB_2), lw=2, c='lime', label='$BB$ - 75 maps')

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\tilde{\mathbf{F}}_{\ell \ell} / \mathbf{F}_{\ell \ell}$')
    plt.title('Ratio of conjugate-gradient to analytic Fisher')

    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig('../data/Plots/FisherRatio.pdf')
