"""
Script that plots summary statistics (mean and standard deviation) for a set of Cl values that have been recovered
from an ensemble of maps
"""

import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import constants as consts
import healpy as hp

from Binning import Bins, BinType, uniform_weights, mode_count_weights, inverse_variance_weights


sns.set(font_scale=1.25, rc={'text.usetex': True})

#* Input data
# The map resolution
n_side = 256

# The number of maps that were in the ensemble
num_maps = 2_500

# The location of map data
folder_path_gaussian = "/cephfs/maraio/NonGaussianMaps/N1024_Gaussian_shift_0_01214_whnoise/Maps_N256"
folder_path_lognormal = "/cephfs/maraio/NonGaussianMaps/N1024_LogNormal_shift_0_01214_whnoise/Maps_N256"

# The location of where to save the plots into
plots_folder = "../../data/Plots/LogNormalMaps/N256_comparison_binning"
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

# Compute resolution-associated quantities
l_max = 3 * n_side - 1
ells = np.arange(2, l_max + 1)
n_ell = len(ells)

# The polarisation HealPix window function values for the specified resolution
pix_win = hp.pixwin(n_side, pol=True, lmax=l_max)[1][2:]

# Read in the theory spectra associated with the maps
ells, cl_th_EE = np.loadtxt(f'{folder_path_gaussian}/../TheoryClsShear-f1z1f1z1.dat').T[:, 0: n_ell]

# Compute the expected variance of the shear maps
cl_var = np.sum((2 * ells + 1) * cl_th_EE / (4 * np.pi) / 2 * (pix_win ** 2))
print(f'Expected shear variance is {cl_var:.4e}')

# Read in our QML Fisher matrix
QML_fisher_filepath = f"../../data/numerical_fishers/ConjGrad_Fisher_N{n_side}_nummaps25_noise3_EB_whstars.dat"

# Read in the Fisher matrix and then remove EB entries, and make symmetric
QML_fisher_matrix = np.loadtxt(QML_fisher_filepath)

QML_fisher_matrix = np.delete(QML_fisher_matrix, np.s_[n_ell: 2 * n_ell], axis=0)
QML_fisher_matrix = np.delete(QML_fisher_matrix, np.s_[n_ell: 2 * n_ell], axis=1)

QML_fisher_matrix = (QML_fisher_matrix.copy() + QML_fisher_matrix.copy().T) / 2

# Invert the Fisher matrix
QML_fisher_matrix_inv = np.linalg.inv(QML_fisher_matrix)

# Noise values
intrinsic_gal_ellip = 0.21
avg_gal_den = 3
theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (consts.arcminute ** 2))

# Create bins object
# Ells per bin of one corresponds to an unbinned spectra
ells_per_bin = 1
bins = Bins(n_side, BinType.linear, ells_per_bin, mode_count_weights)

# Bin the theory vector
cl_th_EE_binned = bins.bin_vector(cl_th_EE)

# Create arrays where we will store our data into
cl_samples_EE_QML_gaussian = np.zeros([num_maps, bins.num_bins])
cl_samples_BB_QML_gaussian = np.zeros([num_maps, bins.num_bins])

cl_samples_EE_PCl_gaussian = np.zeros([num_maps, bins.num_bins])
cl_samples_BB_PCl_gaussian = np.zeros([num_maps, bins.num_bins])

cl_samples_EE_QML_lognormal = np.zeros([num_maps, bins.num_bins])
cl_samples_BB_QML_lognormal = np.zeros([num_maps, bins.num_bins])

cl_samples_EE_PCl_lognormal = np.zeros([num_maps, bins.num_bins])
cl_samples_BB_PCl_lognormal = np.zeros([num_maps, bins.num_bins])

# Go through each map in our ensemble
for map_num in range(num_maps):
    # Load the Gaussian QML power spectrum and store into QML data vectors
    cl_EE_QML_gaussian = np.loadtxt(f'{folder_path_gaussian}/Map{map_num}_Cl_EE_QML.dat')
    cl_BB_QML_gaussian = np.loadtxt(f'{folder_path_gaussian}/Map{map_num}_Cl_BB_QML.dat')

    # Bin the power spectrum
    cl_EE_QML_gaussian = bins.bin_vector(cl_EE_QML_gaussian)
    cl_BB_QML_gaussian = bins.bin_vector(cl_BB_QML_gaussian)

    cl_samples_EE_QML_gaussian[map_num] = cl_EE_QML_gaussian
    cl_samples_BB_QML_gaussian[map_num] = cl_BB_QML_gaussian

    # Load the Gaussian PCl power spectrum and store into PCl data vectors
    cl_EE_PCl_gaussian = np.loadtxt(f'{folder_path_gaussian}/Map{map_num}_Cl_EE_PCl.dat')
    cl_BB_PCl_gaussian = np.loadtxt(f'{folder_path_gaussian}/Map{map_num}_Cl_BB_PCl.dat')

    cl_EE_PCl_gaussian = bins.bin_vector(cl_EE_PCl_gaussian)
    cl_BB_PCl_gaussian = bins.bin_vector(cl_BB_PCl_gaussian)

    cl_samples_EE_PCl_gaussian[map_num] = cl_EE_PCl_gaussian
    cl_samples_BB_PCl_gaussian[map_num] = cl_BB_PCl_gaussian

    # Load the log-normal QML power spectrum and store into QML data vectors
    cl_EE_QML_lognormal = np.loadtxt(f'{folder_path_lognormal}/Map{map_num}_Cl_EE_QML.dat')
    cl_BB_QML_lognormal = np.loadtxt(f'{folder_path_lognormal}/Map{map_num}_Cl_BB_QML.dat')

    cl_EE_QML_lognormal = bins.bin_vector(cl_EE_QML_lognormal)
    cl_BB_QML_lognormal = bins.bin_vector(cl_BB_QML_lognormal)

    cl_samples_EE_QML_lognormal[map_num] = cl_EE_QML_lognormal
    cl_samples_BB_QML_lognormal[map_num] = cl_BB_QML_lognormal

    # Load the log-normal PCl power spectrum and store into PCl data vectors
    cl_EE_PCl_lognormal = np.loadtxt(f'{folder_path_lognormal}/Map{map_num}_Cl_EE_PCl.dat')
    cl_BB_PCl_lognormal = np.loadtxt(f'{folder_path_lognormal}/Map{map_num}_Cl_BB_PCl.dat')

    cl_EE_PCl_lognormal = bins.bin_vector(cl_EE_PCl_lognormal)
    cl_BB_PCl_lognormal = bins.bin_vector(cl_BB_PCl_lognormal)

    cl_samples_EE_PCl_lognormal[map_num] = cl_EE_PCl_lognormal
    cl_samples_BB_PCl_lognormal[map_num] = cl_BB_PCl_lognormal

# Compute the average of our Cl values for QML - Gaussian
cl_avg_EE_QML_gaussian = np.mean(cl_samples_EE_QML_gaussian, axis=0)
cl_avg_BB_QML_gaussian = np.mean(cl_samples_BB_QML_gaussian, axis=0)

# Compute PCl average
cl_avg_EE_PCl_gaussian = np.mean(cl_samples_EE_PCl_gaussian, axis=0)
cl_avg_BB_PCl_gaussian = np.mean(cl_samples_BB_PCl_gaussian, axis=0)

# Compute the standard deviation of the QML Cl values
cl_std_EE_QML_gaussian = np.std(cl_samples_EE_QML_gaussian, axis=0)
cl_std_BB_QML_gaussian = np.std(cl_samples_BB_QML_gaussian, axis=0)

# Compute the standard deviation of the PCl Cl values
cl_std_EE_PCl_gaussian = np.std(cl_samples_EE_PCl_gaussian, axis=0)
cl_std_BB_PCl_gaussian = np.std(cl_samples_BB_PCl_gaussian, axis=0)

# Compute averages & st-dev's for log-normal case
cl_avg_EE_QML_lognormal = np.mean(cl_samples_EE_QML_lognormal, axis=0)
cl_avg_BB_QML_lognormal = np.mean(cl_samples_BB_QML_lognormal, axis=0)

# Compute PCl average
cl_avg_EE_PCl_lognormal = np.mean(cl_samples_EE_PCl_lognormal, axis=0)
cl_avg_BB_PCl_lognormal = np.mean(cl_samples_BB_PCl_lognormal, axis=0)

# Compute the standard deviation of the QML Cl values
cl_std_EE_QML_lognormal = np.std(cl_samples_EE_QML_lognormal, axis=0)
cl_std_BB_QML_lognormal = np.std(cl_samples_BB_QML_lognormal, axis=0)

# Compute the standard deviation of the PCl Cl values
cl_std_EE_PCl_lognormal = np.std(cl_samples_EE_PCl_lognormal, axis=0)
cl_std_BB_PCl_lognormal = np.std(cl_samples_BB_PCl_lognormal, axis=0)

# * Now plots!

# Plot the average EE values
plt.figure(figsize=(11, 6))

plt.loglog(bins.bin_centres, cl_avg_EE_QML_gaussian, lw=2, c='cornflowerblue', label='QML Gaussian')
plt.loglog(bins.bin_centres, cl_avg_EE_QML_lognormal, lw=2, c='purple', label='QML log-normal')

plt.loglog(bins.bin_centres, cl_avg_EE_PCl_gaussian, lw=2, c='lightseagreen', label='PCl Gaussian')
plt.loglog(bins.bin_centres, cl_avg_EE_PCl_lognormal, lw=2, c='darkseagreen', label='PCl log-normal')

plt.loglog(ells, cl_th_EE + theory_cl_noise, ls='--', c='cyan', label='Theory')
plt.loglog(bins.bin_centres, cl_th_EE_binned + theory_cl_noise, ls='--', c='yellow', label='Theory')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{EE}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=3)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_EE_avg_ells{ells_per_bin}.pdf')

# Plot the average BB values
plt.figure(figsize=(11, 6))

plt.loglog(bins.bin_centres, cl_avg_BB_QML_gaussian, lw=2, c='cornflowerblue', label='QML Gaussian')
plt.loglog(bins.bin_centres, cl_avg_BB_QML_lognormal, lw=2, c='purple', label='QML log-normal')

plt.loglog(bins.bin_centres, cl_avg_BB_PCl_gaussian, lw=2, c='lightseagreen', label='PCl Gaussian')
plt.loglog(bins.bin_centres, cl_avg_BB_PCl_lognormal, lw=2, c='darkseagreen', label='PCl log-normal')

plt.loglog(ells, [theory_cl_noise] * n_ell, lw=2, ls='--', c='cyan', label='Theory')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=3)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_BB_avg_ells{ells_per_bin}.pdf')

# EE average ratio
plt.figure(figsize=(11, 6))

plt.semilogx(bins.bin_centres, cl_avg_EE_QML_gaussian / (cl_th_EE_binned + theory_cl_noise), lw=2, c='cornflowerblue', label='QML Gaussian')
plt.semilogx(bins.bin_centres, cl_avg_EE_QML_lognormal / (cl_th_EE_binned + theory_cl_noise), lw=2, c='purple', label='QML log-normal')

plt.semilogx(bins.bin_centres, cl_avg_EE_PCl_gaussian / (cl_th_EE_binned + theory_cl_noise), lw=2, c='lightseagreen', label='PCl Gaussian')
plt.semilogx(bins.bin_centres, cl_avg_EE_PCl_lognormal / (cl_th_EE_binned + theory_cl_noise), lw=2, c='darkseagreen', label='PCl log-normal')

plt.semilogx(ells, pix_win, lw=2, c='crimson', label='Pix win')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{EE} / \hat{C}_{\ell}^{EE}$')
plt.title(f'Ratio of recovered $EE$ power spectra to theory values - Downgraded from $N=1024$ to $N=256$, $\ell_{{b}} = {ells_per_bin}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=3)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_EE_avg_ratio_ells{ells_per_bin}.pdf')

# BB average ratio
plt.figure(figsize=(11, 6))

plt.semilogx(bins.bin_centres, cl_avg_BB_QML_gaussian / theory_cl_noise, lw=2, c='cornflowerblue', label='QML Gaussian')
plt.semilogx(bins.bin_centres, cl_avg_BB_QML_lognormal / theory_cl_noise, lw=2, c='purple', label='QML log-normal')

plt.semilogx(bins.bin_centres, cl_avg_BB_PCl_gaussian / theory_cl_noise, lw=2, c='lightseagreen', label='PCl Gaussian')
plt.semilogx(bins.bin_centres, cl_avg_BB_PCl_lognormal / theory_cl_noise, lw=2, c='darkseagreen', label='PCl log-normal')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB} / \hat{C}_{\ell}^{BB}$')
plt.title(f'Ratio of recovered $BB$ power spectra to theory values - Downgraded from $N=1024$ to $N=256$, $\ell_{{b}} = {ells_per_bin}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=2)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_BB_avg_ratio_ells{ells_per_bin}.pdf')

# EE std-dev's
plt.figure(figsize=(11, 6))

plt.loglog(bins.bin_centres, cl_std_EE_QML_gaussian, lw=2, c='cornflowerblue', label='QML Gaussian')
plt.loglog(bins.bin_centres, cl_std_EE_QML_lognormal, lw=2, c='purple', label='QML log-normal')

plt.loglog(bins.bin_centres, cl_std_EE_PCl_gaussian, lw=2, c='lightseagreen', label='PCl Gaussian')
plt.loglog(bins.bin_centres, cl_std_EE_PCl_lognormal, lw=2, c='darkseagreen', label='PCl log-normal')

plt.loglog(ells, np.sqrt(np.diag(QML_fisher_matrix_inv[0: n_ell, 0: n_ell])), lw=2, ls='--', c='orange',
           label='inv-Fisher')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma_{C_{\ell}}^{EE}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=3)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_EE_std_ells{ells_per_bin}.pdf')

plt.figure(figsize=(11, 6))

plt.loglog(bins.bin_centres, cl_std_BB_QML_gaussian, lw=2, c='cornflowerblue', label='QML Gaussian')
plt.loglog(bins.bin_centres, cl_std_BB_QML_lognormal, lw=2, c='purple', label='QML log-normal')

plt.loglog(bins.bin_centres, cl_std_BB_PCl_gaussian, lw=2, c='lightseagreen', label='PCl Gaussian')
plt.loglog(bins.bin_centres, cl_std_BB_PCl_lognormal, lw=2, c='darkseagreen', label='PCl log-normal')

plt.loglog(ells,
           np.sqrt(np.diag(QML_fisher_matrix_inv[n_ell: 2 * n_ell, n_ell: 2 * n_ell])),
           lw=2, ls='--', c='orange', label='inv-Fisher')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma_{C_{\ell}}^{BB}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=3)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_BB_std_ells{ells_per_bin}.pdf')

# Now plot the EE and BB std-dev ratio
sns.reset_orig()
mpl.rc_file_defaults()
plt.rcParams['text.usetex'] = True

plt.figure(figsize=(6, 3.5))

plt.semilogx(bins.bin_centres, cl_std_EE_PCl_gaussian / cl_std_EE_QML_gaussian, lw=1.5, c='cornflowerblue', label='$EE$ Gaussian')
plt.semilogx(bins.bin_centres, cl_std_EE_PCl_lognormal / cl_std_EE_QML_lognormal, lw=1.5, ls='--', c='purple', label='$EE$ Lognormal')

plt.semilogx(bins.bin_centres, cl_std_BB_PCl_gaussian / cl_std_BB_QML_gaussian, lw=1.5, c='lightseagreen', label='$BB$ Gaussian')
plt.semilogx(bins.bin_centres, cl_std_BB_PCl_lognormal / cl_std_BB_QML_lognormal, lw=1.5, ls='--', c='darkseagreen', label='$BB$ Lognormal')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma_{C_{\ell}}^{\textrm{PCl}} / \sigma_{C_{\ell}}^{\textrm{QML}}$')
# plt.title(f'Ratio of power spectrum standard deviations - Downgraded from $N=1024$ to $N=256$, $\ell_{{b}} = {ells_per_bin}$')

plt.xlim(right=2 * n_side)
plt.legend(ncol=2)
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_std_ratio_ells{ells_per_bin}.pdf')

plt.close('all')
