"""
Script that plots summary statistics (mean and standard deviation) for a set of Cl values that have been recovered
from an ensemble of maps
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import constants as consts
import healpy as hp


sns.set(font_scale=1.25, rc={'text.usetex': True})

#* Input data
# The map resolution
n_side = 256

# The number of maps that were in the ensemble
num_maps = 1250

# The location of map data
folder_path = "/disk01/maraio/NonGaussianShear/N1024_LogNormal/Maps_N256"

# The location of where to save the plots into
plots_folder = "../data/Plots/N256_LogNormal"
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

# Compute resolution-associated quantities
l_max = 2 * n_side
l_max_full = 3 * n_side - 1
ells = np.arange(2, l_max + 1)
ells_full = np.arange(2, l_max_full + 1)
n_ell = len(ells)
n_ell_full = len(ells_full)

# The polarisation HealPix window function values for the specified resolution
pix_win = hp.pixwin(n_side, pol=True, lmax=l_max)[1][2:]

# Read in the theory spectra associated with the maps
ells, cl_th_EE = np.loadtxt(f'{folder_path}/../TheoryClsShear-f1z1f1z1.dat').T[:, 0: n_ell]

# Read in our QML Fisher matrix
QML_fisher_filepath = f"../data/numerical_fishers/ConjGrad_Fisher_N{n_side}_nummaps10_noise3_EB_nostars.dat"

# Read in the Fisher matrix and then remove EB entries, and make symmetric
QML_fisher_matrix = np.loadtxt(QML_fisher_filepath)

QML_fisher_matrix = np.delete(QML_fisher_matrix, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
QML_fisher_matrix = np.delete(QML_fisher_matrix, np.s_[n_ell_full: 2 * n_ell_full], axis=1)

QML_fisher_matrix = (QML_fisher_matrix.copy() + QML_fisher_matrix.copy().T) / 2

# Invert the Fisher matrix
QML_fisher_matrix_inv = np.linalg.inv(QML_fisher_matrix)

# Noise values
intrinsic_gal_ellip = 0.21
avg_gal_den = 3
theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (consts.arcminute ** 2))

# Create arrays where we will store our data into
cl_samples_EE_QML = np.zeros([num_maps, len(ells_full)])
cl_samples_BB_QML = np.zeros([num_maps, len(ells_full)])

cl_samples_EE_PCl = np.zeros([num_maps, len(ells_full)])
cl_samples_BB_PCl = np.zeros([num_maps, len(ells_full)])

# Go through each map in our ensemble
for map_num in range(num_maps):
    # Load the QML power spectrum and store into QML data vectors
    cl_EE_QML = np.loadtxt(f'{folder_path}/Map{map_num}_Cl_EE_QML.dat')
    cl_BB_QML = np.loadtxt(f'{folder_path}/Map{map_num}_Cl_BB_QML.dat')

    cl_samples_EE_QML[map_num - 1] = cl_EE_QML
    cl_samples_BB_QML[map_num - 1] = cl_BB_QML

    # Load the PCl power spectrum and store into PCl data vectors
    cl_EE_PCl = np.loadtxt(f'{folder_path}/Map{map_num}_Cl_EE_PCl.dat')
    cl_BB_PCl = np.loadtxt(f'{folder_path}/Map{map_num}_Cl_BB_PCl.dat')

    cl_samples_EE_PCl[map_num - 1] = cl_EE_PCl
    cl_samples_BB_PCl[map_num - 1] = cl_BB_PCl

# Compute the average of our Cl values for QML
cl_avg_EE_QML = np.mean(cl_samples_EE_QML, axis=0)[0: n_ell]
cl_avg_BB_QML = np.mean(cl_samples_BB_QML, axis=0)[0: n_ell]

# Compute PCl average
cl_avg_EE_PCl = np.mean(cl_samples_EE_PCl, axis=0)[0: n_ell]
cl_avg_BB_PCl = np.mean(cl_samples_BB_PCl, axis=0)[0: n_ell]

# Compute the standard deviation of the QML Cl values
cl_std_EE_QML = np.std(cl_samples_EE_QML, axis=0)[0: n_ell]
cl_std_BB_QML = np.std(cl_samples_BB_QML, axis=0)[0: n_ell]

# Compute the standard deviation of the PCl Cl values
cl_std_EE_PCl = np.std(cl_samples_EE_PCl, axis=0)[0: n_ell]
cl_std_BB_PCl = np.std(cl_samples_BB_PCl, axis=0)[0: n_ell]

# * Now plots!

# Plot the average EE values
plt.figure(figsize=(11, 6))

plt.plot(ells, cl_avg_EE_QML, lw=2, c='cornflowerblue', label='QML')
plt.errorbar(ells, cl_avg_EE_QML + theory_cl_noise, cl_std_EE_QML, linestyle='', c='cornflowerblue')

plt.plot(ells, cl_avg_EE_PCl, lw=2, c='mediumseagreen', label='PCl')
plt.errorbar(ells + 0.125, cl_avg_EE_PCl + theory_cl_noise, cl_std_EE_PCl, linestyle='', c='mediumseagreen')

plt.plot(ells, cl_th_EE + theory_cl_noise, ls='--', c='purple', label='Theory')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{EE}$')

plt.legend()
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_EE_avg.pdf')

# Plot the average BB values
plt.figure(figsize=(11, 6))

plt.plot(ells, cl_avg_BB_QML, lw=2, c='cornflowerblue', label='QML')
# plt.errorbar(ells, cl_avg_BB_QML, cl_std_BB_QML, linestyle='', c='cornflowerblue')

plt.plot(ells, cl_avg_BB_PCl, lw=2, c='mediumseagreen', label='PCl')
# plt.errorbar(ells+0.125, cl_avg_BB_PCl, cl_std_BB_PCl, linestyle='', c='mediumseagreen')

plt.plot(ells, [theory_cl_noise] * n_ell, lw=2, ls='--', c='purple', label='Theory')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')

plt.legend()
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_BB_avg.pdf')

# EE average ratio
plt.figure(figsize=(11, 6))

plt.plot(ells, cl_avg_EE_QML / (cl_th_EE + theory_cl_noise), lw=2, c='cornflowerblue', label='QML')

plt.plot(ells, cl_avg_EE_PCl / (cl_th_EE + theory_cl_noise), lw=2, c='mediumseagreen', label='PCl')

plt.plot(ells, pix_win, lw=2, c='crimson', label='Pix win')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{EE} / \hat{C}_{\ell}^{EE}$')
plt.title('Ratio of recovered power spectra to theory values - Downgraded from $N=1024$ to $N=256$')

plt.legend()
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_EE_avg_ratio.pdf')

# EE std-dev's
plt.figure(figsize=(11, 6))

plt.loglog(ells, cl_std_EE_QML, lw=2, c='cornflowerblue', label='QML')
plt.loglog(ells, cl_std_EE_PCl, lw=2, c='mediumseagreen', label='PCl')

plt.loglog(ells, np.sqrt(np.diag(QML_fisher_matrix_inv[0: n_ell, 0: n_ell])), lw=2, ls='--', c='orange',
           label='inv-Fisher')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma_{C_{\ell}}^{EE}$')

plt.legend()
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_EE_std.pdf')

plt.figure(figsize=(11, 6))

plt.loglog(ells, cl_std_BB_QML, lw=2, c='cornflowerblue', label='QML')
plt.loglog(ells, cl_std_BB_PCl, lw=2, c='mediumseagreen', label='PCl')

plt.loglog(ells,
           np.sqrt(np.diag(QML_fisher_matrix_inv[n_ell_full: n_ell_full + n_ell, n_ell_full: n_ell_full + n_ell])),
           lw=2, ls='--', c='orange', label='inv-Fisher')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma_{C_{\ell}}^{BB}$')

plt.legend()
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_BB_std.pdf')

# Now plot the EE and BB std-dev ratio
plt.figure(figsize=(11, 6))

plt.loglog(ells, cl_std_EE_PCl / cl_std_EE_QML, lw=2, c='cornflowerblue', label='$EE$')
plt.loglog(ells, cl_std_BB_PCl / cl_std_BB_QML, lw=2, c='mediumseagreen', label='$BB$')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma_{C_{\ell}}^{\textrm{PCl}} / \sigma_{C_{\ell}}^{\textrm{QML}}$')
plt.title('Ratio of power spectrum standard deviations for Log-Normal maps')

plt.legend()
plt.tight_layout()

plt.savefig(f'{plots_folder}/Cl_std_ratio.pdf')

plt.close('all')
