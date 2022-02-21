import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

import seaborn as sns
sns.set(font_scale=1.0, rc={'text.usetex': True})

from lib.PostProcessing.Enums import ClType, CovType
from lib.PostProcessing.PCl_covariances import PCl_covariances
from lib.PostProcessing.QML_covariances import QML_covariances
from lib.PostProcessing.ParameterFisher import ParamFisher, plot_param_Fisher

import matplotlib as mpl
mpl.use('Agg')

# The output folder under ./Plots
folder = 'Noise30_Unbinned_2deg'

# See if folder doesn't exist already, then make it if not
if not os.path.isdir(f'Plots_New/{folder}'):
    os.makedirs(f'Plots_New/{folder}')

# HEALPix map resolution
n_side = 256

# Number of ells per bin
ells_per_bin = 1

# The apodisation scale (in deg)
apo_scale = 2.0
apo_scale_str = '2'

# Number of numerical samples
num_samples = 25_000

#* Read in the masks
mask_nostars = hp.read_map('/home/maraio/Codes/QMLForWeakLensingHome/Data/Masks/New/SkyMask_N256_nostars.fits', dtype=float)
# mask_whstars = hp.read_map('/home/maraio/Codes/QMLForWeakLensingHome/Data/Masks/New/SkyMask_N256_whstars.fits', dtype=float)

# Apodise the masks
mask_nostars_apo = nmt.mask_apodization(mask_nostars, aposize=apo_scale, apotype='C2')
# mask_whstars_apo = nmt.mask_apodization(mask_whstars, aposize=apo_scale, apotype='C2')

# Or read in the star mask itself and combine that with out apodized galactic/ecliptic mask
# star_mask = hp.read_map('/home/maraio/Codes/flask-UCL/bin/StarMask_N256.fits', dtype=float)
# mask_whstars_apo = mask_nostars_apo * star_mask

# Compute the four sets of mask's f_sky
mask_nostars_fsky = mask_nostars.sum() / mask_nostars.size
mask_whstars_fsky = mask_whstars.sum() / mask_whstars.size
mask_nostars_apo_fsky = mask_nostars_apo.sum() / mask_nostars_apo.size
mask_whstars_apo_fsky = mask_whstars_apo.sum() / mask_whstars_apo.size

# Print f_sky of mask with and without apodisation
print(f'f_sky for mask without stars and without apodisation is {100 * mask_nostars_fsky:.2f} %')
print(f'f_sky for mask without stars and with apodisation is {100 * mask_whstars_fsky:.2f} %')
print(f'f_sky for mask with stars and without apodisation is {100 * mask_nostars_apo_fsky:.2f} %')
print(f'f_sky for mask with stars and with apodisation is {100 * mask_whstars_apo_fsky:.2f} %')

from matplotlib import cm
inferno = cm.get_cmap('inferno', 2)

#* Plot our four sets of masks, including their f_sky
hp.mollview(mask_nostars,
            title=f'Main mask without stars - no apodisation - $f_\\textrm{{sky}} = {100 * mask_nostars_fsky:.2f}\\,\\%$',
            cmap='viridis')
plt.savefig(f'{plots_folder}/Mask_nostars.pdf')

hp.mollview(mask_whstars,
            title=f'Main mask with stars - no apodisation - $f_\\textrm{{sky}} = {100 * mask_whstars_fsky:.2f}\\,\\%$',
            cmap='viridis')
plt.savefig(f'{plots_folder}/Mask_whstars.pdf')

hp.mollview(mask_nostars_apo,
            title=f'Main mask without stars - with apodisation - $f_\\textrm{{sky}} = {100 * mask_whstars_apo_fsky:.2f}\\,\\%$',
            cmap='viridis')
plt.savefig(f'{plots_folder}/Mask_nostars_apo.pdf')

hp.mollview(mask_whstars_apo,
            title=f'Main mask with stars - with apodisation - $f_\\textrm{{sky}} = {100 * mask_whstars_apo_fsky:.2f}\\,\\%$',
            cmap=inferno)
plt.savefig(f'{plots_folder}/Mask_whstars_apo.pdf')

#* Compute and plot the power spectrum of the masks
ells = np.arange(2, 3 * n_side)
mask_nostars_cls = hp.anafast(mask_nostars)[2:]
mask_whstars_cls = hp.anafast(mask_whstars)[2:]
mask_nostars_apo_cls = hp.anafast(mask_nostars_apo)[2:]
mask_whstars_apo_cls = hp.anafast(mask_whstars_apo)[2:]

plt.figure(figsize=(11, 7))

plt.loglog(ells[::2], mask_nostars_cls[::2], lw=2, c='cornflowerblue', label='No apo, no stars')
plt.loglog(ells[::2], mask_whstars_cls[::2], lw=2, c='mediumseagreen', label='No apo, with stars')
# plt.loglog(ells[::2], mask_nostars_apo_cls[::2], lw=2, c='orange', label='With apo, no stars')
# plt.loglog(ells[::2], mask_whstars_apo_cls[::2], lw=2, c='hotpink', label='With apo, with stars')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}$')
plt.title(r'Power spectrum of masks (even $\ell$ modes only)')

plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f'{plots_folder}/Masks_powerspectrum.pdf')

#* Noise values
from scipy import constants as consts
intrinsic_gal_ellip = 0.21
avg_gal_den = 30
area_per_pix = 148510661 / (12 * n_side * n_side)
num_gal_per_pix = avg_gal_den * area_per_pix

theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (consts.arcminute ** 2))
noise_std = intrinsic_gal_ellip / np.sqrt(num_gal_per_pix)

# theory_cl_noise = 0
# noise_std = 0

#* Compute power spectrum
from scipy import stats
import pyccl as ccl
ells = np.arange(0, 3 * n_side)
redshift_range = np.linspace(0.0, 3.0, 500)

n_ell = len(ells[2:])

# Create new galaxy distribution at our current redshift
dNdz = stats.norm.pdf(redshift_range, loc=1.0, scale=0.15)

cosmo = ccl.Cosmology(**{'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96, 'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0})

# Lensing bins at current redshift
field_E = ccl.WeakLensingTracer(cosmo, dndz=(redshift_range, dNdz))

# Compute Cl values
cl_ee = ccl.angular_cl(cosmo, field_E, field_E, ells)

cl_ee[:2] = 0.0

cl_eb = np.zeros_like(cl_ee)
cl_bb = np.zeros_like(cl_ee)

#* Set up PCl covariance objects
PCl_nostars = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False, mask=mask_nostars, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise)
# PCl_whstars = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False, mask=mask_whstars, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise)

PCl_nostars_apo = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False, mask=mask_nostars_apo, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise)
# PCl_whstars_apo = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False, mask=mask_whstars_apo, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise)

# PCl_nostars_apo_pure = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=True, mask=mask_nostars_apo, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise)
# PCl_whstars_apo_pure = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=True, mask=mask_whstars_apo, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise)

# Compute numerical covariances
PCl_nostars.compute_numerical_covariance()
# PCl_whstars.compute_numerical_covariance()

PCl_nostars_apo.compute_numerical_covariance()
# PCl_whstars_apo.compute_numerical_covariance()

# PCl_nostars_apo_pure.compute_numerical_covariance()
# PCl_whstars_apo_pure.compute_numerical_covariance()

# Compute analytic matrix
PCl_nostars.compute_analytic_covariance()

#* Now set up our QML objects
QML_nostars = QML_covariances(n_side, '/home/maraio/Codes/QMLForWeakLensingHome/Data/ConjGrad_NumFishers/Fisher_N256_nummaps5_noise30_nostars_EB.dat', ells_per_bin=ells_per_bin)
# QML_whstars = QML_covariances(n_side, '/home/maraio/Codes/QMLForWeakLensingHome/Data/ConjGrad_NumFishers/Fisher_N256_nummaps5_noise3_whstars_EB.dat', ells_per_bin=ells_per_bin)

# Optionally bin our QML estimators
# QML_nostars.bin_covariance_matrix()
# QML_whstars.bin_covariance_matrix()

#? Parameter Fisher plots
fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96, 'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

params = {'sigma8': 0.75, 'Omega_c': 0.27, 'h': 0.7}
params_latex = [r'\sigma_8', r'\Omega_c', r'h']
d_params = {'sigma8': 0.75 / 100, 'Omega_c': 0.27 / 100, 'h': 0.7 / 100}

params = {'w0': -1.0, 'wa': 0.0}
params_latex = [r'w_0', r'w_a']
d_params = {'w0': 0.005, 'wa': 0.005}


param_F_PCl_numeric = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.numeric, PCl_nostars, 'PCl numeric', params, params_latex, d_params, num_samples)
param_F_PCl_analytic = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.analytic, PCl_nostars, 'PCl analytic', params, params_latex, d_params, num_samples)
param_F_PCl_numeric_apo = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.numeric, PCl_nostars_apo, 'PCl numeric w/ apo', params, params_latex, d_params, num_samples)

param_F_QML_numeric = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.QML, CovType.numeric, QML_nostars, 'QML', params, params_latex, d_params, num_samples)

param_F_PCl_numeric.compute_param_Fisher()
param_F_PCl_analytic.compute_param_Fisher()
param_F_PCl_numeric_apo.compute_param_Fisher()
param_F_QML_numeric.compute_param_Fisher()

plot_param_Fisher([param_F_PCl_numeric, param_F_PCl_analytic, param_F_PCl_numeric_apo, param_F_QML_numeric], num_samples, folder)

plot_param_Fisher([param_F_PCl_numeric, param_F_PCl_analytic, param_F_QML_numeric], num_samples, 'ParamFisher1')

plot_param_Fisher([param_F_PCl_analytic, param_F_QML_numeric], num_samples+2, 'ParamFisher1')


# Reset Seaborn
sns.reset_orig()
mpl.rc_file_defaults()
sns.set(font_scale=1.0, rc={'text.usetex': True}, style='darkgrid')

#? Now plots!

#* EE average plots
plt.figure(figsize=(12, 7))

plt.loglog(PCl_nostars.ells, PCl_nostars.num_avg_EE, lw=2, c='cornflowerblue', label='No apodisation, no stars')
# plt.loglog(PCl_nostars.ells, PCl_whstars.num_avg_EE, lw=2, c='purple', label='No apodisation, stars')

plt.loglog(PCl_nostars_apo.ells, PCl_nostars_apo.num_avg_EE, lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_whstars_apo.ells, PCl_whstars_apo.num_avg_EE, lw=2, c='darkseagreen', label='With apodisation, stars')

# plt.loglog(PCl_nostars_apo_pure.ells, PCl_nostars_apo_pure.num_avg_EE, lw=2, c='orange', label='With apodisation, purification, no stars')
# plt.loglog(PCl_whstars_apo_pure.ells, PCl_whstars_apo_pure.num_avg_EE, lw=2, c='darkgoldenrod', label='With apodisation, purification, stars')

plt.loglog(ells[2:], cl_ee[2:] + theory_cl_noise, lw=2, ls='--', c='crimson', label='Theory + Noise')

plt.ylabel(r'$\hat{C}_{\ell}^{EE}$')
plt.xlabel(r'$\ell$')
plt.title(f'Average $C_{{\ell}}^{{EE}}$ - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Avg_EE.pdf')

#* BB average plots
plt.figure(figsize=(12, 7))

plt.loglog(PCl_nostars.ells, np.abs(PCl_nostars.num_avg_BB), lw=2, c='cornflowerblue', label='No apo, no stars')
# plt.loglog(PCl_nostars.ells, np.abs(PCl_whstars.num_avg_BB), lw=2, c='purple', label='No apo, stars')

plt.loglog(PCl_nostars_apo.ells, np.abs(PCl_nostars_apo.num_avg_BB), lw=2, c='mediumseagreen', label='With apo, no stars')
# plt.loglog(PCl_whstars_apo.ells, np.abs(PCl_whstars_apo.num_avg_BB), lw=2, c='darkseagreen', label='With apo, stars')

# plt.loglog(PCl_nostars_apo_pure.ells, np.abs(PCl_nostars_apo_pure.num_avg_BB), lw=2, c='orange', label='With apo, $B$-pure, no stars')
# plt.loglog(PCl_whstars_apo_pure.ells, np.abs(PCl_whstars_apo_pure.num_avg_BB), lw=2, c='darkgoldenrod', label='With apo, $B$-pure, stars')

# plt.semilogx(ells, [theory_cl_noise]*len(ells), lw=2, ls='--', c='crimson', label='Noise level')
plt.semilogx(ells[2:], cl_bb[2:] + theory_cl_noise, lw=2, ls='--', c='crimson', label='Noise level')

plt.ylabel(r'$|\hat{C}_{\ell}^{BB}|$')
plt.xlabel(r'$\ell$')
plt.title(f'Average $C_{{\ell}}^{{BB}}$ - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Avg_BB.pdf')


#* EE-EE variances
plt.figure(figsize=(11, 6))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_EE_EE), lw=2, c='cornflowerblue', label='No apodisation, no stars')
# plt.loglog(PCl_whstars.ells, np.diag(PCl_whstars.num_cov_EE_EE), lw=2, c='hotpink', label='No apodisation, with stars')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_EE_EE), lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_nostars_apo_pure.ells, np.diag(PCl_nostars_apo_pure.num_cov_EE_EE), lw=2, c='orange', label='With apodisation, no stars, \& purification')

plt.loglog(QML_nostars.ells, np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='purple', label='QML without stars')
# plt.loglog(QML_whstars.ells, np.diag(QML_whstars.num_cov_EE_EE), lw=2, c='crimson', label='QML with stars')

plt.ylabel(r'$\Delta \ell = 0$')
plt.xlabel(r'$\ell$')
plt.title(f'Covariances $EE-EE$ - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_EE_EE.pdf')

#* Now plot the EE-EE ratio of PCl to QML
plt.figure(figsize=(11, 6))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='cornflowerblue', label='No apodisation, no stars')
# plt.loglog(PCl_whstars.ells, np.diag(PCl_whstars.num_cov_EE_EE) / np.diag(QML_whstars.num_cov_EE_EE), lw=2, c='hotpink', label='No apodisation, with stars')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_nostars_apo_pure.ells, np.diag(PCl_nostars_apo_pure.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='orange', label='With apodisation, no stars, \& purification')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances $EE-EE$ - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_EE_EE_ratio.pdf')

#* BB-BB variances
plt.figure(figsize=(11, 6))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_BB_BB), lw=2, c='cornflowerblue', label='No apodisation, no stars')
# plt.loglog(PCl_whstars.ells, np.diag(PCl_whstars.num_cov_BB_BB), lw=2, c='hotpink', label='No apodisation, with stars')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_BB_BB), lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_nostars_apo_pure.ells, np.diag(PCl_nostars_apo_pure.num_cov_BB_BB), lw=2, c='orange', label='With apodisation, no stars, \& purification')

plt.loglog(QML_nostars.ells, np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='purple', label='QML without stars')
# plt.loglog(QML_whstars.ells, np.diag(QML_whstars.num_cov_BB_BB), lw=2, c='crimson', label='QML with stars')

plt.ylabel(r'$\Delta \ell = 0$')
plt.xlabel(r'$\ell$')
plt.title(f'Covariances $BB-BB$ - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_BB_BB.pdf')

#* BB-BB variances ratio
plt.figure(figsize=(11, 6))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='cornflowerblue', label='No apodisation, no stars')
# plt.loglog(PCl_whstars.ells, np.diag(PCl_whstars.num_cov_BB_BB) / np.diag(QML_whstars.num_cov_BB_BB), lw=2, c='hotpink', label='No apodisation, with stars')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_nostars_apo_pure.ells, np.diag(PCl_nostars_apo_pure.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='orange', label='With apodisation, no stars, \& purification')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances $BB-BB$ - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_BB_BB_ratio.pdf')

#* Combined ratio plot - TODO
plt.figure(figsize=(7, 4))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='cornflowerblue', label='$EE-EE$ No apo')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='mediumseagreen', label='$EE-EE$ With apo')

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='purple', label='$BB-BB$ No apo')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='lightseagreen', label='$BB-BB$ With apo')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation - no stars')
plt.xlim(right=2 * n_side)

plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_both_ratio2.pdf')

#? Plots without stars
# EE-EE
plt.figure(figsize=(11, 6))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='cornflowerblue', label='No apodisation, no stars')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_nostars_apo_pure.ells, np.diag(PCl_nostars_apo_pure.num_cov_EE_EE) / np.diag(QML_nostars.num_cov_EE_EE), lw=2, c='orange', label='With apodisation, no stars, \& purification')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances $EE-EE$ - without stars - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_EE_EE_ratio_nostars.pdf')

# BB-BB
plt.figure(figsize=(11, 6))

plt.loglog(PCl_nostars.ells, np.diag(PCl_nostars.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='cornflowerblue', label='No apodisation, no stars')
plt.loglog(PCl_nostars_apo.ells, np.diag(PCl_nostars_apo.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='mediumseagreen', label='With apodisation, no stars')
# plt.loglog(PCl_nostars_apo_pure.ells, np.diag(PCl_nostars_apo_pure.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='orange', label='With apodisation, no stars, \& purification')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances $BB-BB$ - without stars - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_BB_BB_ratio_nostars.pdf')

plt.close('all')

import sys
sys.exit()

#? Plots with stars
# EE-EE
plt.figure(figsize=(11, 6))

plt.loglog(PCl_whstars.ells, np.diag(PCl_whstars.num_cov_EE_EE) / np.diag(QML_whstars.num_cov_EE_EE), lw=2, c='cornflowerblue', label='No apodisation')
plt.loglog(PCl_whstars_apo.ells, np.diag(PCl_whstars_apo.num_cov_EE_EE) / np.diag(QML_whstars.num_cov_EE_EE), lw=2, c='mediumseagreen', label='With apodisation')
plt.loglog(PCl_whstars_apo_pure.ells, np.diag(PCl_whstars_apo_pure.num_cov_EE_EE) / np.diag(QML_whstars.num_cov_EE_EE), lw=2, c='orange', label='With apodisation \& purification')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances $EE-EE$ - with stars - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_EE_EE_ratio_withstars.pdf')

# BB-BB
plt.figure(figsize=(11, 6))

plt.loglog(PCl_whstars.ells, np.diag(PCl_whstars.num_cov_BB_BB) / np.diag(QML_whstars.num_cov_BB_BB), lw=2, c='cornflowerblue', label='No apodisation')
plt.loglog(PCl_whstars_apo.ells, np.diag(PCl_whstars_apo.num_cov_BB_BB) / np.diag(QML_nostars.num_cov_BB_BB), lw=2, c='mediumseagreen', label='With apodisation')
plt.loglog(PCl_whstars_apo_pure.ells, np.diag(PCl_whstars_apo_pure.num_cov_BB_BB) / np.diag(QML_whstars.num_cov_BB_BB), lw=2, c='orange', label='With apodisation \& purification')

plt.ylabel(r'PCl to QML')
plt.xlabel(r'$\ell$')
plt.title(f'PCl to QML ratio of covariances $BB-BB$ - with stars - with {ells_per_bin} $\ell$ per bin and ${apo_scale_str}\,^{{\\circ}}$ apodisation')
plt.xlim(right=2 * n_side)

plt.legend()
plt.tight_layout()
plt.savefig(f'Plots_New/{folder}/Diag_BB_BB_ratio_withstars.pdf')

plt.close('all')