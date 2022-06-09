"""
File to investigate the effects of adding stars to the mask and their impact on the PCl covariance matrix
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

import seaborn as sns

sns.set(font_scale=1.0, rc={'text.usetex': True})

from lib.PostProcessing.PCl_covariances import PCl_covariances
from lib.PostProcessing.QML_covariances import QML_covariances
from lib.PostProcessing.ParameterFisher import ParamFisher, plot_param_Fisher_triangle, plot_param_Fisher_1D
from lib.PostProcessing.Enums import ClType, CovType

import matplotlib as mpl

mpl.use('Agg')

# The output folder under ./Plots
folder = 'Final/AnalyticCovRatio'
plots_folder = f'../data/Plots/{folder}'

# See if folder doesn't exist already, then make it if not
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

# HEALPix map resolution
n_side = 256

# Number of ells per bin
ells_per_bin = 1

num_samples = 5_000
use_pixwin = False
manual_l_max = None

# The apodisation scale (in deg)
apo_scale = 2.0

star_mask = hp.read_map('../data/masks/StarMask_N256.fits', dtype=float)

mask_nostars = hp.read_map('../data/Masks/SkyMask_N256_nostars.fits', dtype=float)

mask_whstars = mask_nostars * star_mask

# Apodize the main mask and mask with stars
mask_nostars_apo = nmt.mask_apodization(mask_nostars.copy(), aposize=apo_scale, apotype='C2')
mask_whstars_apo = nmt.mask_apodization(mask_whstars.copy(), aposize=apo_scale, apotype='C2')

# Compute the power spectrum of the mask with and without stars
ells = np.arange(2, 3 * n_side)
mask_nostars_cls = hp.anafast(mask_nostars)[2:]
mask_whstars_cls = hp.anafast(mask_whstars)[2:]

# Plot the power spectrum of the mask with and without stars
fig, ax = plt.subplots(figsize=(5, 2.5))

ax.loglog(ells, ells * (ells + 1) * mask_whstars_cls / (2 * np.pi), lw=2, c='purple', label='With stars')

ax.loglog(ells[::2], ells[::2] * (ells[::2] + 1) * mask_nostars_cls[::2] / (2 * np.pi), lw=2, c='cornflowerblue',
          label='No stars')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2 \pi$')

ax.legend()
fig.tight_layout()
fig.savefig(f'{plots_folder}/Mask_power_spectrum_stars.pdf')
plt.close(fig)

# Noise values
from scipy import constants as consts

intrinsic_gal_ellip = 0.21
avg_gal_den = 30
area_per_pix = 148510661 / (12 * n_side * n_side)
num_gal_per_pix = avg_gal_den * area_per_pix

theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (consts.arcminute ** 2))
noise_std = intrinsic_gal_ellip / np.sqrt(num_gal_per_pix)

# * Compute power spectrum
from scipy import stats
import pyccl as ccl

ells = np.arange(0, 3 * n_side)
redshift_range = np.linspace(0.0, 3.0, 500)

n_ell = len(ells[2:])

# Create new galaxy distribution at our current redshift
dNdz = stats.norm.pdf(redshift_range, loc=1.0, scale=0.15)

cosmo = ccl.Cosmology(
    **{'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96, 'm_nu': 0.0, 'T_CMB': 2.7255,
       'w0': -1.0, 'wa': 0.0})

# Lensing bins at current redshift
field_E = ccl.WeakLensingTracer(cosmo, dndz=(redshift_range, dNdz))

# Compute Cl values
cl_ee = ccl.angular_cl(cosmo, field_E, field_E, ells)

cl_ee[:2] = 0.0

cl_eb = np.zeros_like(cl_ee)
cl_bb = np.zeros_like(cl_ee)

# * Set up PCl covariance objects
PCl_nostars = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False,
                              mask=mask_nostars, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise,
                              label='no_stars_no_apo', use_pixwin=use_pixwin)
PCl_whstars = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False,
                              mask=mask_whstars, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise,
                              label='wh_stars_no_apo_presentation', use_pixwin=use_pixwin)

# Compute analytic covariance matrix for the case with stars for with and without apodisation
PCl_nostars.compute_analytic_covariance()
PCl_whstars.compute_analytic_covariance()

# Plot the ratio of the analytic covariance matrices
fig, ax = plt.subplots(figsize=(5, 4))

data = np.log10(np.abs(PCl_whstars.ana_cov_EE_EE / PCl_nostars.ana_cov_EE_EE))[0: 2 * n_side - 1: 2,
       0: 2 * n_side - 1: 2]

vmax = np.max(data)

plot = ax.imshow(data, cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', extent=(2, 2 * n_side, 2, 2 * n_side))
cbar = fig.colorbar(plot, ax=ax)

cbar.set_label(label=r'$\textrm{Log}_{10}[ | \mathbf{C}^{\prime} / \mathbf{C} | ]$',
               fontsize=10)

ax.set_xlabel(r'$\ell_{1}$')
ax.set_ylabel(r'$\ell_{2}$')

fig.tight_layout()

plt.savefig(f'{plots_folder}/RatioOfAnalyticCovariances_stars.pdf')

plt.close('all')

# Read in QML data
QML_nostars = QML_covariances(n_side,
                              '/home/maraio/Codes/WeakLensingQML/data/numerical_fishers/ConjGrad_Fisher_N256_nummaps10_noise30_EB_nostars.dat',
                              ells_per_bin=ells_per_bin)
QML_whstars = QML_covariances(n_side,
                              '/home/maraio/Codes/WeakLensingQML/data/numerical_fishers/ConjGrad_Fisher_N256_nummaps10_noise30_EB_whstars.dat',
                              ells_per_bin=ells_per_bin)

fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96, 'm_nu': 0.0,
                      'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

# Cosmological parameters to use in our Fisher forecast
params = {'Omega_c': 0.27, 'sigma8': 0.75}
params_latex = [r'\Omega_c', r'\sigma_8']
d_params = {'Omega_c': 0.27 / 250, 'sigma8': 0.75 / 250}

# Create parameter Fisher objects for our different cases
param_F_PCl_analytic_nostars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.analytic, PCl_nostars,
                                           r'P$C_{\ell}$ without stars', params, params_latex, d_params, num_samples,
                                           ells_per_bin=ells_per_bin, manual_l_max=None)
param_F_PCl_analytic_whstars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.analytic, PCl_whstars,
                                           r'P$C_{\ell}$ with stars', params, params_latex, d_params, num_samples,
                                           ells_per_bin=ells_per_bin, manual_l_max=None)
param_F_QML_numeric_nostars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.QML, CovType.analytic, QML_nostars,
                                          'QML no stars', params, params_latex, d_params, num_samples,
                                          ells_per_bin=ells_per_bin, manual_l_max=manual_l_max)
param_F_QML_numeric_whstars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.QML, CovType.analytic, QML_whstars,
                                          'QML with stars', params, params_latex, d_params, num_samples,
                                          ells_per_bin=ells_per_bin, manual_l_max=manual_l_max)

param_F_PCl_analytic_nostars.compute_param_Fisher()
param_F_PCl_analytic_whstars.compute_param_Fisher()
param_F_QML_numeric_nostars.compute_param_Fisher()
param_F_QML_numeric_whstars.compute_param_Fisher()

plot_param_Fisher_triangle([param_F_PCl_analytic_nostars, param_F_PCl_analytic_whstars,
                            param_F_QML_numeric_nostars, param_F_QML_numeric_whstars],
                           output_folder=plots_folder,
                           plot_filename='Omegac_sigma8_stars', plot_title='')
