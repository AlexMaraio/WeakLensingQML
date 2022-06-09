"""
File to plot the ratio of the analytic Cl covariance matrix for the Pseudo-Cl method for the case of no and applying
apodisation to the mask
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

import seaborn as sns
sns.set(font_scale=1.0, rc={'text.usetex': True})

from lib.PostProcessing.PCl_covariances import PCl_covariances

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

# The apodisation scale (in deg)
apo_scale = 2.0

star_mask = hp.read_map('../data/masks/StarMask_N256.fits', dtype=float)

mask_nostars = hp.read_map('../data/Masks/SkyMask_N256_nostars.fits', dtype=float)

mask_whstars = mask_nostars * star_mask

# Apodize the main mask and mask with stars
mask_whstars_apo = nmt.mask_apodization(mask_whstars.copy(), aposize=apo_scale, apotype='C2')

# Compute the power spectrum of the mask with and without apodisation
ells = np.arange(2, 3 * n_side)
mask_whstars_cls = hp.anafast(mask_whstars)[2:]
mask_whstars_apo_cls = hp.anafast(mask_whstars_apo)[2:]

# Plot the power spectrum of the mask with and without apodisation
fig, ax = plt.subplots(figsize=(5, 2.5))

ax.loglog(ells, ells * (ells + 1) * mask_whstars_cls / (2 * np.pi), lw=2, c='cornflowerblue', label='No apodisation')

ax.loglog(ells, ells * (ells + 1) * mask_whstars_apo_cls / (2 * np.pi), lw=2, c='purple', label='With apodisation')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2 \pi$')

ax.legend()
fig.tight_layout()
fig.savefig(f'{plots_folder}/Mask_power_spectrum_apodisation.pdf')
plt.close(fig)

# Noise values
from scipy import constants as consts
intrinsic_gal_ellip = 0.21
avg_gal_den = 3
area_per_pix = 148510661 / (12 * n_side * n_side)
num_gal_per_pix = avg_gal_den * area_per_pix

theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (consts.arcminute ** 2))
noise_std = intrinsic_gal_ellip / np.sqrt(num_gal_per_pix)

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
PCl_whstars = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False, mask=mask_whstars, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise, label='wh_stars_no_apo_presentation', use_pixwin=use_pixwin)

PCl_whstars_apo = PCl_covariances(n_side, cl_ee, cl_eb, cl_bb, ells_per_bin=ells_per_bin, purify_E=False, purify_B=False, mask=mask_whstars_apo, num_samples=num_samples, noise_std=noise_std, cl_noise=theory_cl_noise, label='wh_stars_wh_apo_presentation', use_pixwin=use_pixwin)

# Compute analytic covariance matrix for the case with stars for with and without apodisation
PCl_whstars.compute_analytic_covariance()
PCl_whstars_apo.compute_analytic_covariance()

# Plot the ratio of the analytic covariance matrices
fig, ax = plt.subplots(figsize=(5, 4))

data = np.log10(np.abs(PCl_whstars_apo.ana_cov_EE_EE / PCl_whstars.ana_cov_EE_EE))[0: 2 * n_side - 1, 0: 2 * n_side - 1]

vmax = np.max(data)

plot = ax.imshow(data, cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', extent=(2, 2 * n_side, 2, 2 * n_side))
cbar = fig.colorbar(plot, ax=ax)

cbar.set_label(label=r'$\textrm{Log}_{10}[ | \mathbf{C}^{\prime} / \mathbf{C} | ]$',
               fontsize=10)

ax.set_xlabel(r'$\ell_{1}$')
ax.set_ylabel(r'$\ell_{2}$')

fig.tight_layout()

plt.savefig(f'{plots_folder}/RatioOfAnalyticCovariances_apodisation.pdf')

plt.close('all')
