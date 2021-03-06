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
from lib.PostProcessing.ParameterFisher import ParamFisher, plot_param_Fisher_triangle, plot_param_Fisher_1D

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
do_Fisher = True

# Only do parameter Fisher calculations if we want to
if do_Fisher:
    print('Producing parameter constraints using Fisher matrix estimates')
    fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96, 'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

    manual_l_max = None

    # * Just sigma_8 and Omega_c
    params = {'Omega_c': 0.27, 'sigma8': 0.75}
    params_latex = [r'\Omega_c', r'\sigma_8']
    d_params = {'Omega_c': 0.27 / 250, 'sigma8': 0.75 / 250}

    param_F_PCl_analytic_stars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.analytic, PCl_whstars, r'Pseudo-$C_{\ell}$', params, params_latex, d_params, num_samples, ells_per_bin=ells_per_bin, manual_l_max=manual_l_max)
    param_F_QML_numeric_stars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.QML, CovType.analytic, QML_whstars, 'QML', params, params_latex, d_params, num_samples, ells_per_bin=ells_per_bin, manual_l_max=manual_l_max)

    param_F_PCl_analytic_stars.compute_param_Fisher()
    param_F_QML_numeric_stars.compute_param_Fisher()

    plot_param_Fisher_triangle([param_F_PCl_analytic_stars, param_F_QML_numeric_stars], output_folder=plots_folder,
                               plot_filename='Omegac_sigma8', plot_title=None)

    # * Now doing Omega_c, Omega_b, sigma_8
    params = {'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75}
    params_latex = [r'\Omega_c', r'\Omega_b', r'\sigma_8']
    d_params = {'Omega_c': 0.27 / 250, 'Omega_b': 0.045 / 250, 'sigma8': 0.75 / 250}

    param_F_PCl_analytic_stars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.analytic, PCl_whstars,
                                             r'Pseudo-$C_{\ell}$', params, params_latex, d_params, num_samples,
                                             ells_per_bin=ells_per_bin, manual_l_max=manual_l_max)
    param_F_QML_numeric_stars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.QML, CovType.analytic, QML_whstars,
                                            'QML', params, params_latex, d_params, num_samples,
                                            ells_per_bin=ells_per_bin, manual_l_max=manual_l_max)

    param_F_PCl_analytic_stars.compute_param_Fisher()
    param_F_QML_numeric_stars.compute_param_Fisher()

    plot_param_Fisher_triangle([param_F_PCl_analytic_stars, param_F_QML_numeric_stars], output_folder=plots_folder,
                               plot_filename='Omegac_Omegab_sigma8', plot_title=None)

    # * Now want to transform our three-parameter Fisher matrix into one for Omega_m and sigma_8

    # Jacobian matrix for this (Omega_c, Omega_b, sigma_8) -> (Omega_m, sigma_8) transformation
    jacobian_matrix = np.array([[1, 0], [1, 0], [0, 1]])

    # Fiducial values for our new set of parameters
    derived_fiducial_values = np.array([params['Omega_c'] + params['Omega_b'], params['sigma8']])

    derived_parameters_names = ['omega_m', 'sigma_8']
    derived_parameters_latex = [r'\Omega_m', r'\sigma_8']

    param_F_PCl_analytic_stars.transform_param_Fisher(jacobian_matrix, derived_parameters_names,
                                                      derived_parameters_latex, derived_fiducial_values)

    param_F_QML_numeric_stars.transform_param_Fisher(jacobian_matrix, derived_parameters_names,
                                                     derived_parameters_latex, derived_fiducial_values)

    plot_param_Fisher_triangle([param_F_PCl_analytic_stars, param_F_QML_numeric_stars], output_folder=plots_folder,
                               plot_filename='Omegam_sigma8', plot_title=None)

    # * Now want to transform our two-parameter Fisher matrix into a 1D one for S_8

    # Function to compute the Jacobian matrix  for this (Omega_m, sigma_8) -> (S_8) transformation
    def omegam_sigma8_to_s8(omega_m, sigma_8):
        d_s8_d_omegam = (1 / 2) * sigma_8 * (1 / np.sqrt(0.3 * omega_m))

        d_s8_d_sigma8 = np.sqrt(omega_m / 0.3)

        return np.array([[d_s8_d_omegam], [d_s8_d_sigma8]])

    # Construct Jacobian matrix using fiducial values
    jacobian_matrix = omegam_sigma8_to_s8(params['Omega_c'] + params['Omega_b'], params['sigma8'])

    # Fiducial values for S_8
    derived_fiducial_values = np.array([params['sigma8'] * np.sqrt((params['Omega_c'] + params['Omega_b']) / 0.3)])

    derived_parameters_names = ['S_8']
    derived_parameters_latex = [r'S_8']

    param_F_PCl_analytic_stars.transform_param_Fisher(jacobian_matrix, derived_parameters_names,
                                                      derived_parameters_latex, derived_fiducial_values)

    param_F_QML_numeric_stars.transform_param_Fisher(jacobian_matrix, derived_parameters_names,
                                                     derived_parameters_latex, derived_fiducial_values)

    plot_param_Fisher_1D([param_F_PCl_analytic_stars, param_F_QML_numeric_stars], output_folder=plots_folder,
                         plot_filename='S8', plot_title=None)

    do_range_of_l_max = False
    if do_range_of_l_max:
        print('Computing the figure of merit over a range of l_max values \nl_max:', end=' ', flush=True)

        # Range of l_max values under consideration
        l_max_range = np.logspace(3, 9, num=7, base=2, dtype=int)

        # Re-set parameters to just Omega_c and sigma_8
        params = {'Omega_c': 0.27, 'sigma8': 0.75}
        params_latex = [r'\Omega_c', r'\sigma_8']
        d_params = {'Omega_c': 0.27 / 250, 'sigma8': 0.75 / 250}

        # Lists to store figure of merits in
        fom_qml_nostars = []
        fom_qml_whstars = []
        fom_pcl_nostars = []
        fom_pcl_whstars = []
        fom_pcl_nostars_apo = []
        fom_pcl_whstars_apo = []

        for l_max_iter in l_max_range:
            print(l_max_iter, end=' ', flush=True)

            param_F_QML_whstars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.QML, CovType.analytic,
                                              QML_whstars, 'QML', params, params_latex, d_params, num_samples,
                                              ells_per_bin=ells_per_bin, manual_l_max=l_max_iter)

            param_F_PCl_whstars = ParamFisher(n_side, fiducial_cosmology, 1.0, ClType.PCl, CovType.analytic,
                                              PCl_whstars, r'Pseudo-Cl', params, params_latex, d_params,
                                              num_samples, ells_per_bin=ells_per_bin, manual_l_max=l_max_iter)

            param_F_QML_whstars.compute_param_Fisher()
            param_F_PCl_whstars.compute_param_Fisher()

            # Store the figure of merit in our data lists
            fom_qml_whstars.append(param_F_QML_whstars.fig_of_merit)
            fom_pcl_whstars.append(param_F_PCl_whstars.fig_of_merit)

        # * Now plot the figure of merit as a function of l_max
        print('')
        import matplotlib.ticker as ticker
        plt.rcParams['text.usetex'] = True

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.loglog(l_max_range, fom_qml_whstars, lw=2, c='cornflowerblue', label='QML', ls='-')

        ax.loglog(l_max_range, fom_pcl_whstars, lw=2, c='mediumseagreen', label=r'Pseudo-$C_{\ell}$', ls='-')

        ax.set_xlabel(r'$\ell_\textrm{max}$')
        ax.set_ylabel('Figure of merit')

        # Use log base 2 for x-axis
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        ax.legend(ncol=1)

        plt.tight_layout()
        plt.savefig(f'{plots_folder}/figure_of_merit_vs_l_max_stars.pdf')

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