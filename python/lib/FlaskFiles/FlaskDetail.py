import os
import numpy as np
from scipy import stats, constants as consts
import pyccl as ccl
from enum import Enum
from random import randint
import subprocess
import healpy as hp


class Distribution(Enum):
    Gaussian = 1
    LogNormal = 2

    def __str__(self):
        if self.value == 1:
            return 'Gaussian'

        else:
            return 'LogNormal'

    def to_str_upper(self):
        return self.__str__().upper()


def XavierShift(z):
    """
    Function that takes the redshift value and computes the "shift" parameter for the log-normal field that it will
    be associated with.

    Taken from the Flask source code
    """
    a0 = 0.2
    s0 = 0.568591

    return a0 * (((z * s0) ** 2 + z * s0 + 1) / (z * s0 + 1) - 1)


class FlaskRun:

    def __init__(self, n_side, output_n_side, dist_type, redshift, folder_path, flask_executable):
        # The N_side parameter for all generated maps
        self.n_side = n_side
        self.n_pix = 12 * self.n_side * self.n_side

        # The final output resolution of the maps wanted after (optional) downgrading
        self.n_side_output = output_n_side

        self.l_max = 3 * self.n_side - 1
        self.ells = np.arange(2, self.l_max + 1)

        # The type of distribution that will be used (either Gaussian or LogNormal)
        self.dist_type = dist_type

        # The central redshift for this cosmic shear tomographic bin
        self.redshift = redshift

        # Where should we store data for this run?
        self.folder_path = folder_path

        # Strip out the ending backslash, if one is provided
        if self.folder_path[-1] == '/':
            self.folder_path = self.folder_path[:-1]

        # See if the provided folder path already exists, if not then make it
        if not os.path.isdir(self.folder_path):
            os.makedirs(self.folder_path)

        # Also see if the ./Maps subdirectory exists, and if not make it too
        if not os.path.isdir(f'{self.folder_path}/Maps_N{self.n_side}/'):
            os.makedirs(f'{self.folder_path}/Maps_N{self.n_side}/')

        # See if the subdirectory for the down-graded maps exists AND we are downgrading, if not then make it
        if self.n_side != self.n_side_output:
            if not os.path.isdir(f'{self.folder_path}/Maps_N{self.n_side_output}/'):
                os.makedirs(f'{self.folder_path}/Maps_N{self.n_side_output}/')

        # The location of the Flask executable file
        self.flask_executable = flask_executable

        # Fiducial cosmology values to compute the theory spectra with
        self.fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96,
                                   'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

        # Noise values
        intrinsic_gal_ellip = 0.21
        avg_gal_den = 3
        area_per_pix = 148510661 / (12 * n_side * n_side)
        num_gal_per_pix = avg_gal_den * area_per_pix

        # theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (consts.arcminute ** 2))
        self.noise_std = intrinsic_gal_ellip / np.sqrt(num_gal_per_pix)

    def compute_theory_spectra(self):
        """
        Function to compute the shear power spectrum at the given redshift and save the values accordingly
        """
        print('Evaluating the theory power spectrum values')

        # Create our array of redshift values to evaluate dN/dz at
        redshift_range = np.linspace(0.0, 3.0, 500)

        # Create new galaxy distribution at our current redshift
        dNdz = stats.norm.pdf(redshift_range, loc=self.redshift, scale=0.15)

        cosmo = ccl.Cosmology(**self.fiducial_cosmology)

        # Lensing bins at current redshift
        field_E = ccl.WeakLensingTracer(cosmo, dndz=(redshift_range, dNdz))

        # Compute Cl values
        Cl_EE = ccl.angular_cl(cosmo, field_E, field_E, self.ells)

        # Flask requires the convergence Cl values, to we need to turn Cl_EE into Cl_kappakappa
        Cl_kappakappa = ((self.ells * (self.ells + 1)) / ((self.ells - 1) * (self.ells + 2))) * Cl_EE

        # Save the (ell, Cl_EE) combination to the file to be read in by Flask for both convergence and shear-EE
        np.savetxt(f'{self.folder_path}/TheoryClsShear-f1z1f1z1.dat', np.array([self.ells, Cl_EE]).T)
        np.savetxt(f'{self.folder_path}/TheoryClsConvergence-f1z1f1z1.dat', np.array([self.ells, Cl_kappakappa]).T)

    def generate_flask_config_file(self):
        """
        Function to generate the Flask .config file which is used for generating the maps with Flask
        """
        file = open(f'{self.folder_path}/FlaskConfig_N{self.n_side}.config', 'w')

        file.write('# This is a config file auto-generated\n\n')

        file.write('## Simulation basics ##\n\n')
        file.write(f'DIST: \t {self.dist_type.to_str_upper()} \n')
        file.write(f'RNDSEED: \t {randint(1, 100000)} \n')
        file.write('POISSON: \t 0 \n')

        file.write('\n## Cosmology ##\n\n')
        file.write('OMEGA_m: \t 0.3 \n')
        file.write('OMEGA_L: \t 0.7 \n')
        file.write('W_de: \t -1.0 \n\n')
        file.write('ELLIP_SIGMA: \t 0.21 \n')
        file.write('GALDENSITY: \t 3 \n\n')

        file.write('\n## Input data ## \n\n')
        file.write(f'FIELDS_INFO: \t {self.folder_path}/fields_info.dat \n')
        file.write('CHOL_IN_PREFIX: \t 0 \n')
        file.write(f'CL_PREFIX: \t {self.folder_path}/TheoryClsConvergence- \n')
        file.write('ALLOW_MISS_CL: \t 0 \n')
        file.write('SCALE_CLS: \t 1.0 \n')
        file.write('WINFUNC_SIGMA: \t -1 \n')
        file.write('APPLY_PIXWIN: \t 0 \n')
        file.write('SUPPRESS_L: \t -1 \n')
        file.write('SUP_INDEX: \t -1 \n\n')

        file.write('\n## Survey selection functions ##\n\n')
        file.write('SELEC_SEPARABLE: \t 1 \n')
        file.write('SELEC_PREFIX: \t 0 \n')
        file.write('SELEC_Z_PREFIX: \t 0 \n')
        file.write('SELEC_SCALE: \t 1 \n')
        file.write('SELEC_TYPE: \t 0 \n')
        file.write('STARMASK: \t 0 \n\n')

        file.write('\n## Multipole information ##\n\n')
        file.write('EXTRAP_DIPOLE: \t 0 \n')
        file.write(f'LRANGE: \t 2 {self.l_max} \n')
        file.write('CROP_CL: \t 0 \n')

        if self.dist_type == Distribution.Gaussian:
            file.write(f'SHEAR_LMAX: \t {self.l_max} \n')

        else:
            file.write(f'SHEAR_LMAX: \t {self.n_side} \n')

        file.write(f'NSIDE: \t {self.n_side} \n')
        file.write('USE_HEALPIX_WGTS: \t 0 \n\n')

        file.write('\n## Covariance matrix regularisation##\n\n')
        file.write('MINDIAG_FRAC: \t 1e-12 \n')
        file.write('BADCORR_FRAC: \t 0 \n')
        file.write('REGULARIZE_METHOD: \t 1 \n')
        file.write('NEW_EVAL: \t 1e-18 \n')
        file.write('REGULARIZE_STEP: \t 0.0001 \n')
        file.write('REG_MAXSTEPS: \t 1000 \n')
        file.write('ADD_FRAC: \t 1e-10 \n')
        file.write('ZSEARCH_TOL: \t 0.0001 \n\n')

        file.write('\n## Output ##\n\n')

        file.write('EXIT_AT: \t SHEAR_FITS_PREFIX \n')

        file.write('FITS2TGA: \t 0 \n')
        file.write('USE_UNSEEN: \t 0 \n')
        file.write(f'LRANGE_OUT: \t 2 {self.l_max} \n')
        file.write('MMAX_OUT: \t -1 \n')
        file.write('ANGULAR_COORD: \t 0 \n')
        file.write('DENS2KAPPA: \t 0 \n\n')

        file.write('FLIST_OUT: \t 0 \n')
        file.write('SMOOTH_CL_PREFIX: \t 0 \n')
        file.write('XIOUT_PREFIX: \t 0 \n')
        file.write('GXIOUT_PREFIX: \t 0 \n')
        file.write('GCLOUT_PREFIX: \t 0 \n')
        file.write('COVL_PREFIX: \t 0 \n')
        file.write('REG_COVL_PREFIX: \t 0 \n')
        file.write('REG_CL_PREFIX: \t 0 \n')
        file.write('CHOLESKY_PREFIX: \t 0 \n')
        file.write('AUXALM_OUT: \t 0 \n')
        file.write('RECOVAUXCLS_OUT: \t 0 \n')
        file.write('AUXMAP_OUT: \t 0 \n')
        file.write('DENS2KAPPA_STAT: \t 0 \n')

        file.write('MAP_OUT: \t 0 \n')
        file.write('MAPFITS_PREFIX: \t 0 \n')

        file.write('RECOVALM_OUT: \t 0 \n')
        file.write('RECOVCLS_OUT: \t 0 \n')

        file.write('SHEAR_ALM_PREFIX: \t 0 \n')
        file.write(f'SHEAR_FITS_PREFIX: \t {self.folder_path}/Maps_N{self.n_side}/Map- \n')
        file.write('SHEAR_MAP_OUT: \t 0 \n')

        file.write('MAPWERFITS_PREFIX: \t 0 \n')
        file.write('MAPWER_OUT: \t 0 \n')

        file.write('ELLIPFITS_PREFIX: \t 0 \n')
        file.write('ELLIP_MAP_OUT: \t 0 \n')

        file.write('CATALOG_OUT: \t 0 \n')
        file.write('\nCATALOG_COLS: \t theta phi z kappa gamma1 gamma2 ellip1 ellip2\n')
        file.write('\nCAT_COL_NAMES: \t theta phi z kappa gamma1 gamma2 ellip1 ellip2\n')

        file.write('REDUCED_SHEAR: \t 1 \n')
        file.write('CAT32BIT: \t 1 \n')

        file.close()

    def generate_field_file(self):
        """
        File to generate the field information file that Flask requires
        """

        file = open(f'{self.folder_path}/fields_info.dat', 'w')

        file.write('# Fields information file that has been auto-generated\n')
        file.write('# Field number, z bin number, mean, shift, field type (1: galaxy, 2: shear), z_min, z_max\n')

        # file.write(f'1 \t 1 \t 0 \t {XavierShift(self.redshift):.6f} \t 2 \t 0 \t 3 \n')
        file.write(f'1 \t 1 \t 0 \t 0.01214 \t 2 \t 0 \t 3 \n')

        file.close()

    def run_flask(self, run_num=1):
        """
        Function to run a single instance of Flask to generate a single realisation of the cosmic shear maps
        """

        # Launch Flask as a sub-process
        command = subprocess.run(f'{self.flask_executable} {self.folder_path}/FlaskConfig_N{self.n_side}.config' +
                                 f' RNDSEED: {randint(1, 10_000_000)}' +
                                 f' SHEAR_FITS_PREFIX: {self.folder_path}/Maps_N{self.n_side}/Map{run_num}-',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                 cwd=self.folder_path, shell=True)

        # See if Flask ran successfully or not
        if command.returncode == 0:
            pass
            # Write Flask's output to a file
            # output_file = open(self.folder_path + 'FlaskOutput.txt', 'w')
            # output_file.write(command.stdout)
            #
            # output_file.write('\nWarnings generated by Flask:\n' + command.stderr + '\n')
            #
            # output_file.close()

        # Else, there was an error somewhere, print this error and exit the program
        else:
            print('Flask did not run successfully, hopefully the error is contained below')
            print(command.stdout)
            print(command.stderr)

            raise RuntimeError('Flask did not run successfully! :(')

    def run_flask_ensemble(self, num_maps, add_noise=False):
        """
        Function to run Flask a number of times, as given by the num_maps argument
        """
        print(f'Generating {num_maps} maps for Nside = {self.n_side}, Nside output = {self.n_side_output}, with '
              f'distribution = {self.dist_type}, and add_noise = {add_noise}')

        # Go through and generate the requested number of maps
        for map_num in range(num_maps):
            if np.mod(map_num, 50) == 0:
                print(map_num, end=' ', flush=True)

            self.run_flask(run_num=map_num)

            # See if we're manually adding shape noise to our generated maps
            if add_noise:
                # Read in our shear maps that have been generated by Flask
                map_kappa, map_gamma1, map_gamma2 = hp.read_map(
                    f'{self.folder_path}/Maps_N{self.n_side}/Map{map_num}-f1z1.fits',
                    field=[0, 1, 2], dtype=float)

                # Generate random realisation of our shape noise
                noise_gamma1 = np.random.normal(loc=0, scale=self.noise_std, size=self.n_pix)
                noise_gamma2 = np.random.normal(loc=0, scale=self.noise_std, size=self.n_pix)

                # Add noise to our signal maps
                map_gamma1 += noise_gamma1
                map_gamma2 += noise_gamma2

                # Overwrite the existing shear maps
                hp.write_map(f'{self.folder_path}/Maps_N{self.n_side}/Map{map_num}-f1z1.fits',
                             [map_kappa, map_gamma1, map_gamma2],
                             overwrite=True, fits_IDL=False, dtype=np.float64)

            # See if we're then downgrading our maps from a higher resolution to a lower one
            if self.n_side_output != self.n_side:
                # Read in the maps
                map_kappa, map_gamma1, map_gamma2 = hp.read_map(
                    f'{self.folder_path}/Maps_N{self.n_side}/Map{map_num}-f1z1.fits',
                    field=[0, 1, 2], dtype=float)

                # Downgrade the maps
                map_kappa = hp.ud_grade(map_kappa, self.n_side_output)
                map_gamma1 = hp.ud_grade(map_gamma1, self.n_side_output)
                map_gamma2 = hp.ud_grade(map_gamma2, self.n_side_output)

                hp.write_map(f'{self.folder_path}/Maps_N{self.n_side_output}/Map{map_num}-f1z1.fits',
                             [map_kappa, map_gamma1, map_gamma2],
                             overwrite=True, fits_IDL=False, dtype=np.float64)

        print('...Done!')
