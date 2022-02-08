# File that contains all the necessary information about a QML run
import gc
import itertools
import numpy as np
import numba as na
import pyccl as ccl
from scipy import constants as sciconst
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from .Enums import SpecType
from .Field import Field


sns.set(font_scale=1.2, rc={'text.usetex': True})
mpl.rcParams["savefig.dpi"] = 250


# Function to evaluate the Fisher matrix at a given (ell1, ell2) value
@na.njit(parallel=True)
def evaluate_Fisher(ell1, ell2, Y_inv_cov_Y1, Y_inv_cov_Y2):
    idx_min1 = ell1 ** 2 - 4
    idx_min2 = ell2 ** 2 - 4
    idx_max1 = idx_min1 + (2 * ell1 + 1)
    idx_max2 = idx_min2 + (2 * ell2 + 1)

    summand = 0
    for k_ell1 in na.prange(idx_min1, idx_max1):
        for k_ell2 in range(idx_min2, idx_max2):
            summand += Y_inv_cov_Y1[k_ell1, k_ell2] * Y_inv_cov_Y2[k_ell2, k_ell1]

    return summand / 2


# Function to evaluate the QML y_ell value at a given ell value for two maps
def evaluate_y_ell(ell, input_map_cov_Y1, input_map_cov_Y2):
    idx_min = ell ** 2 - 4
    idx_max = idx_min + (2 * ell + 1)

    # Optimised version for two maps
    return np.sum(input_map_cov_Y1[idx_min: idx_max] * np.conj(input_map_cov_Y2[idx_min: idx_max])).real / 2


# Vectorize the above function over the input ell values
evaluate_y_ell = np.vectorize(evaluate_y_ell, otypes=[float], excluded=[1, 2])


def evaluate_y_ell_TE(ell, input_map_cov_Y1, input_map_cov_Y2):
    idx_min = ell ** 2 - 4
    idx_max = idx_min + (2 * ell + 1)

    # Optimised version for two maps
    return np.sum(input_map_cov_Y1[idx_min: idx_max] * np.conj(input_map_cov_Y2[idx_min: idx_max]) +
                  input_map_cov_Y2[idx_min: idx_max] * np.conj(input_map_cov_Y1[idx_min: idx_max])).real / 2


# Vectorize the above function over the input ell values
evaluate_y_ell_TE = np.vectorize(evaluate_y_ell_TE, otypes=[float], excluded=[1, 2])


# Function to evaluate the noise bias
def evaluate_b_ell_auto(ell, N_cov_inv_Y_in, cov_inv_Y_in, offset):
    # Find the correct index for the P_tilde matrix
    idx_min = ell ** 2 - 4 + offset
    idx_max = idx_min + (2 * ell + 1)

    return np.sum(N_cov_inv_Y_in[:, idx_min: idx_max] * cov_inv_Y_in[:, idx_min: idx_max]).real / 2


# Vectorize our noise function over ell
evaluate_b_ell_auto = np.vectorize(evaluate_b_ell_auto, otypes=[float], excluded=[1, 2, 3])


def evaluate_b_ell_cross(ell, N_cov_inv_Y_in, cov_inv_Y_in, offset1, offset2):
    # Find the correct index for the P_tilde matrix
    idx_min_1 = ell ** 2 - 4 + offset1
    idx_min_2 = ell ** 2 - 4 + offset2
    idx_max_1 = idx_min_1 + (2 * ell + 1)
    idx_max_2 = idx_min_2 + (2 * ell + 1)

    return (np.sum(N_cov_inv_Y_in[:, idx_min_1: idx_max_1] * cov_inv_Y_in[:, idx_min_2: idx_max_2]) +
            np.sum(N_cov_inv_Y_in[:, idx_min_2: idx_max_2] * cov_inv_Y_in[:, idx_min_1: idx_max_1])).real / 2


# Vectorize our noise function over ell
evaluate_b_ell_cross = np.vectorize(evaluate_b_ell_cross, otypes=[float], excluded=[1, 2, 3])


# noinspection PyPep8Naming
class QML:

    def __init__(self, n_side, l_max, redshift, mask, Y_matrix):
        self.n_side = n_side
        self.n_pix = 12 * (self.n_side ** 2)
        self.l_max = l_max
        self.num_l_modes = self.l_max - 1
        self.num_m_modes = (self.l_max + 1) ** 2 - 4

        # Store the redshift for this field
        self.redshift = redshift

        # Create ranges of ell values
        self.ells = np.arange(2, self.l_max + 1)

        # Set the mask
        self.mask = mask

        # Compute number of pixels active in mask
        self.n_pix_mask = self.mask.sum()

        # Compute the mask's f_sky
        self.f_sky = self.mask.sum() / self.mask.size

        print(f'Creating QML class with N_side of {self.n_side}, l_max of {self.l_max},'
              f' n_pix_mask of {self.n_pix_mask}, and an f_sky of {self.f_sky*100:.2f}%')

        # If we've already computed the Y_matrix in another class, then take a reference here
        self.Y_matrix = Y_matrix.Y_matrix
        self.Y_matrix_dagger = Y_matrix.Y_matrix_dagger

        # Initialise fields for our T & E fields which we want to recover.
        # field_T = Field(spec='T', spin=0)
        # field_E = Field(spec='E', spin=2)
        # field_B = Field(spec='B', spin=2)

        # self.fields = [field_E]
        # self.all_fields = [field_E, field_B]

        self.fields = ['E']
        self.all_fields = ['E', 'B']

        # Combine our list of individual fields to  a list of spectra, where each entry is unique.
        # i.e. go from T1 & E1 --> T1T1, T1E1, E1E1
        self.spectra = list(itertools.combinations_with_replacement(self.fields, 2))
        self.all_spectra = list(itertools.combinations_with_replacement(self.all_fields, 2))  # Also include B's here

        # self.all_spectra_str = [spec[0].spec + spec[1].spec for spec in self.all_spectra]

        # Number of unique spectra in this class, including all B-modes
        self.num_spectra = len(self.all_spectra)

        # Fiducial cosmology values to compute the theory spectra with
        self.fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.27, 'Omega_b': 0.045, 'sigma8': 0.75, 'n_s': 0.96,
                                   'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

        # Theory Cl spectra values
        self.cl_EE_theory = np.zeros(self.l_max - 1)
        self.cl_EB_theory = np.zeros(self.l_max - 1)
        self.cl_BB_theory = np.zeros(self.l_max - 1)

        # Create variables to store the S_tilde arrays in
        self.S_tilde_EE = np.zeros(self.num_m_modes)
        self.S_tilde_BB = np.zeros(self.num_m_modes)

        # The pixel covariance matrix and its inverse
        self.cov = None
        self.cov_inv = None

        # Inverse covariance @ Y_matrix
        self.cov_inv_Y = None

        # Large matrix of Y.H @ C^-1 @ Y
        self.Y_dagger_cov_inv_Y = None

        # Full Fisher matrix, and its inverse
        self.F = None
        self.F_inv = None

        # Noise bias values
        self.b_ell = None

        # Recovered QML Cl values
        self.QML_Cl_EE = None
        self.QML_Cl_BB = None

        # Noise values
        intrinsic_gal_ellip = 0.21
        avg_gal_den = 30  # We have approx 30 galaxies per arc-min^2
        area_per_pix = 1.49E8 / self.n_pix
        num_gal_per_pix = avg_gal_den * area_per_pix

        # Compute the noise st.dev. and variance
        self.noise_std = intrinsic_gal_ellip / np.sqrt(num_gal_per_pix)
        self.noise_var = self.noise_std ** 2

        # Using the noise parameters, compute the theory noise Cl
        self.theory_cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (sciconst.arcminute ** 2))

        # Using the noise values, compute the noise array for a single map
        temp_noise_array = np.array([self.noise_var] * self.n_pix_mask)

        # Join the single array to form the noise arrays for Q and U maps
        self.noise_array = np.concatenate([temp_noise_array, temp_noise_array])

        # Data vector of two masked maps
        self.data_vec = None

    def compute_theory_spectra(self):
        print('Evaluating the CCL theory Cl values')

        # Create our array of redshift values to evaluate dN/dz at
        redshift_range = np.linspace(0.0, 3.0, 500)

        # Create new galaxy distribution at our current redshift
        dNdz = stats.norm.pdf(redshift_range, loc=self.redshift, scale=0.15)

        cosmo = ccl.Cosmology(**self.fiducial_cosmology)

        # Lensing bins at current redshift
        field_E = ccl.WeakLensingTracer(cosmo, dndz=(redshift_range, dNdz))

        # Compute Cl values
        Cl_EE = ccl.angular_cl(cosmo, field_E, field_E, self.ells)

        # Store the theory values into our class at the correct position
        self.cl_EE_theory = Cl_EE
        self.cl_EB_theory = np.zeros_like(Cl_EE)
        self.cl_BB_theory = np.zeros_like(Cl_EE)

        # Add two sets of zeros for ell=0 and ell=1
        Cl_EE = np.concatenate([[0, 0], Cl_EE.copy()])

        # Save the EE theory spectra to a file
        np.savetxt(f'../data/TheorySpectra_N{self.n_side}.dat', np.array([Cl_EE]).T)

    def plot_cls(self):
        plt.figure(figsize=(11, 7))

        plt.loglog(self.ells, self.ells * (self.ells + 1) * self.cl_EE_theory / (2 * np.pi), c='mediumseagreen', lw=3, label='Correct lensing signal')
        plt.loglog(self.ells, self.ells * (self.ells + 1) * self.theory_cl_noise / (2 * np.pi), c='orange', lw=3, label='Noise')

        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell (\ell + 1) C_{\ell} / 2 \pi$')

        plt.legend()
        plt.tight_layout()
        plt.savefig('Output/Cls.pdf')

        plt.close('all')

    def compute_S_tilde_matrix(self):
        print('Computing S_tilde matrix')
        # Go through each ell value to set the correct Cl value
        for ell in range(2, self.l_max + 1):
            idx_min = ell ** 2 - 4
            idx_max = idx_min + (2 * ell + 1)

            # Store Cl values into our S_tilde matrix
            self.S_tilde_EE[idx_min: idx_max] = self.cl_EE_theory[ell - 2]
            self.S_tilde_BB[idx_min: idx_max] = self.cl_BB_theory[ell - 2]

    def compute_covariance_matrix(self):
        # Extract the relevant parts of the Y matrix
        Y_matrix_QE = self.Y_matrix[0: self.n_pix_mask, 0: self.num_m_modes]
        Y_matrix_UE = self.Y_matrix[self.n_pix_mask: 2 * self.n_pix_mask, 0: self.num_m_modes]

        # Y_matrix_QB = self.Y_matrix[0: self.n_pix_mask, self.num_m_modes: 2 * self.num_m_modes]
        # Y_matrix_UB = self.Y_matrix[self.n_pix_mask: 2 * self.n_pix_mask, self.num_m_modes: 2 * self.num_m_modes]

        print('Computing signal matrices')
        # Polarisation-only signal matrices
        S_matrix_QQ = (Y_matrix_QE * self.S_tilde_EE @ np.conj(Y_matrix_QE).T).real
        S_matrix_QU = (Y_matrix_QE * self.S_tilde_EE @ np.conj(Y_matrix_UE).T).real
        S_matrix_UU = (Y_matrix_UE * self.S_tilde_EE @ np.conj(Y_matrix_UE).T).real

        print('Initialising the covariance matrix')
        self.cov = np.zeros([2 * self.n_pix_mask, 2 * self.n_pix_mask], dtype=float)

        print('Setting the elements of the covariance matrix')
        self.cov[0: self.n_pix_mask, 0: self.n_pix_mask] = S_matrix_QQ
        self.cov[0: self.n_pix_mask, self.n_pix_mask: 2 * self.n_pix_mask] = S_matrix_QU
        self.cov[self.n_pix_mask: 2 * self.n_pix_mask, 0: self.n_pix_mask] = S_matrix_QU.T
        self.cov[self.n_pix_mask: 2 * self.n_pix_mask, self.n_pix_mask: 2 * self.n_pix_mask] = S_matrix_UU

        # Add the noise matrix to the covariance matrix
        np.fill_diagonal(self.cov, self.cov.diagonal() + self.noise_array)

        print('Inverting the covariance matrix')
        self.cov_inv = np.linalg.inv(self.cov)

        # Check that the covariance matrix has been inverted successfully
        print(self.cov[0, :] @ self.cov_inv[:, 0] - 1)
        print(self.cov[0, :] @ self.cov_inv[:, 1])

    def compute_cov_inv_Y(self):
        print('Evaluating C^-1 @ Y')
        self.cov_inv_Y = self.cov_inv @ self.Y_matrix

    def compute_Y_cov_inv_Y(self):
        print('Evaluating Y.H @ C^-1 @ Y')
        self.Y_dagger_cov_inv_Y = np.conj(self.Y_matrix).T @ self.cov_inv_Y

    def F_idx(self, ell, offset):
        return (ell - 2) + (offset * self.num_l_modes)

    def compute_Fisher_matrix(self):
        #* Take elements of the Y.H @ C^-1 @ Y matrix
        # Create views for the EE part
        Y_C_Y_EE = self.Y_dagger_cov_inv_Y[0: self.num_m_modes, 0: self.num_m_modes]

        # Also the EB parts
        # Y_C_Y_EB = self.Y_dagger_cov_inv_Y[0: self.num_m_modes, self.num_m_modes: 2 * self.num_m_modes]
        # Y_C_Y_BE = self.Y_dagger_cov_inv_Y[self.num_m_modes: 2 * self.num_m_modes, 0: self.num_m_modes]

        Y_C_Y_BE = self.Y_dagger_cov_inv_Y[0: self.num_m_modes, self.num_m_modes: 2 * self.num_m_modes]
        Y_C_Y_EB = self.Y_dagger_cov_inv_Y[self.num_m_modes: 2 * self.num_m_modes, 0: self.num_m_modes]

        # Also the BB part
        Y_C_Y_BB = self.Y_dagger_cov_inv_Y[self.num_m_modes: 2 * self.num_m_modes,
                                           self.num_m_modes: 2 * self.num_m_modes]

        # Create our complete Fisher matrix
        self.F = np.zeros([self.num_spectra * self.num_l_modes,
                           self.num_spectra * self.num_l_modes], dtype=float)

        print('Evaluating the Fisher matrix')
        # Go through each ell combination
        for idx1, ell1 in enumerate(range(2, self.l_max + 1)):
            # for idx2, ell2 in enumerate(range(2, ell1 + 1)):
            for idx2, ell2 in enumerate(range(2, self.l_max + 1)):

                # Go through each field combination
                for field_ab in self.all_spectra:
                    for field_cd in self.all_spectra:
                        # Extract each individual field from the combination
                        field_1, field_2 = field_ab
                        field_3, field_4 = field_cd

                        # Case of two auto-spectra
                        if (field_1 == field_2) and (field_3 == field_4):
                            tmp_F = eval(f'evaluate_Fisher(ell1, ell2, Y_C_Y_{field_3 + field_1}, Y_C_Y_{field_1 + field_3}).real')

                        # Field1 & Field2 are auto, 3 & 4 are cross-spectra
                        elif (field_1 == field_2) and (field_3 != field_4):
                            tmp_F = 2 * eval(f'evaluate_Fisher(ell1, ell2, Y_C_Y_{field_3 + field_1}, Y_C_Y_{field_1 + field_4}).real')

                        # Field1 & Field2 are cross, 3 & 4 are auto-spectra
                        elif (field_1 != field_2) and (field_3 == field_4):
                            tmp_F = 2 * eval(f'evaluate_Fisher(ell1, ell2, Y_C_Y_{field_1 + field_3}, Y_C_Y_{field_3 + field_2}).real')

                        # All sets of cross-spectra
                        else:
                            tmp_F = 2 * eval(f'evaluate_Fisher(ell1, ell2, Y_C_Y_{field_3 + field_2}, Y_C_Y_{field_1 + field_4}).real') + 2 * eval(f'evaluate_Fisher(ell1, ell2, Y_C_Y_{field_3 + field_1}, Y_C_Y_{field_2 + field_4}).real')

                        # Compute offset in the Fisher matrix for current field combination
                        offset_1 = self.all_spectra.index((field_1, field_2))
                        offset_2 = self.all_spectra.index((field_3, field_4))

                        # Set the correct element of the Fisher matrix using symmetry of the ell values
                        # Where we have previously not set it's value
                        self.F[self.F_idx(ell1, offset_1), self.F_idx(ell2, offset_2)] = tmp_F

                        # if self.F[self.F_idx(ell1, offset_1), self.F_idx(ell2, offset_2)] == 0:
                        #     self.F[self.F_idx(ell1, offset_1), self.F_idx(ell2, offset_2)] = tmp_F
                            # self.F[self.F_idx(ell2, offset_2), self.F_idx(ell1, offset_1)] = tmp_F

        # Invert the Fisher matrix
        self.F_inv = np.linalg.inv(self.F)

        # Once Fisher matrix has been computed, save to disk
        np.save(f'Output/Fishers/F_N{self.n_side}', self.F)
        np.save(f'Output/Fishers/F_inv_N{self.n_side}', self.F_inv)

    def compute_noise_bias(self):
        # Here, we compute the b_ell noise bias terms

        # First, compute N @ (C^-1 @ Y)
        print('Evaluating N @ (C^-1 @ Y)')
        N_cov_inv_Y = self.noise_array[:, None] * self.cov_inv_Y

        # Compute individual b_ell terms
        print('Evaluating the noise bias terms')
        b_ell_EE = evaluate_b_ell_auto(self.ells, N_cov_inv_Y, np.conj(self.cov_inv_Y),
                                       offset=0)
        b_ell_BB = evaluate_b_ell_auto(self.ells, N_cov_inv_Y, np.conj(self.cov_inv_Y),
                                       offset=self.num_m_modes)
        b_ell_EB = evaluate_b_ell_cross(self.ells, N_cov_inv_Y, np.conj(self.cov_inv_Y),
                                        offset1=0, offset2=self.num_m_modes)

        # Combine into a single array of b_ell values
        self.b_ell = np.array([b_ell_EE, b_ell_EB, b_ell_BB]).flatten()

    def compute_cl_from_map(self, input_map_Q, input_map_U):
        self.data_vec = np.zeros([2 * self.n_pix_mask])

        # Set the elements of our data vector to our masked map
        self.data_vec[0: self.n_pix_mask] = input_map_Q[self.mask]
        self.data_vec[self.n_pix_mask: 2 * self.n_pix_mask] = input_map_U[self.mask]

        # Multiply our map with (C^-1 @ Y)
        map_cov_inv_Y = self.data_vec @ self.cov_inv_Y

        # Now cut it down to E- and B-modes separately
        map_E_modes = map_cov_inv_Y[0: self.num_m_modes]
        map_B_modes = map_cov_inv_Y[self.num_m_modes: 2 * self.num_m_modes]

        y_ells = []

        for spectra in self.all_spectra_str:
            field_1 = spectra[0]
            field_2 = spectra[1]

            if field_1 == field_2:
                y_l_tmp = eval(f'evaluate_y_ell(self.ells, map_{field_1}_modes, map_{field_2}_modes)')
            else:
                y_l_tmp = eval(f'evaluate_y_ell_TE(self.ells, map_{field_1}_modes, map_{field_2}_modes)')

            y_ells.append(y_l_tmp)

        y_ells = np.array(y_ells).flatten()

        # Deconvolve the y_l values with the Fisher matrix to form C_l values
        C_ells = self.F_inv @ (y_ells - self.b_ell)

        # Extract specific positions of Cls
        QML_Cl_EE = C_ells[0: self.num_l_modes]
        QML_Cl_BB = C_ells[self.num_l_modes: 2 * self.num_l_modes]
        QML_Cl_EB = C_ells[2 * self.num_l_modes: 3 * self.num_l_modes]

        return [[QML_Cl_EE, QML_Cl_BB, QML_Cl_EB], y_ells]

    def collect_memory(self, full_clean=False):
        """
        Function to clean up memory in object once the Cl's have been computed by resetting the large matrix variables
        """
        self.Y_matrix = None
        self.cov = None
        self.cov_inv = None
        self.Y_dagger_cov_inv_Y = None

        self.S_tilde_EE = None
        self.S_tilde_BB = None

        self.data_vec = None

        if full_clean:
            # Clean cov @ Y only for a full clean-up
            self.cov_inv_Y = None

        # Manually call the garbage collector to ensure memory is freed
        gc.collect()

    def save_arrays(self):
        # Save the arrays
        print('Saving arrays')
        np.save(f'/disk01/maraio/QML_TEB_data/cov_z{self.redshift}_N{self.n_side}', self.cov)
        np.save(f'/disk01/maraio/QML_TEB_data/inv_cov_z{self.redshift}_N{self.n_side}', self.cov_inv)
        np.save(f'/disk01/maraio/QML_TEB_data/inv_cov_Y_z{self.redshift}_N{self.n_side}', self.cov_inv_Y)
        np.save(f'/disk01/maraio/QML_TEB_data/Y_inv_cov_Y_z{self.redshift}_N{self.n_side}',
                self.Y_dagger_cov_inv_Y)

        self.cov = None
        self.cov_inv = None
        self.cov_inv_Y = None
        self.Y_dagger_cov_inv_Y = None

        # Collect
        gc.collect()

        print('IO done!')

    def load_arrays(self):
        # Load the arrays
        print('Loading arrays')
        # self.cov = np.load(f'/disk01/maraio/QML_Data/DES_Noise/cov_z{self.redshift}_N{self.n_side}.npy')
        # self.cov_inv = np.load(f'/disk01/maraio/QML_Data/DES_Noise/inv_cov_z{self.redshift}_N{self.n_side}.npy')
        self.cov_inv_Y = np.load(f'/disk01/maraio/QML_TEB_data/inv_cov_Y_z{self.redshift}_N{self.n_side}.npy',
                                 mmap_mode=None)
        self.Y_dagger_cov_inv_Y = np.load(
                f'/disk01/maraio/QML_TEB_data/Y_inv_cov_Y_z{self.redshift}_N{self.n_side}.npy', mmap_mode='r')

        print('IO done!')
