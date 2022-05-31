"""
Script to turn the y_ell estimates produced from my C++ code into power spectrum estimates though inverting the
Cl-Fisher matrix
"""
import numpy as np


# The map resolution
n_side = 256

# The number of maps in the ensemble
num_maps = 2_500

# Location of maps to read in from
folder_path = "/cephfs/maraio/NonGaussianMaps/N1024_Gaussian_shift_0_01214_whnoise/Maps_N256"
# folder_path = "/cephfs/maraio/NonGaussianMaps/N1024_LogNormal_shift_0_01214_whnoise/Maps_N256"

print(f'Post-processing QML data located at {folder_path} for {num_maps} maps at N_side of {n_side}')

# Using our resolution, create a number of associated quantities
l_max = 2 * n_side
l_max_full = 3 * n_side - 1
ells = np.arange(2, l_max + 1)
ells_full = np.arange(2, l_max_full + 1)
n_ell = len(ells)
n_ell_full = len(ells_full)

# The filepath for our Fisher matrix
fisher_filepath = "../../data/numerical_fishers/ConjGrad_Fisher_N256_nummaps25_noise3_EB_whstars.dat"

# Read in the Fisher matrix
fisher_matrix = np.loadtxt(fisher_filepath)

# Delete entries that correspond to the EB modes
fisher_matrix = np.delete(fisher_matrix, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
fisher_matrix = np.delete(fisher_matrix, np.s_[n_ell_full: 2 * n_ell_full], axis=1)

# Make the Fisher matrix symmetric
fisher_matrix = (fisher_matrix.copy() + fisher_matrix.copy().T) / 2

# Invert the Fisher matrix
fisher_matrix_inv = np.linalg.inv(fisher_matrix)

# Go through each map and normalise the y_ell values using the inverse Fisher matrix
for map_num in range(num_maps):
    if np.mod(map_num, 50) == 0:
        print(map_num, end=' ', flush=True)

    # Read in the y_ell values
    y_ells = np.loadtxt(f'{folder_path}/Map{map_num}_yell_QML.dat')

    # Remove EB data from y_ells
    y_ells = np.delete(y_ells, np.s_[n_ell_full: 2 * n_ell_full], axis=0)

    # Compute Cl values
    c_ells = fisher_matrix_inv @ y_ells

    # Save the Cl values (EE and BB separately)
    np.savetxt(f'{folder_path}/Map{map_num}_Cl_EE_QML.dat', c_ells[0: n_ell_full])
    np.savetxt(f'{folder_path}/Map{map_num}_Cl_BB_QML.dat', c_ells[n_ell_full: 2 * n_ell_full])

# Tell user we're done with the deconvolution step
print('... Done!')

#* Now want to compute and then save the numerical covariance matrix recovered from the above spectra

# The number of ell modes for this n_side
num_ell_modes = (3 * n_side - 1) - 1

cl_samples_EE = np.zeros([num_maps, num_ell_modes])
cl_samples_BB = np.zeros([num_maps, num_ell_modes])

print('Reading in Cl values for covariance matrices')
for map_num in range(num_maps):
    cl_EE = np.loadtxt(f'{folder_path}/Map{map_num}_Cl_EE_QML.dat')
    cl_BB = np.loadtxt(f'{folder_path}/Map{map_num}_Cl_BB_QML.dat')

    cl_samples_EE[map_num] = cl_EE
    cl_samples_BB[map_num] = cl_BB

# Compute covariance matrices
print('Computing covariance matrices')
cl_EE_cov = np.cov(cl_samples_EE, rowvar=False)
cl_BB_cov = np.cov(cl_samples_BB, rowvar=False)

# Save covariance matrices
print('Saving covariance matrices')
np.savetxt(f'{folder_path}/../Numerical_covariance_EE_QML_N{n_side}.dat', cl_EE_cov)
np.savetxt(f'{folder_path}/../Numerical_covariance_BB_QML_N{n_side}.dat', cl_BB_cov)
