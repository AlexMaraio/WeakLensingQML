"""
Script to turn the y_ell estimates produced from my C++ code into power spectrum estimates though inverting the
Cl-Fisher matrix
"""
import numpy as np


# The map resolution
n_side = 256

# The number of maps in the ensemble
num_maps = 1250

# Location of maps to read in from
folder_path = "/disk01/maraio/NonGaussianShear/N1024_Gaussian/Maps_N256"

# Using our resolution, create a number of associated quantities
l_max = 2 * n_side
l_max_full = 3 * n_side - 1
ells = np.arange(2, l_max + 1)
ells_full = np.arange(2, l_max_full + 1)
n_ell = len(ells)
n_ell_full = len(ells_full)

# The filepath for our Fisher matrix
fisher_filepath = "../data/numerical_fishers/ConjGrad_Fisher_N256_nummaps10_noise3_EB_nostars.dat"

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
    # Read in the y_ell values
    y_ells = np.loadtxt(f'{folder_path}/Map{map_num}_yell_QML.dat')

    # Remove EB data from y_ells
    y_ells = np.delete(y_ells, np.s_[n_ell_full: 2 * n_ell_full], axis=0)

    # Compute Cl values
    c_ells = fisher_matrix_inv @ y_ells

    # Save the Cl values (EE and BB separately)
    np.savetxt(f'{folder_path}/Map{map_num}_Cl_EE_QML.dat', c_ells[0: n_ell_full])
    np.savetxt(f'{folder_path}/Map{map_num}_Cl_BB_QML.dat', c_ells[n_ell_full: 2 * n_ell_full])
