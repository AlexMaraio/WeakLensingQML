import numpy as np
import healpy as hp
import quaternionic
import spherical
import pymp
import gc


# noinspection PyPep8Naming
class YMatrix:
    """
    Class that computes and stores a Y matrix to be used in other QMLClass's
    """
    def __init__(self, n_side, l_max, mask):
        print('Creating Y matrix class')
        self.n_side = n_side
        self.n_pix = 12 * (n_side ** 2)

        self.l_max = l_max
        self.num_m_modes = (l_max + 1) ** 2 - 4

        self.mask = mask
        self.n_pix_mask = mask.sum()

        # The numerical Y matrices
        self.Y_matrix = None
        self.Y_matrix_dagger = None

    def compute_Y_matrix(self):
        print('Initialising the Y arrays')
        Y_matrix_QE = pymp.shared.array((self.n_pix, self.num_m_modes), dtype='complex')
        Y_matrix_QB = pymp.shared.array((self.n_pix, self.num_m_modes), dtype='complex')
        Y_matrix_UE = pymp.shared.array((self.n_pix, self.num_m_modes), dtype='complex')
        Y_matrix_UB = pymp.shared.array((self.n_pix, self.num_m_modes), dtype='complex')

        print('Evaluating the Y matrices')
        with pymp.Parallel(8) as p:
            wigner = spherical.Wigner(self.l_max)
            workspace = wigner.new_workspace()
            for pix_idx in p.xrange(self.n_pix):
                if self.mask[pix_idx] == 0:
                    continue

                theta, phi = hp.pix2ang(self.n_side, pix_idx)

                # Set up an array at our given (theta, phi) location
                R = quaternionic.array.from_spherical_coordinates(theta, phi)

                sYlm_p = wigner.sYlm(2, R, workspace=workspace)  # Positive spin spherical harmonic
                sYlm_n = wigner.sYlm(-2, R, workspace=workspace)  # Negative spin sp. harm.

                # Go through each (ell, m) combination
                for ell in range(2, self.l_max + 1):
                    for m_idx, m in enumerate(range(-ell, ell + 1)):
                        l_m_idx = ell ** 2 + m_idx - 4

                        # Evaluate the positive and negative spin harmonics at (l, m)
                        positive_s_val = sYlm_p[wigner.Yindex(ell, m)] / 2
                        negative_s_val = sYlm_n[wigner.Yindex(ell, m)] / 2

                        Y_matrix_QE[pix_idx, l_m_idx] = -(positive_s_val + negative_s_val)
                        Y_matrix_QB[pix_idx, l_m_idx] = -1J * (positive_s_val - negative_s_val)

                        Y_matrix_UE[pix_idx, l_m_idx] = 1J * (positive_s_val - negative_s_val)
                        Y_matrix_UB[pix_idx, l_m_idx] = -(positive_s_val + negative_s_val)

        # Cut down the matrices to ignore masked out pixels
        print('Cutting down the Y matrices')
        Y_matrix_QE = Y_matrix_QE[self.mask, :]
        Y_matrix_QB = Y_matrix_QB[self.mask, :]
        Y_matrix_UE = Y_matrix_UE[self.mask, :]
        Y_matrix_UB = Y_matrix_UB[self.mask, :]

        # Combine the four Y-matrices into a single one
        self.Y_matrix = np.zeros([2 * self.n_pix_mask, 2 * self.num_m_modes], dtype=complex)

        # Set the elements to their respective arrays
        print('Combining the Y matrices')
        self.Y_matrix[0: self.n_pix_mask, 0: self.num_m_modes] = Y_matrix_QE
        self.Y_matrix[0: self.n_pix_mask, self.num_m_modes: 2 * self.num_m_modes] = Y_matrix_QB
        self.Y_matrix[self.n_pix_mask: 2 * self.n_pix_mask, 0: self.num_m_modes] = Y_matrix_UE
        self.Y_matrix[self.n_pix_mask: 2 * self.n_pix_mask, self.num_m_modes: 2 * self.num_m_modes] = Y_matrix_UB

    def save_array(self):
        # Saves the Y matrix array to the disk
        print('Saving the Y matrix')

        np.save(f'/disk01/maraio/Y_matrices/Y_matrix_{self.n_side}', self.Y_matrix)
        np.save(f'/disk01/maraio/Y_matrices/Y_matrix_dagger_{self.n_side}', self.Y_matrix_dagger)

        self.Y_matrix = None
        gc.collect()

        print('Done Y_matrix IO')

    def load_array(self):
        # Imports the saved Y matrix
        print('Reading in the Y_matrix')

        # Read in the Y_matrices, using memory-mapping to save RAM
        self.Y_matrix = np.load(f'/disk01/maraio/Y_matrices/Y_matrix_{self.n_side}.npy', mmap_mode='r')
        self.Y_matrix_dagger = np.load(f'/disk01/maraio/Y_matrices/Y_matrix_dagger_{self.n_side}.npy', mmap_mode='r')

        print('Done Y_matrix IO')
