"""
File to generate the cosmic shear power spectra given an input cosmology, redshift distribution, and map resolution
"""

import numpy as np
import pyccl as ccl


class PowerSpec:
    def __init__(self, n_side, cosmology, redshift_range, redshift_dist, file_path):
        """
        Class to compute, save, and store a power spectrum corresponding to a set of maps for a given map resolution
        n_side, CCL Cosmology class, source redshift range and distribution, and output filepath.
        """

        # The resolution of the maps considered
        self.n_side = n_side

        # Values to compute the power spectrum over
        self.ells = np.arange(2, 3 * self.n_side)

        self.cosmology = cosmology

        self.redshift_range = redshift_range
        self.redshift_dist = redshift_dist

        self.cl_EE = None

        self.file_path = file_path

    def compute_power_spec(self):
        """
        Function to compute and save the power spectrum using CCL
        """
        print('Evaluating the theory Cl values using CCL')

        # Create a CCl cosmology class
        cosmo = ccl.Cosmology(**self.cosmology)

        # Lensing bins at current redshift
        field_E = ccl.WeakLensingTracer(cosmo, dndz=(self.redshift_range, self.redshift_dist))

        # Compute Cl values
        Cl_EE = ccl.angular_cl(cosmo, field_E, field_E, self.ells)

        # Add two sets of zeros for ell=0 and ell=1
        self.cl_EE = np.concatenate([[0, 0], Cl_EE.copy()])

        # Save the EE theory spectra to a file
        np.savetxt(self.file_path, self.cl_EE.T)

        print(f'Saved the power spectrum to {self.file_path}')
