"""
File that allows Python code to easily integrate with our C++ library
"""

import ctypes


class CppLib:
    def __init__(self, lib_path, mask_path, cl_data_path):
        """
        Class that interfaces the exposed functions in our C++ library to Python for easy calling.
        Takes in the path to the shared library location, a filepath for the mask, and a filepath for the
        fiducial power spectrum coefficients.
        """

        # Try importing the C++ library, raise custom exception if it cannot be found
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError:
            print(f'Cannot find the shared library at the specified path of {lib_path}, please check that it has '
                  f'first been compiled and the path is correct')

        self.mask_path = mask_path
        self.cl_data_path = cl_data_path

    def compute_fisher_matrix(self, fisher_matrix_out_path):
        """
        Function to compute the Fisher matrix using our C++ code and store it at the specified output path
        """
        self.lib.compute_fisher_matrix(self.mask_path.encode(), self.cl_data_path.encode(),
                                       fisher_matrix_out_path.encode())

    def compute_power_spectrum(self, map_gamma1_path, map_gamma2_path, y_ell_output_path):
        """
        Function to compute the set of y_ell values using our C++ code for a gamma_1 and gamma_2 maps that are given
        as inputs and store them at the location given by the output_path
        """
        self.lib.compute_power_spectrum(self.mask_path.encode(), self.cl_data_path.encode(),
                                        map_gamma1_path.encode(), map_gamma2_path.encode(), y_ell_output_path.encode())

