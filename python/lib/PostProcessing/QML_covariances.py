import numpy as np
import pymaster as nmt


class QML_covariances:
    # Class to store information about a certain QML run

    def __init__(self, n_side, file_location, ells_per_bin):
        # Set run properties
        self.n_side = n_side

        self.ells = np.arange(2, 3 * self.n_side)

        # Number of ell modes for this class
        self.n_ell = (3 * self.n_side - 1) - 1

        # Store the number of ells per bin, if we're going to bin
        self.ells_per_bin = ells_per_bin

        # Load in our full Cl Fisher matrix
        self.fisher_matrix = np.loadtxt(file_location)

        # Form a symmetric version of our Fisher matrix
        fisher_matrix_sym = (self.fisher_matrix + self.fisher_matrix.T) / 2

        # Form a cut-down version of our Fisher matrix, where we ignore any EB correlations
        self.fisher_sym = np.zeros([2 * self.n_ell, 2 * self.n_ell])

        # EE-EE
        self.fisher_sym[0 * self.n_ell: 1 * self.n_ell, 0 * self.n_ell: 1 * self.n_ell] = fisher_matrix_sym[0 * self.n_ell: 1 * self.n_ell, 0 * self.n_ell: 1 * self.n_ell]
        
        # EE-BB
        self.fisher_sym[0 * self.n_ell: 1 * self.n_ell, 1 * self.n_ell: 2 * self.n_ell] = fisher_matrix_sym[0 * self.n_ell: 1 * self.n_ell, 2 * self.n_ell: 3 * self.n_ell]

        # BB-EE
        self.fisher_sym[1 * self.n_ell: 2 * self.n_ell, 0 * self.n_ell: 1 * self.n_ell] = fisher_matrix_sym[2 * self.n_ell: 3 * self.n_ell, 0 * self.n_ell: 1 * self.n_ell]
        
        # BB-BB
        self.fisher_sym[1 * self.n_ell: 2 * self.n_ell, 1 * self.n_ell: 2 * self.n_ell] = fisher_matrix_sym[2 * self.n_ell: 3 * self.n_ell, 2 * self.n_ell: 3 * self.n_ell]

        # Now invert our symmetric cut-down Fisher matrix to get the Cl covariance matrix
        self.cl_inv_cov = np.linalg.inv(self.fisher_sym)

        # Now extract the individual elements:
        self.num_cov_EE_EE = self.cl_inv_cov[0 * self.n_ell: 1 * self.n_ell, 0 * self.n_ell: 1 * self.n_ell]
        self.num_cov_BB_BB = self.cl_inv_cov[1 * self.n_ell: 2 * self.n_ell, 1 * self.n_ell: 2 * self.n_ell]

    def bin_covariance_matrix(self):
        #* Function to bin our numerical covariance matrix
        bins = nmt.NmtBin.from_nside_linear(self.n_side, self.ells_per_bin)

        # Get the number of bins
        num_bins = bins.get_n_bands()

        # Initialise empty covariance matrices for our binned version
        binned_cov_EE_EE = np.zeros([num_bins, num_bins])
        binned_cov_BB_BB = np.zeros([num_bins, num_bins])

        # Go through each pair of bins
        for bin_i in range(num_bins):
            for bin_j in range(num_bins):
                # Get the ell values at this bin (start at ell=2)
                ells_i = bins.get_ell_list(bin_i) - 2
                ells_j = bins.get_ell_list(bin_j) - 2

                # Bin our covariance matrix                
                binned_cov_EE_EE[bin_i, bin_j] = bins.get_weight_list(bin_i) @ self.num_cov_EE_EE[ells_i][:, ells_j] @ bins.get_weight_list(bin_j)
                binned_cov_BB_BB[bin_i, bin_j] = bins.get_weight_list(bin_i) @ self.num_cov_BB_BB[ells_i][:, ells_j] @ bins.get_weight_list(bin_j)

        # Set the covariance matrix to the binned version
        self.num_cov_EE_EE = binned_cov_EE_EE
        self.num_cov_BB_BB = binned_cov_BB_BB

        # Now set the ell-range to be the binned version
        self.ells = bins.get_effective_ells()
