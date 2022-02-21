import numpy as np
import healpy as hp
import pymaster as nmt


class PCl_covariances:

    def __init__(self, n_side, Cl_EE, Cl_EB, Cl_BB, ells_per_bin, purify_E, purify_B, mask, num_samples, noise_std,
                 cl_noise, label):
        # Set the n_side attribute for the generates maps
        self.n_side = n_side
        self.n_pix = 12 * self.n_side * self.n_side
        self.l_max = 3 * self.n_side - 1

        # Set the fiducial shear power spectrum
        self.Cl_EE = Cl_EE
        self.Cl_EB = Cl_EB
        self.Cl_BB = Cl_BB

        # Store the number of ells per bin and then create the bins object
        self.ells_per_bin = ells_per_bin
        self.bins = nmt.NmtBin.from_nside_linear(self.n_side, self.ells_per_bin)
        self.ells = self.bins.get_effective_ells()
        self.num_binned_ells = len(self.bins.get_effective_ells())

        # Purification options
        self.purify_E = purify_E
        self.purify_B = purify_B

        # The mask to be used
        self.mask = mask

        # Noise attributes
        self.noise_std = noise_std
        self.cl_noise = cl_noise

        # Number of samples to use in the numerical covariance matrix
        self.num_samples = num_samples

        # Objects where the average Cl values will be stored into
        self.num_avg_EE = None
        self.num_avg_BB = None

        # Objects where our covariance matrices will be stored into
        self.num_cov_EE_EE = None
        self.num_cov_BB_BB = None
        
        self.ana_cov_EE_EE = None
        self.ana_cov_BB_BB = None

        # Store the user-defined label for this PCl class
        self.label = label

    def compute_numerical_covariance(self):
        #* Function to estimate the numerical covariance matrix for this class

        # List to store Cl values into
        cl_EE_samples = []
        cl_BB_samples = []

        # Create a field that only contains our mask
        field_tmp = nmt.NmtField(self.mask, None, spin=2, purify_e=self.purify_E, purify_b=self.purify_B)

        # Now compute the mode-coupling matrix once
        workspace = nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(field_tmp, field_tmp, self.bins)

        for i in range(self.num_samples + 1):
            if np.mod(i, 50) == 0: print(i, end=' ', flush=True)

            # Generate our new random realisation
            alm_E, alm_B = hp.synalm([self.Cl_EE, self.Cl_BB, self.Cl_EB], lmax=self.l_max, new=True)
            map_Q, map_U = hp.alm2map_spin([alm_E, alm_B], self.n_side, 2, self.l_max)

            if self.noise_std != 0:
                noise_Q = np.random.normal(loc=0, scale=self.noise_std, size=self.n_pix)
                noise_U = np.random.normal(loc=0, scale=self.noise_std, size=self.n_pix)

                map_Q += noise_Q
                map_U += noise_U

            fields = nmt.NmtField(self.mask, [map_Q, map_U], purify_e=self.purify_E, purify_b=self.purify_B)

            # Now recover the power spectrum of these fields
            cl_coupled = nmt.compute_coupled_cell(fields, fields)
            cl_EE, cl_EB, cl_BE, cl_BB = workspace.decouple_cell(cl_coupled)

            # Store recovered Cl values
            cl_EE_samples.append(cl_EE)
            cl_BB_samples.append(cl_BB)

        # Compute and store the average Cl value
        self.num_avg_EE = np.mean(cl_EE_samples, axis=0)
        self.num_avg_BB = np.mean(cl_BB_samples, axis=0)

        # Now compute the numerical covariance matrix for our list of samples
        self.num_cov_EE_EE = np.cov(cl_EE_samples, rowvar=False)
        self.num_cov_BB_BB = np.cov(cl_BB_samples, rowvar=False)

        # Save the numerical EE-EE and BB-BB covariance matrices
        np.save(f'../data/Fishers/num_cov_EE_EE_{self.label}', self.num_cov_EE_EE)
        np.save(f'../data/Fishers/num_cov_BB_BB_{self.label}', self.num_cov_BB_BB)

        # Print new line once done
        print('')

    def load_numerical_covariance(self):
        self.num_cov_EE_EE = np.load(f'../data/Fishers/num_cov_EE_EE_{self.label}.npy')
        self.num_cov_BB_BB = np.load(f'../data/Fishers/num_cov_BB_BB_{self.label}.npy')

    def compute_analytic_covariance(self):
        #* Function to estimate the analytical covariance matrix for this class

        # Create a field that only contains our mask
        field_tmp = nmt.NmtField(self.mask, None, spin=2, purify_e=self.purify_E, purify_b=self.purify_B)

        # Create a regular workspace object
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(field_tmp, field_tmp, self.bins)

        # Create a covariance workspace object and then compute the coupling coefficients
        covar_wsp = nmt.NmtCovarianceWorkspace()
        covar_wsp.compute_coupling_coefficients(field_tmp, field_tmp, field_tmp, field_tmp)

        covar_22_22 = nmt.gaussian_covariance(covar_wsp, 2, 2, 2, 2,  # Spins of the 4 fields
                                            [self.Cl_EE + self.cl_noise, self.Cl_EB, self.Cl_EB, self.Cl_BB + self.cl_noise],
                                            [self.Cl_EE + self.cl_noise, self.Cl_EB, self.Cl_EB, self.Cl_BB + self.cl_noise],
                                            [self.Cl_EE + self.cl_noise, self.Cl_EB, self.Cl_EB, self.Cl_BB + self.cl_noise],
                                            [self.Cl_EE + self.cl_noise, self.Cl_EB, self.Cl_EB, self.Cl_BB + self.cl_noise],
                                            wsp, wb=wsp).reshape([self.num_binned_ells, 4, self.num_binned_ells, 4])

        self.ana_cov_EE_EE = covar_22_22[:, 0, :, 0]
        self.ana_cov_BB_BB = covar_22_22[:, 3, :, 3]
