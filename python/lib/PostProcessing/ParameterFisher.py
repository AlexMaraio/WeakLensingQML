import numpy as np
from scipy import stats, interpolate as interp
import pyccl as ccl
import sys
import os

import matplotlib as mpl
import seaborn as sns

from .Enums import ClType, CovType

cosmicfish_pylib_path = '/home/maraio/Codes/CosmicFish/python/'
sys.path.insert(0, os.path.normpath(cosmicfish_pylib_path))

import cosmicfish_pylib as cf
import cosmicfish_pylib.utilities as fu
import cosmicfish_pylib.colors as fc
import cosmicfish_pylib.fisher_matrix as fm
import cosmicfish_pylib.fisher_derived as fd
import cosmicfish_pylib.fisher_operations as fo
import cosmicfish_pylib.fisher_plot_settings as fps
import cosmicfish_pylib.fisher_plot_analysis as fpa
import cosmicfish_pylib.fisher_plot as fp


class ParamFisher:

    def __init__(self, n_side, fiducial_cosmology, redshift, cl_type, cov_type, cov_class, label, params, params_latex, d_params, n_samples):

        # Extract map resolution & associated quantities from QML class
        self.n_side = n_side
        # self.l_max = self.QML_class.l_max
        self.l_max = 2 * self.n_side
        self.n_ell = self.l_max - 1
        self.ells = np.arange(2, self.l_max + 1)

        # Copy the fiducial cosmology from the QML class
        self.fiducial_cosmology = fiducial_cosmology

        # Set up a redshift range for all z values
        self.redshift = redshift
        self.redshift_range = np.linspace(0.0, 3.0, 200)

        # Our galaxy distribution
        self.dNdz = stats.norm.pdf(self.redshift_range, loc=self.redshift, scale=0.15)

        # The density to galaxy bias values
        self.bias = np.ones_like(self.redshift_range) * np.sqrt(1 + self.redshift)

        # The type of the input covariance matrix - PCl or QML
        self.cl_type = cl_type
        
        # Analytic or numeric covariance matrix type
        self.cov_type = cov_type
        
        # Store the main covariance matrix
        self.cov_class = cov_class
        
        # See if this is a PCl or QML type
        if self.cl_type == ClType.PCl:
            # If PCl, then test for numeric or analytic
            if self.cov_type == CovType.numeric:
                # Invert the numeric Cl covariance matrix
                self.inv_cov = np.linalg.inv(self.cov_class.num_cov_EE_EE)[0:self.l_max-1, 0:self.l_max-1]

                # Include numerical covariance factor
                self.inv_cov *= (n_samples - 2 - len(self.ells)) / (n_samples - 1)
            
            else:
                # Then invert the analytic Cl covariance matrix
                self.inv_cov = np.linalg.inv(self.cov_class.ana_cov_EE_EE)[0:self.l_max-1, 0:self.l_max-1]
        
        else:
            # No need to invert the QML Fisher matrix
            self.inv_cov = self.cov_class.fisher_sym[0: self.l_max - 1, 0: self.l_max - 1]

        # Store the label for this Fisher matrix, used for plotting
        self.label = label

        # List of parameter names
        self.params = list(params.keys())

        # List of LaTeX labels for parameters
        self.params_latex = params_latex

        # Dictionary of parameter names & central values
        self.cent_params = params

        # Dictionary of parameter values with their step when computing their derivatives
        self.d_params = d_params

        # The parameter Fisher matrix
        self.param_F = None

        # The cosmic_fish Fisher matrix
        self.cf_fisher_matrix = None

    def compute_param_Fisher(self):
        d_param_steps = 11

        cl_derivs = {key: [] for key in self.params}

        print('Computing the derivates of Cl with respect to parameter: ')
        for param in self.params:
            print(param, end='\t', flush=True)
            param_vals = np.zeros([d_param_steps])

            cl_vals = np.zeros([d_param_steps, len(self.ells)])

            # Copy the fiducial cosmology
            current_cosmology = self.fiducial_cosmology.copy()

            # Go through our small parameter range that's zeroed around the fiducial value
            for d_param_idx, d_param in enumerate(range(-d_param_steps // 2 + 1, d_param_steps // 2 + 1)):

                # Update our current cosmology using our small parameter interval
                current_cosmology[param] = self.cent_params[param] + d_param * self.d_params[param]

                # Create new CCL cosmology class
                cosmo = ccl.Cosmology(**current_cosmology)

                # Create field objects for our T & E fields
                field_E = ccl.WeakLensingTracer(cosmo, dndz=(self.redshift_range, self.dNdz))

                # Recompute the Cl values for each bin combination
                cl_E_E = ccl.angular_cl(cosmo, field_E, field_E, self.ells)

                # Store the current parameter value in our list
                param_vals[d_param_idx] = current_cosmology[param]

                # Store the Cl values
                cl_vals[d_param_idx] = cl_E_E.copy()

            # Go through each ell value
            for ell_idx, ell in enumerate(self.ells):
                cl_deriv = float(interp.InterpolatedUnivariateSpline(param_vals, cl_vals[:, ell_idx], k=4).derivative()(self.cent_params[param]))
                cl_derivs[param].append(cl_deriv)

        self.param_F = np.zeros([len(self.params), len(self.params)])

        print('\nEvaluating the parameter Fisher matrix')
        for param_idx_1, param_1 in enumerate(self.params):
            for param_idx_2, param_2 in enumerate(self.params):
                cl_deriv_vec_1 = cl_derivs[param_1]
                cl_deriv_vec_2 = cl_derivs[param_2]

                self.param_F[param_idx_1, param_idx_2] = cl_deriv_vec_1 @ self.inv_cov @ cl_deriv_vec_2

        # Now turn our numerical Fisher matrix into a Cosmic-Fish Fisher matrix object
        self.cf_fisher_matrix = fm.fisher_matrix(fisher_matrix=self.param_F, param_names=self.params,
                                                 param_names_latex=self.params_latex,
                                                 fiducial=list(self.cent_params.values()))

        self.cf_fisher_matrix.name = self.label

        # Compute the Figure-of-merit for this Fisher matrix
        fom = np.sqrt(np.linalg.det(self.param_F))

        print(f'Figure-of-merit with Nside of {self.n_side} is {fom:.4e} for {self.label}')


def plot_param_Fisher(fishers, n_samples, output_folder):
    sns.reset_orig()
    mpl.rc_file_defaults()

    # Extract the Cosmic-Fish Fisher matrix class's from input
    fishers_plot = fpa.CosmicFish_FisherAnalysis()
    fishers_plot.add_fisher_matrix([fisher.cf_fisher_matrix for fisher in fishers])

    # Create plotter class from Fishers provided above
    fisher_plotter = fp.CosmicFishPlotter(fishers=fishers_plot)

    fisher_plotter.new_plot()

    # Plot our Fisher matrices as a triangle plot
    fisher_plotter.plot_tri(title=f'Comparison of Fisher constraints for N_{{\\textsc{{side}}}} of {fishers[0].n_side} and {n_samples} samples and inv-cov correction')

    fisher_plotter.export(f'Plots_New/{output_folder}/ParamFisher_N{fishers[0].n_side}_OmS8H0_n{n_samples}.pdf')
