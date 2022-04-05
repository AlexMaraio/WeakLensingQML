import sys
import os
import numpy as np
from scipy import stats, interpolate as interp
import pyccl as ccl
import pymaster as nmt

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

    def __init__(self, n_side, fiducial_cosmology, redshift, cl_type, cov_type, cov_class, label, params, params_latex,
                 d_params, n_samples, ells_per_bin, manual_l_max=None, debug=False):
        # Extract map resolution & associated quantities
        self.n_side = n_side
        self.l_max = 2 * self.n_side
        self.n_ell = self.l_max - 1
        self.ells = np.arange(2, self.l_max + 1)

        # If we're artificially limiting the maximum ell-multipole that we're going to, then store it here
        if manual_l_max is not None:
            self.l_max = manual_l_max
            self.n_ell = self.l_max - 1
            self.ells = np.arange(2, self.l_max + 1)
            self.manual_l_max = manual_l_max
        else:
            self.manual_l_max = self.l_max

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

        # Simply a list of fiducial parameter values
        self.fiducial_values = list(params.values())

        # Dictionary of parameter values with their step when computing their derivatives
        self.d_params = d_params

        # The parameter Fisher matrix
        self.param_F = None

        # The cosmic_fish Fisher matrix
        self.cf_fisher_matrix = None

        # The figure of merit for this Fisher matrix
        self.fig_of_merit = None

        # The number of ell modes per bin, if we're going to bin our data
        self.ells_per_bin = ells_per_bin

        # Set up associated bin object
        self.bins = nmt.NmtBin.from_nside_linear(self.n_side, self.ells_per_bin)

        # Should we print debugging information?
        self.debug = debug

    def compute_param_Fisher(self):
        """
        Function to compute the parameter Fisher matrix given a Cl covariance matrix
        """
        # The number of steps in d_param that we should take when evaluating the derivative of the Cl values with
        # respect to the parameter
        d_param_steps = 25

        cl_derivs = {key: [] for key in self.params}

        if self.debug: print('Computing the derivatives of Cl with respect to parameter: ')
        for param in self.params:
            if self.debug: print(param, end='\t', flush=True)
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

        # If we're using more than one ell mode per bin, then we need to bin our vector of cl_derivs
        if self.ells_per_bin != 1:
            if self.debug: print('Binning the cl_deriv vector')
            num_bins = self.bins.get_n_bands()

            cl_derivs_binned = {key: np.zeros(num_bins) for key in self.params}

            # Bin the derivative for each parameter
            for param_idx, param in enumerate(self.params):
                # Then go through each bin and combine our values using the linear binning scheme
                for bin_idx in range(num_bins):
                    ells_bin = self.bins.get_ell_list(bin_idx)

                    # The NaMaster bins continue above ell of 2 * N_side, so cut off here
                    if ells_bin[0] > self.l_max:
                        break

                    # Get list of bin weights for current bin
                    bin_weights = self.bins.get_weight_list(bin_idx)[ells_bin <= 2 * self.n_side]

                    # Get list of ell values for current bub
                    bin_ells = ells_bin[ells_bin <= 2 * self.n_side]

                    # Convert ell values to bin indices as we start from ell=2
                    bin_ells -= 2

                    cl_derivs_binned[param][bin_idx] = bin_weights @ np.array(cl_derivs[param])[bin_ells]

            # Overwrite our existing vector of derivatives with our binned version
            cl_derivs = cl_derivs_binned

        self.param_F = np.zeros([len(self.params), len(self.params)])

        if self.debug: print('\nEvaluating the parameter Fisher matrix')
        for param_idx_1, param_1 in enumerate(self.params):
            for param_idx_2, param_2 in enumerate(self.params):
                cl_deriv_vec_1 = cl_derivs[param_1]
                cl_deriv_vec_2 = cl_derivs[param_2]

                self.param_F[param_idx_1, param_idx_2] = cl_deriv_vec_1 @ self.inv_cov @ cl_deriv_vec_2

        # Now turn our numerical Fisher matrix into a Cosmic-Fish Fisher matrix object
        cf_fisher_matrix = fm.fisher_matrix(fisher_matrix=self.param_F, param_names=self.params,
                                            param_names_latex=self.params_latex,
                                            fiducial=self.fiducial_values)

        self.cf_fisher_matrix = cf_fisher_matrix

        self.cf_fisher_matrix.name = self.label

        # Compute and store the Figure-of-merit for this Fisher matrix
        self.fig_of_merit = np.sqrt(np.linalg.det(self.param_F))

    def transform_param_Fisher(self, jacobian, derived_param_names, derived_param_latex, derived_fiducial_values):
        """
        Function to take our existing Fisher matrix and transform it using the provided Jacobian matrix
        """
        if self.debug: print('Transforming the Fisher matrix now')

        # Set up our class that holds the transformation
        fisher_transformation = fd.fisher_derived(derived_matrix=jacobian,
                                                  param_names=self.params,
                                                  derived_param_names=derived_param_names,
                                                  derived_param_names_latex=derived_param_latex,
                                                  fiducial=self.fiducial_values,
                                                  fiducial_derived=derived_fiducial_values)

        # Add our existing base Fisher matrix to our transformation
        derived_fisher = fisher_transformation.add_derived(self.cf_fisher_matrix)

        # Set this class's Fisher matrix to our transformed version
        self.cf_fisher_matrix = derived_fisher

        # Update Fisher matrix's name
        self.cf_fisher_matrix.name = self.label

        # Compute and store the new figure-of-merit for this Fisher matrix
        self.fig_of_merit = np.sqrt(np.linalg.det(self.param_F))

        # Now overwrite the existing set of parameters to our new derived ones
        self.params = derived_param_names
        self.fiducial_values = derived_fiducial_values


def plot_param_Fisher_triangle(fishers, output_folder, plot_filename, plot_title):
    """
    Function to plot a set of Fisher matrices for multiple parameters as a triangle plot
    """
    sns.reset_orig()
    mpl.rc_file_defaults()

    # Extract the Cosmic-Fish Fisher matrix class's from input
    fishers_plot = fpa.CosmicFish_FisherAnalysis()
    fishers_plot.add_fisher_matrix([fisher.cf_fisher_matrix for fisher in fishers])

    # Create plotter class from Fishers provided above
    fisher_plotter = fp.CosmicFishPlotter(fishers=fishers_plot)

    fisher_plotter.new_plot()

    # Plot our Fisher matrices as a triangle plot
    # fisher_plotter.plot_tri(title=f'Comparison of Fisher constraints for different estimators', settings={'figure_width': 4, 'D1_main_fontsize': 12, 'D1_secondary_fontsize': 15, 'D2_main_fontsize': 15, 'D2_secondary_fontsize': 15, 'legend_fontsize': 15, 'title_fontsize': 15})
    fisher_plotter.plot_tri(title=plot_title)

    # Save the triangle plot
    fisher_plotter.export(f'{output_folder}/ParamFisher_N{fishers[0].n_side}_{plot_filename}_paper.pdf')


def plot_param_Fisher_1D(fishers, output_folder, plot_filename, plot_title):
    """
    Function to plot a set of Fisher matrices for just a single parameters as a one-dimensional density plot
    """
    # Reset any current visual plot settings
    sns.reset_orig()
    mpl.rc_file_defaults()

    # Extract the Cosmic-Fish Fisher matrix class's from input
    fishers_plot = fpa.CosmicFish_FisherAnalysis()
    fishers_plot.add_fisher_matrix([fisher.cf_fisher_matrix for fisher in fishers])

    # Create plotter class from Fishers provided above
    fisher_plotter = fp.CosmicFishPlotter(fishers=fishers_plot)

    # Create new plot
    fisher_plotter.new_plot()

    # Plot the 1D distribution on the figure
    fisher_plotter.plot1D(title=plot_title, legend_loc='center right', subplot_x_size=6)

    # fisher_plotter.set_legend(legend_loc='center right')

    # Save the plot to the output folder given the filename & N_side value
    fisher_plotter.export(f'{output_folder}/ParamFisher_N{fishers[0].n_side}_{plot_filename}_paper.pdf')
