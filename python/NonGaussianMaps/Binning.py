"""
File that contains the process of binning a power spectrum or covariance matrix from discrete ells into bins
"""

from enum import Enum
import numpy as np


class BinType(Enum):
    linear = 1
    logarithmic = 2


def uniform_weights(ell):
    # Uniform weights binning, here we're binning into four ells per bin
    return (1 / 4) * np.ones_like(ell)


def mode_count_weights(ell):
    # Weighting scheme corresponding the number of m-modes per ell-mode
    return 2 * ell + 1


def inverse_variance_weights(ell):
    # Weighting scheme corresponding to variance per ell-mode
    return ell * (ell + 1)


class Bins:

    def __init__(self, n_side, binning_type, delta_ell, bin_weights):
        # The map resolution for our specified binning scheme
        self.n_side = n_side

        # Usually it's fine to start binning from ell = 2
        self.l_min = 2

        # Use the map resolution to determine the maximum multipole under consideration
        self.l_max = 3 * n_side - 1

        # Construct our range of ell values
        self.ells = np.arange(self.l_min, self.l_max + 1)

        # The binning type used - either linear or logarithmic
        self.bin_type = binning_type

        # The delta ell value for our binning scheme
        self.delta_ell = delta_ell

        # A function that returns the bin weight at a specific ell mode
        self.bin_weights = bin_weights

        # The number of bins constructed for this binning scheme
        self.num_bins = 0

        # Array containing the lower and upper edges of each bin
        self.bin_edges = []

        # The effective ell mode for each bin
        self.bin_centres = []

        # Call the construct bins function here
        self.construct_bins()

    def construct_bins(self):
        lower_ell = self.l_min

        while True:
            if self.bin_type == BinType.linear:
                # If we're using linear binning, then just add our delta_ell to the lower ell value to obtain the
                # bin's upper edge
                upper_ell = lower_ell + self.delta_ell - 1

            else:
                # Otherwise, we're using logarithmic binning so use the logarithmic formula to obtain the upper bin edge
                upper_ell = np.floor(10 ** (np.log10(lower_ell) + self.delta_ell))

            # If the upper edge is greater than the maximum ell mode under consideration then exit as we're done
            if upper_ell > self.l_max:
                break

            # Store the bin edges in the class
            self.bin_edges.append([int(lower_ell), int(upper_ell)])

            # Compute the effective ell for this bin
            bin_ells = np.arange(lower_ell, upper_ell + 1)
            self.bin_centres.append(np.sum(self.bin_weights(bin_ells) * bin_ells) / np.sum(self.bin_weights(bin_ells)))

            # Move onto the next bin
            lower_ell = upper_ell + 1

        # Convert our bin edges and bin centres to NumPy arrays
        self.bin_edges = np.array(self.bin_edges)
        self.bin_centres = np.array(self.bin_centres)

        # Compute the number of bins
        self.num_bins = len(self.bin_edges)

        print(f'Created bins object with {self.num_bins} bins using {self.bin_weights.__name__} weighting scheme for a '
              f'delta-ell of {self.delta_ell} for maps with an N_side of {self.n_side}')

    def bin_vector(self, input_vector):
        # If we're using uniform weights with delta_ell of one, then just return original vector
        if (self.bin_type == BinType.linear) and (self.delta_ell == 1):
            return input_vector

        # Create a vector to store our binned vector into
        binned_vector = np.zeros(self.num_bins)

        # Go through each bin
        for bin_idx, (bin_lower, bin_upper) in enumerate(self.bin_edges):
            # Get the ell values for this bin
            bin_ells = np.arange(bin_lower, bin_upper + 1)

            # Bin the vector
            binned_vector[bin_idx] = (np.sum(self.bin_weights(bin_ells) * input_vector[bin_ells - 2]) /
                                      np.sum(self.bin_weights(bin_ells)))

        # Return binned vector
        return binned_vector.copy()
