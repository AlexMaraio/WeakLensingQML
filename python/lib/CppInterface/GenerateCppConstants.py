"""
File to generate the C++ constants.h file that's necessary for our C++ code
"""


def generate_constants_h(n_side, mask, noise_var, num_maps, output_file_path):
    constants_text = fr"""
//
// Created by amaraio on 22/09/2021.
//

#ifndef CONJGRAD_EIGEN_CONSTANTS_H
#define CONJGRAD_EIGEN_CONSTANTS_H

// The working precision of all numeric types in conjugate-gradient
using precision = double;

// The N_side parameter of our maps
constexpr auto n_side = {int(n_side)};

// The number of pixels in our unmasked maps
constexpr auto n_pix = 12 * n_side * n_side;

// The number of pixels in our masked maps
constexpr auto n_pix_mask = {int(mask.sum())};

// The maximum ell mode to consider when performing alm expansions
constexpr auto l_max = 3 * n_side - 1;

// Number of ell modes, from ell=2 up to and including ell=l_max
constexpr auto num_l_modes = l_max - 1;

// The number of iterations that is used in map2alm_iter
constexpr int n_hp_iter = 3;

// The tolerance used in the conjugate-gradient algorithm
constexpr auto conj_grad_tol = 1e-5;

// The number of maps to average over
constexpr auto num_maps = {int(num_maps)};

// The factor to multiply the active ell mode when estimating the Fisher matrix
constexpr auto cl_mult_fact = 1e7;

// The noise variance
constexpr auto noise_var = {noise_var};

// Terminal colours
constexpr auto TERM_RESET = "\033[0m";
constexpr auto TERM_RED   = "\033[0;31m";
constexpr auto TERM_BLUE  = "\033[0;34m";

#endif //CONJGRAD_EIGEN_CONSTANTS_H
    """

    # Open, write, and then close the constants file
    output_file = open(output_file_path, 'w')

    output_file.write(constants_text)

    output_file.close()
