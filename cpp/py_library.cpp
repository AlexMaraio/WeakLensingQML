#include <iostream>
#include <string>

#include <healpix_map.h>
#include <healpix_map_fitsio.h>

#include "detail/constants.h"
#include "detail/ComputeCl.h"

extern "C"
{

  int compute_fisher_matrix(const char* mask_filepath, const char* cl_datapath, const char* fisher_matrix_out_path)
  {
    std::cout << "Starting main now...\n";

    // Convert the const char* C-type arrays into C++ strings
    const auto mask_filepath_str = std::string(mask_filepath);
    const auto cl_datapath_str = std::string(cl_datapath);
    const auto fisher_matrix_out_path_str = std::string(fisher_matrix_out_path);

    // Initialise a HealPix map that'll contain our mask
    Healpix_Map<precision> mask;
    mask.SetNside(n_side, RING);
    mask.fill(0);

    // Read in mask and set mask array
    read_Healpix_map_from_fits(mask_filepath_str, mask);

    // Create EB class and compute Fisher matrix for it
    auto ClClass_EB = ComputeCl_EB(cl_datapath_str, mask, noise_var);

    // Read in the provided power spectrum
    ClClass_EB.read_in_power_spec_EB();

    // Compute a numerical estimate of the Fisher matrix
    ClClass_EB.estimate_Fisher_matrix_EB(fisher_matrix_out_path_str);

    return 0;
  }

  int compute_power_spectrum(const char* mask_filepath, const char* cl_datapath,
                             const char* map_gamma1_path, const char* map_gamma2_path, const char* y_ell_output_path)
  {
    // Convert the const char* C-type arrays into C++ strings
    const auto mask_filepath_str = std::string(mask_filepath);
    const auto cl_datapath_str = std::string(cl_datapath);
    const auto map_gamma1_path_str = std::string(map_gamma1_path);
    const auto map_gamma2_path_str = std::string(map_gamma2_path);
    const auto y_ell_output_path_str = std::string(y_ell_output_path);

    // Initialise a HealPix map that'll contain our mask
    Healpix_Map<precision> mask;
    mask.SetNside(n_side, RING);
    mask.fill(0);

    // Read in mask and set mask array
    read_Healpix_map_from_fits(mask_filepath_str, mask);

    // Create EB class and compute Fisher matrix for it
    auto ClClass_EB = ComputeCl_EB(cl_datapath_str, mask, noise_var);

    // Read in the provided power spectrum
    ClClass_EB.read_in_power_spec_EB();

    ClClass_EB.estimate_y_ell_from_map_EB(map_gamma1_path_str, map_gamma2_path_str, y_ell_output_path_str);

    return 0;
  }

}