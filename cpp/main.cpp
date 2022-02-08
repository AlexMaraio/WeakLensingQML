#include <iostream>

// HealPix includes
#include <healpix_map.h>
#include <healpix_map_fitsio.h>

#include "detail/constants.h"
#include "detail/ComputeCl.h"


int main()
{
  std::cout << "Starting main now...\n";

  //! Noise bias constant values
  constexpr auto intrinsic_gal_ellip = 0.21;
  constexpr auto avg_gal_den = 30;
  constexpr auto area_per_pix = 1.49E8 / n_pix;
  constexpr auto num_gal_per_pix = avg_gal_den * area_per_pix;

  // The variance of the noise in each pixel
  const auto noise_var = static_cast<precision>((intrinsic_gal_ellip * intrinsic_gal_ellip) / num_gal_per_pix);

  // Now want to read in our mask
  const auto filepath = "/home/maraio/Codes/QMLForWeakLensingHome/Data/Masks/New/SkyMask_N256_whstars.fits";

  // Initialise a Healpix map that'll contain our mask
  Healpix_Map<precision> mask;
  mask.SetNside(n_side, RING);
  mask.fill(0);

  // Read in mask and set mask array
  read_Healpix_map_from_fits(filepath, mask);

  // Filepath that points to the fiducial Cl spectrum
  const auto cl_datapath = "/home/maraio/Codes/QMLForWeakLensingHome/Data/TheorySpectra_N256.dat";

  //* Create EB class and compute Fisher matrix for it
  auto ClClass_EB = ComputeCl_EB(cl_datapath, mask, noise_var);

  // Read in the provided power spectrum
  ClClass_EB.read_in_power_spec_EB();

  // Compute a numerical estimate of the Fisher matrix
  ClClass_EB.estimate_Fisher_matrix_EB();
}
