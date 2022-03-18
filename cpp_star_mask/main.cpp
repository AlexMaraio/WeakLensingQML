/*

Program to generate a random star mask, originally based on https://github.com/ucl-cosmoparticles/flask/blob/master/src/GenStarMask.cpp

Edited to generate the stellar radii from a uniform distribution instead of log-normal

Input parameters are: RANDOM_SEED N_SIDE R_MIN R_MAX F_SKY FILENAME

 RAND_SEED (int): Random seed to use for the GSL random number generator
 N_SIDE (int): The N_side parameter for generated star mask map
 R_MIN (float): The minimum angular size of the stars to generate (in arc min)
 R_MAX (float): The maximum angular size of the stars to generate (in arc min)
 F_SKY (float): The fraction of sky area (0.0 < F_SKY < 1.0) to cover with our randomly generated stars
 FILENAME (string): The output file name for our generated star mask

*/

// Standard includes
#include <iostream>
#include <cmath>

// HealPix includes
#include <healpix_map.h>
#include <healpix_map_fitsio.h>
#include <rangeset.h>

// GSL includes
#include <gsl/gsl_randist.h>


// Main function
int main(int argc, char* argv[])
{
  // Check number of input parameters:
  if (argc <= 6)
  {
    std::cout << "USAGE: ./star_mask_generator <SEED> <N_SIDE> <R_MIN> <R_MAX> <F_SKY> <FILENAME>\n";
    return 0;
  }

  //! Read input parameters from command line:

  // The random seed to use
  const int rand_seed = std::stoi(argv[1]);

  // The N_side resolution of our maps
  const int n_side = std::stoi(argv[2]);

  // r_min and r_max are given in arc-min.
  const double r_min = std::stod(argv[3]);
  const double r_max = std::stod(argv[4]);

  // The target f_sky to be covered by the stars
  const double f_sky_goal = std::stod(argv[5]);

  // The output map file name
  const std::string file_name = argv[6];

  // Our star mask that we will generate values within
  Healpix_Map<double> star_mask;

  // GSL random number generator
  gsl_rng* rng;

  // Constants:
  // If we randomly generate a star on our mask then set it to zero - else it should be one as no object is present
  constexpr double star_present_val = 0;
  constexpr double empty_map_val = 1;

  printf("Will generate n_side=%d map with %g frac. of stars with radius %g<r<%g arc-min using random seed %d:\n",
         n_side, f_sky_goal, r_min, r_max, rand_seed);


  // Setting up random number generator:
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, rand_seed);

  // Initializing constants for star generation
  const double r_min_rad = r_min * (M_PI / 180.0 / 60.0); // Convert arc-min to radians.
  const double r_max_rad = r_max * (M_PI / 180.0 / 60.0); // Convert arc-min to radians.
  const double r_diff = r_max_rad - r_min_rad;

  // Initializing mask:
  star_mask.SetNside(n_side, RING);
  star_mask.fill(empty_map_val);

  // The total number of pixels in our map
  const int n_pixels = 12 * n_side * n_side;

  // The number of pixels masked out due to our random stars
  int num_masked_pixels = 0;

  // The number of pixels that we want to mask out to hit our f_sky goal
  const int target_masked_pixels = std::ceil(n_pixels * f_sky_goal);

  // Loop until we've masked out our desired number of pixels
  while (num_masked_pixels < target_masked_pixels)
  {
    // Randomly select a center for a star:
    const int central_pixel = static_cast<int>(gsl_rng_uniform(rng) * n_pixels);

    // Convert our pixel index to an angle on our mask
    const pointing central_angle = star_mask.pix2ang(central_pixel);

    // Generate a new random star radius from our distribution - using a uniform distribution
    const double star_radius = r_min_rad + gsl_rng_uniform(rng) * r_diff;

    // Get the set of pixels that correspond to our new random star with its position and radius
    rangeset<int> stellar_pixels;
    star_mask.query_disc(central_angle, star_radius, stellar_pixels);

    // Convert our range of pixels to a std::vector to iterate over
    std::vector<int> stellar_pixels_vec;
    stellar_pixels.toVector(stellar_pixels_vec);

    // Set selected pixels to zero and count zeroed pixels:
    for (auto stellar_pixel: stellar_pixels_vec)
      // Check that we haven't already masked over the current pixel
      if (star_mask[stellar_pixel] != star_present_val)
      {
        star_mask[stellar_pixel] = star_present_val;
        num_masked_pixels++;
      }
  }

  // Output map to FITS file:
  write_Healpix_map_to_fits("!" + file_name, star_mask, planckType<double>()); // Filename prefixed by ! to overwrite.
  std::cout << "Map written to " << file_name.c_str() << "\n";

  // Return success if we've made it this far!
  return 0;
}