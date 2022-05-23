//
// Created by amaraio on 23/09/2021.
//

#ifndef CONJGRAD_EIGEN_COMPUTECL_EB_H
#define CONJGRAD_EIGEN_COMPUTECL_EB_H

#include <string>
#include <fstream>
#include <tuple>
#include <complex>
#include <iomanip>
#include <limits>
#include <utility>
#include <stdexcept>

#include <powspec.h>
#include <planck_rng.h>
#include <arr.h>
#include <healpix_map_fitsio.h>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"

#include "constants.h"
#include "MatrixReplacement.h"


class ComputeCl_EB
{
public:
    ComputeCl_EB(std::string cl_datapath, const Healpix_Map<precision>& mask, const precision noise_var) :
            cl_datapath(std::move(cl_datapath)),
            mask(mask),
            noise_var(noise_var)
    {
      // Set the precision of the conjugate-gradient algorithm
      cg.setTolerance(conj_grad_tol);

      // Set all elements in our temperature map to zero
      map_T.fill(0);

      // Resize the Cl Fisher matrix to correct size
      F_mat.resize(3 * num_l_modes, 3 * num_l_modes);

      y_ell.resize(3 * num_l_modes);

      // Also initialise our map-vectors with the correct length
      b.resize(2 * n_pix_mask);
      x.resize(2 * n_pix_mask);
      x0.resize(2 * n_pix_mask);
    };

    void read_in_power_spec_EB();

    void generate_map_EB(arr<double>& cl_TT_arr_in, arr<double>& cl_EE_arr_in, arr<double>& cl_BB_arr_in,
                         arr<double>& cl_TE_arr_in, arr<double>& cl_TB_arr_in, arr<double>& cl_EB_arr_in);

    int compute_y_ell_EB(Eigen::Vector<precision, 3 * num_l_modes>& y_ells_in);

    void estimate_Fisher_matrix_EB();

  void save_y_ell_from_map_EB(const std::string& file_location, const int map_num);

private:
    // The filepath for data containing the Cl values
    const std::string cl_datapath;

    // HealPix map of our mask
    const Healpix_Map<precision> mask;

    // The value of the noise variance in each pixel
    const precision noise_var;

    // HealPix array that contains the fiducial cl values
    arr<double> cl_TT_arr = arr<double>(l_max + 1, 0);
    arr<double> cl_EE_arr = arr<double>(l_max + 1, 0);
    arr<double> cl_BB_arr = arr<double>(l_max + 1, 0);
    arr<double> cl_TE_arr = arr<double>(l_max + 1, 0);
    arr<double> cl_TB_arr = arr<double>(l_max + 1, 0);
    arr<double> cl_EB_arr = arr<double>(l_max + 1, 0);

    // HealPix power spectrum class for the Cl's
    PowSpec cls = PowSpec(6, l_max);

    // The ell-weighted sum of Cl values, for preconditioner
    precision cl_sum = 0;

    // HealPix random number generator function
    planck_rng rng = planck_rng(std::time(nullptr));

    // HealPix weights to use, currently all set to unity
    const arr<double> weights = arr<double>(n_pix, 1.0);

    // HealPix Alm class, used for generating random realisation of map
    Alm<xcomplex<precision>> alms_T = Alm<xcomplex<precision>>(l_max, l_max);
    Alm<xcomplex<precision>> alms_E = Alm<xcomplex<precision>>(l_max, l_max);
    Alm<xcomplex<precision>> alms_B = Alm<xcomplex<precision>>(l_max, l_max);

    // HealPix map
    Healpix_Map<precision> map_T = Healpix_Map<precision>(n_side, RING, SET_NSIDE);
    Healpix_Map<precision> map_Q = Healpix_Map<precision>(n_side, RING, SET_NSIDE);
    Healpix_Map<precision> map_U = Healpix_Map<precision>(n_side, RING, SET_NSIDE);

    // The right-hand side (our map)
    Eigen::VectorXd b;

    // Our starting solution
    Eigen::VectorXd x0;

    // Create our solution vector x, which will contain C^-1 @ data_vec
    Eigen::VectorXd x;

    // The estimated y_ell values for the map
    Alm<xcomplex<precision>> Y_C_inv_x_T = Alm<xcomplex<precision>>(l_max, l_max);
    Alm<xcomplex<precision>> Y_C_inv_x_E = Alm<xcomplex<precision>>(l_max, l_max);
    Alm<xcomplex<precision>> Y_C_inv_x_B = Alm<xcomplex<precision>>(l_max, l_max);

    // Array of current y_ell values
    Eigen::Vector<std::complex<precision>, Eigen::Dynamic> y_ell;

    // Create our MatrixReplacement class for the cov matrix
    // (using pointer for cl_arr as it does not have a value yet)
    MatrixReplacement_EB CovMat = MatrixReplacement_EB(noise_var, &cl_EE_arr, mask);

    // Eigen Conj-Grad class
    Eigen::ConjugateGradient<MatrixReplacement_EB, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;

    // Numerical estimate of the Fisher matrix
    Eigen::MatrixXd F_mat;
};


void ComputeCl_EB::read_in_power_spec_EB()
{
  /*
   * Function to read in and store the power spectrum as given in the cl_datapath member
   */

  // Read in the Cl values
  std::ifstream Cl_file(this->cl_datapath);

  // Go through each line of the file, where the current line is our ell value
  for(auto [line_idx, cl] = std::tuple{0, std::string()}; getline(Cl_file, cl); ++line_idx)
  {
    // Convert the current string to a double
    const auto cl_val = std::stod(cl);

    // Store the current Cl value into the arrays
    cl_TT_arr[line_idx] = cl_val;
    cl_EE_arr[line_idx] = cl_val;
    cl_TE_arr[line_idx] = cl_val;
  }

  // Close file once done
  Cl_file.close();

  // Once cl_arr & cl_sum have been computed, init the conjugate-gradient method
  cg.compute(CovMat);
}


void ComputeCl_EB::generate_map_EB(arr<double>& cl_TT_arr_in, arr<double>& cl_EE_arr_in, arr<double>& cl_BB_arr_in,
                                   arr<double>& cl_TE_arr_in, arr<double>& cl_TB_arr_in, arr<double>& cl_EB_arr_in)
{
  // Set the Cl values in our cls class to the input
  // Note that HealPix deletes the arrays once done...
  cls.Set(cl_TT_arr_in, cl_EE_arr_in, cl_BB_arr_in, cl_TE_arr_in, cl_TB_arr_in, cl_EB_arr_in);

  // Using our Cl values, create a random set of alm values
  create_alm_pol(cls, alms_T, alms_E, alms_B, rng);

  // Generate new map from alms
  alm2map_spin(alms_E, alms_B, map_Q, map_U, 2);

  // Now set the Eigen data vector to our HealPix map - respecting the mask
  for(auto [pix_idx, mask_pix_idx] = std::tuple{0, 0}; pix_idx < n_pix; ++pix_idx)
  {
    if(mask[pix_idx])
    {
      b(mask_pix_idx) = static_cast<precision>(map_Q[pix_idx]);
      x0(mask_pix_idx) = static_cast<precision>(map_Q[pix_idx] / (cl_sum + noise_var));

      b(mask_pix_idx + n_pix_mask) = static_cast<precision>(map_U[pix_idx]);
      x0(mask_pix_idx + n_pix_mask) = static_cast<precision>(map_U[pix_idx] / (cl_sum + noise_var));

      ++mask_pix_idx;
    }
  }
}


int ComputeCl_EB::compute_y_ell_EB(Eigen::Vector<precision, 3 * num_l_modes>& y_ells_in)
{
  /*
   * Function to compute the y_ell values for a map that is stored in the class's b vector member
   * Adds the computed y_ell value to the y_ells_in parameter
   */

  // Take the start time
//  const auto start_time = std::chrono::high_resolution_clock::now();

  // Reset the maximum number of iterations, this is quite low so it doesn't get stuck here indefinitely
  cg.setMaxIterations(5000);

  // Solve the C * y = map equation for C^-1 @ map
//  x = cg.solve(b);

  // Alternatively, solve with an initial solution
  x = cg.solveWithGuess(b, x0);

  // Check that CG was successful
  if(cg.info() != Eigen::Success)
  {
    // If not, then print debugging info and raise an error to exit program
    std::cout << TERM_RED <<
              "\nERROR: Number of iterations is: " << cg.iterations() << ", with cg.info() as: " << cg.info() <<
              TERM_RESET << std::endl;

    // Return one for failure
    return 1;
  }

  if(!x.allFinite())
  {
    std::cout << TERM_RED <<
              "\nERROR: Number of iterations is: " << cg.iterations() << ", with cg.info() as: " << cg.info() <<
              TERM_RESET << std::endl;
    std::cout << TERM_BLUE << x << std::endl;

    // Return one for failure
    return 1;
  }

  // Calculate duration of CG process
//  const auto stop_time = std::chrono::high_resolution_clock::now();
//  const std::chrono::duration<double, std::ratio<1, 1>> duration = stop_time - start_time;
//  std::cout << "Each pass of CG took " << duration.count() << " seconds for num_iter: " << cg.iterations() << "\n";

  // Now we need to transform our Eigen solution to a HealPix map
  for(auto [pix_idx, mask_pix_idx] = std::tuple{0, 0}; pix_idx < n_pix; ++pix_idx)
  {
    if(mask[pix_idx])
    {
      map_Q[pix_idx] = x(mask_pix_idx);
      map_U[pix_idx] = x(mask_pix_idx + n_pix_mask);

      ++mask_pix_idx;
    }
    else
    {
      map_Q[pix_idx] = 0.0;
      map_U[pix_idx] = 0.0;
    }
  }

  // Convert our masked map back to a set of alm values
//  map2alm_spin(map_Q, map_U, Y_C_inv_x_E, Y_C_inv_x_B, 2, weights, false);
//  map2alm_spin_iter2(map_Q, map_U, Y_C_inv_x_E, Y_C_inv_x_B, 2, 1e-1, 1e-1);
  map2alm_pol_iter(map_T, map_Q, map_U, Y_C_inv_x_T, Y_C_inv_x_E, Y_C_inv_x_B, n_hp_iter, weights);

  // Remove pixel area normalisation from alm values
  Y_C_inv_x_E.Scale(n_pix / (4 * M_PI));
  Y_C_inv_x_B.Scale(n_pix / (4 * M_PI));

  //* Turn the Y_C_inv_x values into a set of y_ell values
  // First, reset all current y_ell values
  y_ell.setZero();

  for(int ell = 2; ell <= l_max; ++ell)
  {
    // TODO: Check if these values are correct!!

    y_ell(ell - 2)                    += Y_C_inv_x_E(ell, 0) * std::conj(Y_C_inv_x_E(ell, 0));
    y_ell(ell - 2 + num_l_modes)      += Y_C_inv_x_E(ell, 0) * Y_C_inv_x_B(ell, 0);
    y_ell(ell - 2 + 2 * num_l_modes)  += Y_C_inv_x_B(ell, 0) * std::conj(Y_C_inv_x_B(ell, 0));

    for(int m = 1; m <= ell; ++m)
    {
      y_ell(ell - 2) += 2.0 * Y_C_inv_x_E(ell, m) * std::conj(Y_C_inv_x_E(ell, m));

//      y_ell(ell - 2 + num_l_modes) += Y_C_inv_x_E(ell, m) * std::conj(Y_C_inv_x_B(ell, m));
//      y_ell(ell - 2 + num_l_modes) += Y_C_inv_x_E(ell, m) * std::conj(Y_C_inv_x_B(ell, m)) + Y_C_inv_x_B(ell, m) * std::conj(Y_C_inv_x_E(ell, m));

      y_ell(ell - 2 + num_l_modes) += Y_C_inv_x_E(ell, m) * std::conj(Y_C_inv_x_B(ell, m)) + Y_C_inv_x_B(ell, m) * std::conj(Y_C_inv_x_E(ell, m));
      y_ell(ell - 2 + num_l_modes) += pow(-1.0, m) * std::conj(Y_C_inv_x_E(ell, m) * std::conj(Y_C_inv_x_B(ell, m)) + Y_C_inv_x_B(ell, m) * std::conj(Y_C_inv_x_E(ell, m)));

      y_ell(ell - 2 +  2 * num_l_modes) += 2.0 * Y_C_inv_x_B(ell, m) * std::conj(Y_C_inv_x_B(ell, m));
    }
  }

  // Now add our current y_ell values to the input array
  y_ells_in += y_ell.real() / 2;

  // Return zero for success
  return 0;
}


void ComputeCl_EB::estimate_Fisher_matrix_EB()
{
  // First, set all entries in the Fisher matrix to zero
  F_mat.fill(0);

  // Need to compute the fiducial y_ell values for the true power spectrum
  Eigen::Vector<precision, 3 * num_l_modes> y_ells_fid;
  y_ells_fid.fill(0);

  std::cout << "Computing fiducial y_ells now\n";
  // Need a more precise fiducial spectra, so use an order-of-magnitude more maps to compute average from
  for(int i = 0; i < num_maps; ++i)
  {
    // Take copy of fiducial spectrum, as HealPix zeros spectra when setting Cl values
    auto cl_TT_arr_fid = cl_TT_arr;
    auto cl_EE_arr_fid = cl_EE_arr;
    auto cl_BB_arr_fid = cl_BB_arr;
    auto cl_TE_arr_fid = cl_TE_arr;
    auto cl_TB_arr_fid = cl_TB_arr;
    auto cl_EB_arr_fid = cl_EB_arr;

    // Generate a random realisation of the map
    this->generate_map_EB(cl_TT_arr_fid, cl_EE_arr_fid, cl_BB_arr_fid, cl_TE_arr_fid, cl_TB_arr_fid, cl_EB_arr_fid);

    // Use this map to compute a set of y_ell values
    const auto ret_val = this->compute_y_ell_EB(y_ells_fid);

    if(ret_val) --i;
  }
  y_ells_fid /= num_maps;

  // Now, go through each ell mode and inject power at that ell mode

  std::cout << "Going through ell modes now\n";
  for(const auto EB_idx : {0, 2})
  {
    std::cout << "EB_idx: " << EB_idx << "\n";

    std::cout << "ell: ";
    for(int ell = 2; ell <= l_max; ++ell)
    {
      // Print active ell mode
      std::cout << ell << " " << std::flush;

      // Create an array for our y_ell values for our injected power modes
      Eigen::Vector<precision, 3 * num_l_modes> y_ells_var;
      y_ells_var.fill(0);

      // The amount of power we're going to inject into our active ell mode
      const auto delta_cl = cl_mult_fact * cl_EE_arr[ell];

      // Generate an ensemble of maps to average over
      for(int i = 0; i < num_maps; ++i)
      {
        // Take a copy of our fiducial spectrum and inject power at our active ell mode
        auto cl_TT_arr_fid = cl_TT_arr;
        auto cl_EE_arr_tmp = cl_EE_arr;
        auto cl_BB_arr_tmp = cl_BB_arr;
        auto cl_TE_arr_fid = cl_TE_arr;
        auto cl_TB_arr_fid = cl_TB_arr;
        auto cl_EB_arr_tmp = cl_EB_arr;

        if(EB_idx == 0)
        {
          // Just add power to the EE modes
          cl_EE_arr_tmp[ell] += delta_cl;
        }
        else if(EB_idx == 1)
        {
          // Add power to the EB modes
          cl_EB_arr_tmp[ell] += delta_cl;
        }
        else
        {
          // Add power to the BB modes
          cl_BB_arr_tmp[ell] += delta_cl;
        }

        // Generate a map using this injected power spectrum
        this->generate_map_EB(cl_TT_arr_fid, cl_EE_arr_tmp, cl_BB_arr_tmp, cl_TE_arr_fid, cl_TB_arr_fid, cl_EB_arr_tmp);

        // Compute the y_ell values for our new map
        this->compute_y_ell_EB(y_ells_var);
      }

      // Average the y_ells by dividing through by the number of maps in the average
      y_ells_var /= num_maps;

      // Set the elements of the Fisher matrix accordingly
//      this->F_mat((ell - 2) + (EB_idx * num_l_modes), Eigen::all) += (y_ells_var - y_ells_fid) / delta_cl;
      this->F_mat(Eigen::all, (ell - 2) + (EB_idx * num_l_modes)) += (y_ells_var - y_ells_fid) / delta_cl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // Since we set the rows and columns of the Fisher matrix, divide all values by two
//  this->F_mat /= 2;

  // Explicitly symmetrise our Fisher matrix
//  const auto F_mat_sym = (this->F_mat + this->F_mat.transpose().eval()) / 2.0;

  // Save the Fisher matrix to disk
  std::ofstream Fisher("/home/maraio/Codes/WeakLensingQML/data/numerical_fishers/ConjGrad_Fisher_N" + std::to_string(n_side) + "_nummaps" + std::to_string(num_maps) + "_noise30_EB_whstars.dat",
                       std::ios::out | std::ios::trunc);

  Fisher << std::setprecision(std::numeric_limits<precision>::digits10 + 1) << this->F_mat;

  Fisher.close();
}

void ComputeCl_EB::save_y_ell_from_map_EB(const std::string& file_location, const int map_num)
{
  // This function should read in a set of shear maps that is pointed to through the map_filename argument
  // then recover their y_ell values and save this output to the filename provided.

  // Read in the set of shear maps
//  read_Healpix_map_from_fits(file_location + "/Map" + std::to_string(map_num) + "_Q.fits", this->map_Q);
//  read_Healpix_map_from_fits(file_location + "/Map" + std::to_string(map_num) + "_U.fits", this->map_U);

  read_Healpix_map_from_fits(file_location + "/Map" + std::to_string(map_num) + "-f1z1.fits", this->map_Q, 2);
  read_Healpix_map_from_fits(file_location + "/Map" + std::to_string(map_num) + "-f1z1.fits", this->map_U, 3);

  // Now set the Eigen data vector to our HealPix map - respecting the mask
  for (auto [pix_idx, mask_pix_idx] = std::tuple{0, 0}; pix_idx < n_pix; ++pix_idx)
  {
    if (mask[pix_idx])
    {
      b(mask_pix_idx) = static_cast<precision>(map_Q[pix_idx]);
      x0(mask_pix_idx) = static_cast<precision>(map_Q[pix_idx] / (cl_sum + noise_var));

      b(mask_pix_idx + n_pix_mask) = static_cast<precision>(map_U[pix_idx]);
      x0(mask_pix_idx + n_pix_mask) = static_cast<precision>(map_U[pix_idx] / (cl_sum + noise_var));

      ++mask_pix_idx;
    }
  }

  // Create our y_ell vector where the output will be stored into
  Eigen::Vector<precision, 3 * num_l_modes> y_ells;
  y_ells.fill(0);

  // Now compute the y_ell values for the current set of maps
  const auto ret_val = this->compute_y_ell_EB(y_ells);

  // If the return value from the conjugate-gradient step is not zero, then something's gone wrong!
  if (ret_val)
  {
    std::cerr << "Something's gone wrong! Ret val: " << ret_val << "\n";
    return;
  }

  // Save the set of y_ell values to the file path provided
  std::ofstream y_ells_output(file_location + "/Map" + std::to_string(map_num) + "_yell_QML.dat",
                              std::ios::out | std::ios::trunc);
  y_ells_output << std::setprecision(std::numeric_limits<precision>::digits10 + 1) << y_ells;
  y_ells_output.close();
}


#endif //CONJGRAD_EIGEN_COMPUTECL_EB_H
