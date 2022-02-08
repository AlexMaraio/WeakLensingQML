//
// Created by amaraio on 22/09/2021.
//

#ifndef CONJGRAD_EIGEN_MATRIXREPLACEMENT_EB_H
#define CONJGRAD_EIGEN_MATRIXREPLACEMENT_EB_H


#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"

#include <healpix_map.h>
#include <alm.h>
#include <alm_powspec_tools.h>
#include <alm_healpix_tools.h>


#include "constants.h"


class MatrixReplacement_EB;


namespace Eigen::internal
{
  // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
  template<>
  struct traits<MatrixReplacement_EB> : public Eigen::internal::traits<Eigen::SparseMatrix<precision> >
  {
  };
}


class MatrixReplacement_EB : public Eigen::EigenBase<MatrixReplacement_EB>
{
public:
    //! Constructor
    MatrixReplacement_EB(const precision noise_var,
                      const arr<double>* cl_arr,
                      const Healpix_Map<precision>& mask):
            noise_var(noise_var),
            cl_arr(cl_arr),
            mask(mask)
    {
      map_T.fill(0);
    };

    // Required typedefs, constants, and method:
    typedef precision Scalar;
    typedef precision RealScalar;
    typedef int StorageIndex;
    enum
    {
        ColsAtCompileTime = 2 * n_pix_mask,
        MaxColsAtCompileTime = 2 * n_pix_mask,
        IsRowMajor = false
    };

    static Index rows() {return 2 * n_pix_mask;}
    static Index cols() {return 2 * n_pix_mask;}
    static int outerSize() {return 2 * n_pix_mask;}

    template<typename Rhs>
    Eigen::Matrix<precision, Eigen::Dynamic, 1> operator*(const Eigen::MatrixBase<Rhs>& x) const
    {
      // Convert input Eigen vector to HealPix map
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

      // Compute spherical harmonic transform of our map - using number of iterations
      map2alm_pol_iter(map_T, map_Q, map_U, alms_T, alms_E, alms_B, n_hp_iter, weights);

      // Remove pixel area normalisation from alm values
      alms_E.Scale(n_pix / (4 * M_PI));
//      alms_B.Scale(n_pix / (4 * M_PI));

      // Also now scale each alm value with its Cl value
      alms_E.ScaleL(*cl_arr);
      alms_B.Scale(0);

      // Convert alms back to map
      alm2map_spin(alms_E, alms_B, map_Q, map_U, 2);

      // Convert our HealPix map to our Eigen (masked) vector
      for(auto [pix_idx, mask_pix_idx] = std::tuple{0, 0}; pix_idx < n_pix; ++pix_idx)
      {
        if(mask[pix_idx])
        {
          Ax(mask_pix_idx) = map_Q[pix_idx];
          Ax(mask_pix_idx + n_pix_mask) = map_U[pix_idx];

          ++mask_pix_idx;
        }
      }

      // Compute contribution from noise matrix
      const auto N_x = this->noise_var * x;

      // Return the addition of signal & noise parts
      return Ax + N_x;
    }

private:
    // The noise variance value
    const precision noise_var;

    // HealPix weights to use, currently all set to unity
    const arr<double> weights = arr<double>(n_pix, 1.0);

    // HealPix array of the power spectrum Cl coefficients
    const arr<double>* cl_arr;

    // The temperature map
    mutable Healpix_Map<precision> map_T = Healpix_Map<precision>(n_side, RING, SET_NSIDE);
    mutable Healpix_Map<precision> map_Q = Healpix_Map<precision>(n_side, RING, SET_NSIDE);
    mutable Healpix_Map<precision> map_U = Healpix_Map<precision>(n_side, RING, SET_NSIDE);

    // The mask
    const Healpix_Map<precision> mask;

    // Set of alm values
    mutable Alm<xcomplex<precision>> alms_T = Alm<xcomplex<precision>>(l_max, l_max);
    mutable Alm<xcomplex<precision>> alms_E = Alm<xcomplex<precision>>(l_max, l_max);
    mutable Alm<xcomplex<precision>> alms_B = Alm<xcomplex<precision>>(l_max, l_max);

    // The result of the vector-matrix product A @ x
    mutable Eigen::Matrix<precision, Eigen::Dynamic, 1> Ax = Eigen::Matrix<precision, Eigen::Dynamic, 1>(2 * n_pix_mask);
};


#endif //CONJGRAD_EIGEN_MATRIXREPLACEMENT_EB_H
