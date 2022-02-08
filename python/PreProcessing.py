import healpy as hp
import numpy as np

from lib.PreProcessing.QMLClass import QML
from lib.PreProcessing.Y_matrix_class import YMatrix
from lib.PreProcessing.GenerateMask import generate_mask


if __name__ == '__main__':
    print('Starting main process now...')

    #* Map resolution of the run to compute
    n_side = 32
    l_max = 3 * n_side - 1
    ells = np.arange(2, l_max + 1)
    n_ell = l_max - 1

    #* Whether to use a star mask or not
    use_star_mask = True

    # Generate our own mask with a custom f_sky by varying theta
    mask = generate_mask(3.25, n_side)

    if use_star_mask:
        #* Read in our star mask
        print('Reading in the star mask now and combining them')
        star_mask = hp.read_map(f'../data/Masks/StarMask_N{n_side}.fits', dtype=bool)

        # Combine the original mask & star mask
        mask = np.logical_and(mask, star_mask)

        # Save our combined sky mask
        hp.write_map(f'../data/Masks/SkyMask_N{n_side}_whstars.fits', mask, overwrite=True, fits_IDL=False,
                     dtype=np.float64)

    else:
        # Save just the galactic & ecliptic mask
        hp.write_map(f'../data/Masks/SkyMask_N{n_side}_nostars.fits', mask, overwrite=True, fits_IDL=False,
                     dtype=np.float64)

    # Set up our YMatrix class
    Y_matrix = YMatrix(n_side=n_side, l_max=l_max, mask=mask)

    # Either compute the Y matrix now
    Y_matrix.compute_Y_matrix()
    # Y_matrix.save_array()
    # sys.exit()

    # Or save and read it in later
    # Y_matrix.load_array()

    #* Create our QML classes at the required redshifts
    ShearQML1 = QML(n_side=n_side, l_max=l_max, redshift=1.0, mask=mask, Y_matrix=Y_matrix)

    # Go through each class and compute various quantities
    ShearQML1.compute_theory_spectra()
    # ShearQML1.plot_cls()
    ShearQML1.compute_S_tilde_matrix()
    ShearQML1.compute_covariance_matrix()
    ShearQML1.compute_cov_inv_Y()
    ShearQML1.compute_Y_cov_inv_Y()
    # ShearQML.save_arrays()
    # ShearQML.load_arrays()
    ShearQML1.compute_Fisher_matrix()
