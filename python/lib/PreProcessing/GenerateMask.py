import numpy as np
import healpy as hp


def generate_mask(theta, n_side):
    """
    Function to generate a mask that is built up of galactic and ecliptic cuts only.

    The parameter theta describes the angular size of these two cuts.
    """
    n_pix = 12 * (n_side ** 2)

    # Create the galactic cut coordinates, in radians
    gal_th1 = (theta / 2 - 0.5) * np.pi / theta
    gal_th2 = (theta / 2 + 0.5) * np.pi / theta

    # Change the value of theta slightly for the ecliptic cut
    theta += 2

    # Now create the ecliptic plane cut coordinates
    elp_th1 = (theta / 2 - 0.5) * np.pi / theta
    elp_th2 = (theta / 2 + 0.5) * np.pi / theta

    # Create cut in galactic plane
    region_gal = hp.query_strip(nside=n_side, theta1=gal_th1, theta2=gal_th2)

    # Create cut in ecliptic plane
    region_elp = hp.query_strip(nside=n_side, theta1=elp_th1, theta2=elp_th2)

    map_gal = np.ones(n_pix, dtype=bool)
    map_elp = np.ones(n_pix, dtype=bool)

    # Mask out any regions according to our mask
    map_gal[region_gal] = 0
    map_elp[region_elp] = 0

    # Combine both masks into a single map in galactic coordinates
    map_both = np.logical_and(map_gal, hp.rotator.Rotator(coord='EG').rotate_map_pixel(map_elp)).astype(bool)

    # Compute the sky fraction allowed through by this mask
    f_sky = map_both.sum() / map_both.size

    print(f'The f_sky of the combined galactic & ecliptic map is {(f_sky * 100):.3f} % ')

    # Return combined mask
    return map_both


def plot_mask(mask, plot_title, file_name):
    """
    Function to plot a mask in a consistent way that has already been generated
    """
    # Import plotting module locally
    from matplotlib import pyplot as plt

    # Use the Mollweide projection to plot our mask
    hp.mollview(mask, title=plot_title, cmap='nipy_spectral', cbar=False)

    # Save figure to plots subdirectory of the masks folder
    plt.savefig(f'../data/masks/plots/{file_name}.pdf')
