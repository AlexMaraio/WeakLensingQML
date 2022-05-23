"""
Script to plot the histogram of the Gaussian and non-Gaussian shear maps
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import healpy as hp
import pandas as pd

sns.set(font_scale=1.25, rc={'text.usetex': True})


# The location of where to save the plots into
plots_folder = "/home/maraio/Codes/WeakLensingQML/data/Plots/MapHistograms"
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

# Folders where our maps are stored in
folder_path_gaussian = "/disk01/maraio/NonGaussianShear/N1024_Gaussian/Maps_N256"
folder_path_lognormal = "/disk01/maraio/NonGaussianShear/N1024_LogNormal/Maps_N256"

map_num = 0

gamma1_gaussian, gamma2_gaussian = hp.read_map(f"{folder_path_gaussian}/Map{map_num}-f1z1.fits", field=[1, 2])
gamma1_lognormal, gamma2_lognormal = hp.read_map(f"{folder_path_lognormal}/Map{map_num}-f1z1.fits", field=[1, 2])


data_gamma1 = pd.DataFrame.from_dict({'Gaussian': gamma1_gaussian, 'Log-normal': gamma1_lognormal})
data_gamma2 = pd.DataFrame.from_dict({'Gaussian': gamma2_gaussian, 'Log-normal': gamma2_lognormal})

fig, ax = plt.subplots(figsize=(11, 7))

sns.histplot(data_gamma1, kde=True, palette='inferno', stat='count', common_bins=True, line_kws={'lw': 2.25})

plt.xlabel(r'$\gamma_1$')
plt.title(r'$\gamma_1$ distribution for shift $\vartheta = 0.04122$')

x_lim = np.max(np.abs(ax.get_xlim()))
ax.set_xlim(-x_lim, x_lim)

plt.tight_layout()
plt.savefig(f'{plots_folder}/GaussianHist_gamma1_N256.pdf')

# Gamma 2 map
fig, ax = plt.subplots(figsize=(11, 7))

sns.histplot(data_gamma2, kde=True, palette='inferno', stat='count', common_bins=True, line_kws={'lw': 2.25})

plt.xlabel(r'$\gamma_2$')
plt.title(r'$\gamma_2$ distribution for shift $\vartheta = 0.04122$')

x_lim = np.max(np.abs(ax.get_xlim()))
ax.set_xlim(-x_lim, x_lim)

plt.tight_layout()
plt.savefig(f'{plots_folder}/GaussianHist_gamma2_N256.pdf')
