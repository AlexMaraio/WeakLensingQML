"""
File to generate an ensemble of lensing maps using Flask
"""

import sys
sys.path.append('/home/maraio/Codes/WeakLensingQML/python/lib/FlaskFiles/')

# from ..lib.FlaskFiles.FlaskDetail import Distribution, FlaskRun
from FlaskDetail import Distribution, FlaskRun


# The working map resolution
n_side = 1024

# Since we need to generate log-normal maps to a much higher resolution, we may need to downgrade the maps afterwards
output_n_side = 256

# The distribution from which to generate the realisations for - Gaussian or LogNormal
dist_type = Distribution.Gaussian
# dist_type = Distribution.LogNormal

# The number of maps to generate in the ensemble
num_maps = 1500

# Where should we store data for this run
# data_directory = f'/disk01/maraio/NonGaussianMaps/N{n_side}_{dist_type}/'
data_directory = f'/cephfs/maraio/NonGaussianMaps/N{n_side}_{dist_type}_shift_0_01214_whnoise/'

# The location of the Flask executable file
flask_executable = '/home/maraio/Codes/flask-UCL/bin/flask'

# Create our FlaskRun class with specified parameters
flask_run = FlaskRun(n_side=n_side, output_n_side=output_n_side, dist_type=dist_type, redshift=1.0,
                     folder_path=data_directory, flask_executable=flask_executable)

# Start by computing the theory Cl values
flask_run.compute_theory_spectra()

# Then generate our Flask configuration file
flask_run.generate_flask_config_file()

# Then generate the field information file also needed by Flask
flask_run.generate_field_file()

# flask_run.run_flask(run_num=0)

# Then run an ensemble of Flask realisations
flask_run.run_flask_ensemble(num_maps=num_maps, add_noise=True)
