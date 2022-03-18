# star_mask_generator

Program to generate a random star mask, originally based on the `GenStarMask` utility as part of the `flask` code
which can be obtained from https://github.com/ucl-cosmoparticles/flask/blob/master/src/GenStarMask.cpp  
The license for `flask` is available at https://github.com/ucl-cosmoparticles/flask/blob/master/LICENSE.txt

Edits have been made to generate the stellar radii from a uniform distribution instead of log-normal distribution,
and to modernize the code.

## Dependencies

The star mask generation utility depends on the following two codes:

- The GNU scientific library (GSL), which can be obtained from https://www.gnu.org/software/gsl/
- The HealPix spherical pixelation library, which can be obtained from https://healpix.sourceforge.io/

## Compiling

This is a C++ file that can be built using the CMake build system. Once GSL and HealPix have been built and installed,
the `CMakeLists.txt` file should be edited to contain the locations of the `/include` and `/lib` subdirectories for the
two codes. The main code can then be compiled through CMake using the commands

```shell
mkdir build
cd build
cmake ..
make
```

## Running

Once built, the code can be executed from the main directory using the following command:
```shell
./build/star_mask_generator <RANDOM_SEED> <N_SIDE> <R_MIN> <R_MAX> <F_SKY> <FILENAME> 
```

where the input parameters are:

- RAND_SEED (int): Random seed to use for the GSL random number generator
- N_SIDE (int): The N_side parameter for generated star mask map
- R_MIN (float): The minimum angular size of the stars to generate (in arc min)
- R_MAX (float): The maximum angular size of the stars to generate (in arc min)
- F_SKY (float): The fraction of sky area (0.0 < F_SKY < 1.0) to cover with our randomly generated stars
- FILENAME (string): The output file name for our generated star mask
