# QML Core C++ Code

This directory contains the core C++ code that implements our conjugate-gradient approach to the quadratic maximum
likelihood estimator.

## Dependencies

This code requires two external C++ libraries and the CMake build system to have been downloaded and installed before
compiling this code. The dependencies are as follows:

- The [CMake](https://cmake.org/) build system to find external packages and compile this code
- The [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra package
- The [HealPix](https://healpix.sourceforge.io/) pixellisation and spherical harmonic transform library

Once these have been installed, their paths should then be specified in the `CMakeLists.txt` file such that the CMake
build system knows where to look for the dependencies. The variables `INCLUDE_DIRECTORIES` and `LINK_DIRECTORIES`
should be changed to reflect the instillation path.

## Building

Once the two dependencies have been installed, we can then compile the C++ code. To do so, issue the following commands:
```bash
cmake ..
make all
```

This should then build the code.

Please note that the code was written and run on machines using modern versions of Ubuntu Linux (20.04 & 22.04), while
it should run fine on other versions of Linux (and perhaps MacOS), this cannot be guaranteed and your mileage may vary!
If you do encounter any problems, then please raise an issue on the GitHub repository to see if this can be resolved.
