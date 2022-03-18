CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(star_mask_generator)
SET(CMAKE_CXX_STANDARD 17)

# Locate GSL
INCLUDE_DIRECTORIES("/home/maraio/Codes/gsl-build/include/")
LINK_DIRECTORIES("/home/maraio/Codes/gsl-build/lib/")
LINK_LIBRARIES(gsl)
LINK_LIBRARIES(gslcblas)

# Locate HealPix
INCLUDE_DIRECTORIES("/home/maraio/Codes/Healpix_3.70/include/healpix_cxx/")
LINK_DIRECTORIES("/home/maraio/Codes/Healpix_3.70/lib/")
LINK_LIBRARIES(healpix_cxx)

# Set compilation flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -flto -Wall -Wextra -fopenmp")

# Add our .cpp file
ADD_EXECUTABLE(star_mask_generator main.cpp)
