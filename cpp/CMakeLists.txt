CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(ConjGrad_Eigen)
SET(CMAKE_CXX_STANDARD 17)

# Include Eigen
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  "/home/maraio/Codes/eigen-3.4.0/build_install/share/eigen3/cmake/")
INCLUDE_DIRECTORIES("/home/maraio/Codes/eigen-3.4.0/build_install/include/eigen3/")
FIND_PACKAGE(Eigen3 REQUIRED)

# Include HealPix
INCLUDE_DIRECTORIES("/home/maraio/Codes/Healpix_3.70/include/healpix_cxx/")
LINK_DIRECTORIES("/home/maraio/Codes/Healpix_3.70/lib/")
LINK_LIBRARIES(healpix_cxx)

# Set compilation flags to highest available optimisation settings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -flto -Wall -Wextra -fopenmp")

ADD_EXECUTABLE(QMLShearExe main.cpp)
ADD_LIBRARY(QMLShearLib SHARED py_library.cpp)
