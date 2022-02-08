CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(ConjGrad_Eigen)
SET(CMAKE_CXX_STANDARD 17)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  "/home/amaraio/Documents/PhD/Codes/eigen-3.4.0/build_install/share/eigen3/cmake/")
FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES("/home/amaraio/Documents/PhD/Codes/eigen-3.4.0/build_install/include/eigen3/")

INCLUDE_DIRECTORIES("/home/amaraio/Documents/PhD/Codes/Healpix_3.80/include/healpix_cxx/")
LINK_DIRECTORIES("/home/amaraio/Documents/PhD/Codes/Healpix_3.80/lib/")
LINK_LIBRARIES(healpix_cxx)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -flto -Wall -Wextra -fPIC -fopenmp")

ADD_EXECUTABLE(ConjGrad_Eigen main.cpp)

