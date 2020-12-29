cmake_minimum_required(VERSION 3.17)
project(CurveFitting)

set(CMAKE_CXX_STANDARD 11)
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_executable(CurveFitting main.cpp)