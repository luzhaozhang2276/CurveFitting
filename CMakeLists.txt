cmake_minimum_required(VERSION 3.17)
project(CurveFitting)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# G2O
find_package( G2O REQUIRED )
# OpenCV
find_package( OpenCV 3.1 REQUIRED )
# glog
#find_package (glog 0.4.0 REQUIRED)
add_subdirectory(Thirdparty/glog)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${G2O_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
        src/GLogHelper.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        g2o_core g2o_stuff g2o_types_sba
        glog::glog
        )

############### executable ######################
add_executable(demo main.cpp)
target_link_libraries(demo ${PROJECT_NAME})