cmake_minimum_required(VERSION 3.17)
project(CurveFitting)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Sophus
find_package(Sophus REQUIRED)
# Eigen
include_directories("/usr/include/eigen3" )
# G2O
find_package(G2O REQUIRED)
# ceres
find_package(Ceres REQUIRED)
# OpenCV
find_package(OpenCV 3.1 REQUIRED )
# glog
#find_package (glog 0.4.0 REQUIRED)
add_subdirectory(Thirdparty/glog)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${Sophus_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
        src/GLogHelper.cpp
        src/ceresSolver.cpp
        src/optimization.cpp
        src/g2oSolver.cpp
        src/initialize.cpp
        src/poseGN.cpp
        src/poseCeres.cpp
        src/poseG2o.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
#        ${Sophus_LIBRARIES}
        ${CERES_LIBRARIES}
        g2o_core g2o_stuff g2o_types_sba
#        glog::glog
        )

############### executable ######################
add_executable(curve app/CurveFitting.cpp)
target_link_libraries(curve ${PROJECT_NAME})

add_executable(pose app/PoseEstimation.cpp)
target_link_libraries(pose ${PROJECT_NAME})