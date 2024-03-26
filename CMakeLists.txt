cmake_minimum_required(VERSION 3.0.0)
project(poseEstimation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(g2o REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

set(source_files
    src/poseEstimation_2d2d.cpp
)

add_executable(poseEstimation_2d2d ${source_files})
target_link_libraries(poseEstimation_2d2d ${OpenCV_LIBS})
target_link_libraries(poseEstimation_2d2d Eigen3::Eigen)

set(source_files
    src/poseEstimation_2d3d.cpp
)

add_executable(poseEstimation_2d3d ${source_files})
target_link_libraries(poseEstimation_2d3d ${OpenCV_LIBS})
target_link_libraries(poseEstimation_2d3d Eigen3::Eigen)
target_link_libraries(poseEstimation_2d3d Sophus::Sophus)
target_link_libraries(poseEstimation_2d3d Ceres::ceres)
target_link_libraries(poseEstimation_2d3d ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D} ${G2O_TYPES_SBA})
target_link_libraries(poseEstimation_2d3d g2o_core g2o_stuff g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_types_sba g2o_types_slam2d g2o_types_slam3d g2o_viewer)