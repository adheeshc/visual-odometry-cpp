cmake_minimum_required(VERSION 3.0.0)
project(poseEstimation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})

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