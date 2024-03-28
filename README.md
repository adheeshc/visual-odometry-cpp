# Visual Odometry 

![License](https://img.shields.io/github/license/adheeshc/visual-odometry-cpp)
![Forks](https://img.shields.io/github/forks/adheeshc/visual-odometry-cpp)
![Stars](https://img.shields.io/github/stars/adheeshc/visual-odometry-cpp)
![Issues](https://img.shields.io/github/issues/adheeshc/visual-odometry-cpp)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

The demo is tested on an Ubuntu 20.04 Platform.

# Description

The purpose of this repository is to estimate the camera’s motion based on ORB feature matching. 
It is different for different camera settings (or the information available for the calculation):

* Camera is monocular, we have two sets of 2D points.
* If one set of points is 2D and one set is 3D.
* Camera is binocular, RGB-D or the distance is obtained by some method, and we have two sets of 3D points.

## 2D-2D Pose Estimation
The 2D-2D camera pose estimation problem is solved using Epipolar Geometry. It can be summarized as the following steps:

1. Find E or F based on the pixel positions of the matched points.
2. Find R, t based on E or F.
3. Triangulate to find best R & t.

## 2D-3D Pose Estimation
The 2D-2D camera pose estimation problem is solved using PnP (Perspective n-Points). It can be summarized as the following steps: 
1. Estimate PnP 
2. Perform Bundle Adjustment/Non Linear Optimization to estimate the camera pose by minimizing the reprojection error.

Here, EPnP is used to solve the PnP problem.
In this repository, two methods for Bundle Adjustment are explored and compared on accuracy and speed - 
1. A simple Gauss Newton Optimization written from scratch
2. g2o Gauss-Newton Optimization

## 3D-3D Pose Estimation
The 3D-3D camera pose estimation problem is solved using ICP (Iterative Closest Point). It can be summarized as the following steps: 

1. using linear algebra (mainly SVD) to estimate pose
2. using nonlinear optimization (similar to Bundle Adjustment) to improve pose.

# Data description
**1.png**: Image 1 
**1_depth.png**: Depth Image 1

**2.png**: Image 2 
**2_depth.png**: Depth Image 2

**TUM Dataset Camera Intrinsics**:
fx = 520.9 fy = 521.0 cx = 325.1 cy = 249.7

# Prerequisites for this demo
[OpenCV](https://github.com/opencv/opencv) : OpenCV is an C++ library for Image Manipulation.

[Eigen](https://github.com/libigl/eigen) : Eigen is a C++ template library for linear algebra.

[Sophus](https://github.com/strasdat/Sophus) : Sophus is an open-source C++ framework for Lie groups commonly used for 2D and 3D geometric problems. 

[g2o](https://github.com/RainerKuemmerle/g2o) : g2o is an open-source C++ library for optimizing graph-based nonlinear error functions.

# Build and Run

In parent directory 
```
mkdir build 
cd build  
cmake ..
```
run as per need 
```
./poseEstimation_2d2d || ./poseEstimation_2d3d || ./poseEstimation_3d3d
```