# Visual Odometry 

![License](https://img.shields.io/github/license/adheeshc/visual-odometry-cpp)
| <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" />| `https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white`|
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

    ### Some Issues with 2D-2D Pose Estimation
    1. Scale Ambiguity - The normalization of t directly leads to scale ambiguity in monocular vision.
    2. The Problem of Pure Rotation - In the decomposition of E to get R, t, if the camera is purely rotated, causing t to be zero, then E will also be zero, which will make it impossible for us to solve R.
    3. More Than Eight Pairs of Features - use RANSAC instead of least-squares to find the best E based on 8 random pairs.

## 2D-3D Pose Estimation
The 2D-2D camera pose estimation problem is solved using PnP (Perspective n-Points). It can be summarized as the following steps: 
1. Estimate PnP** using either of DLT, P3P, EPnP, UPnP etc.
2. Perform Bundle Adjustment/Non Linear Optimization to estimate the camera pose by finding minimizing the reprojection error.

Here, EPnP is used to solve the PnP problem.
In this repository, two methods for Bundle Adjustment are explored - 
1. A simple Gauss Newton Optimization written from scratch
2. g2o Gauss-Newton Optimization

## 3D-3D Pose Estimation
The 3D-3D camera pose estimation problem is solved using ICP (Iterative Closest Point). It can be summarized as the following steps: 




The demo is tested on Ubuntu 20.04 Platform.

# Data description
**1.png**: Image 1 
**1_depth.png**: Depth Image 1

**2.png**: Image 2 
**2_depth.png**: Depth Image 2

**TUM Dataset Camera Intrinsics**:
fx = 520.9 fy = 521.0 cx = 325.1 cy = 249.7

# Prerequisites for this demo
**OpenCV** : [OpenCV](https://github.com/opencv/opencv) OpenCV is an C++ library for Image Manipulation.
Dowload and install instructions can be found at: https://github.com/opencv/opencv.

**Eigen** : [Eigen](https://github.com/libigl/eigen) Eigen is a C++ template library for linear algebra.
Dowload and install instructions can be found at: https://github.com/libigl/eigen

**Sophus** : [Sophus](https://github.com/strasdat/Sophus) Sophus is an open-source C++ framework for Lie groups commonly used for 2D and 3D geometric problems. 
Dowload and install instructions can be found at: https://github.com/strasdat/Sophus.

**g2o** : We use [g2o](https://github.com/RainerKuemmerle/g2o) g2o is an open-source C++ library for optimizing graph-based nonlinear error functions.
Dowload and install instructions can be found at: https://github.com/RainerKuemmerle/g2o


# Build and Run

```
mkdir build  
cd build  
cmake ..

```