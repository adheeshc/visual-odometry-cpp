#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <g2o/core/base_vertex.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>

cv::Point2d pixel2cam(const cv::Point2d& p, cv::Mat& K) {
    double u = p.x - K.at<double>(0, 2) / K.at<double>(0, 0); // u = x - cx/fx
    double v = p.y - K.at<double>(1, 2) / K.at<double>(1, 1); // v = y - cy/fy

    return cv::Point2d(u, v);
}

void findFeatureMatches(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& matches) {
    cv::Mat desc1, desc2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    detector->detect(img1, kp1);
    detector->detect(img2, kp2);

    descriptor->compute(img1, kp1, desc1);
    descriptor->compute(img2, kp2, desc2);

    std::vector<cv::DMatch> match;
    matcher->match(desc1, desc2, match);

    //remove outliers
    double minDist = INT32_MAX, maxDist = 0;

    for (int i = 0; i < desc1.rows;i++) {
        double dist = match[i].distance;
        minDist = std::min(minDist, dist);
        maxDist = std::max(maxDist, dist);
    }
    for (int i = 0; i < desc1.rows; i++) {
        if (match[i].distance <= std::max(2 * minDist, 30.0)) {
            matches.emplace_back(match[i]);
        }
    }
}

void poseEstimation_3d3d(std::vector<cv::Point3f>& pts1, std::vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t) {
    cv::Point3f p1, p2; //centers
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);

    std::vector<cv::Point3f> q1(N), q2(N); //pcd with centres removed
    for (int i = 0; i < N;i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    //compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N;i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    // std::cout << "W = \n" << W << std::endl;

    //SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0) {
        {
            for (int i = 0; i < 3; i++) {
                U(i, 2) *= -1;
            }
        }
    }
    // std::cout << "U = \n" << U << std::endl;
    // std::cout << "V = \n" << V << std::endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * (Eigen::Vector3d(p2.x, p2.y, p2.z));

    // //convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2)
        );

    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

    cv::Mat T = cv::Mat::eye(3, 4, CV_64F);
    T.at<double>(0, 0) = R.at<double>(0, 0);
    T.at<double>(0, 1) = R.at<double>(0, 1);
    T.at<double>(0, 2) = R.at<double>(0, 2);
    T.at<double>(1, 0) = R.at<double>(1, 0);
    T.at<double>(1, 1) = R.at<double>(1, 1);
    T.at<double>(1, 2) = R.at<double>(1, 2);
    T.at<double>(2, 0) = R.at<double>(2, 0);
    T.at<double>(2, 1) = R.at<double>(2, 1);
    T.at<double>(2, 2) = R.at<double>(2, 2);

    T.at<double>(0, 3) = t.at<double>(0, 0);
    T.at<double>(1, 3) = t.at<double>(1, 0);
    T.at<double>(2, 3) = t.at<double>(2, 0);

    std::cout << "Original Estimated Pose: \n" << T << std::endl;
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //override reset function
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    //override plus operator
    virtual void oplusImpl(const double* update) override {
        Eigen::Matrix<double, 6, 1> updateEigen;
        updateEigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(updateEigen) * _estimate;
    }

    //dummy read/write function
    virtual bool read(std::istream& in) {};
    virtual bool write(std::ostream& out) const {};
};

class EdgeProjectionXYZRGBD : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
private:
    Eigen::Vector3d _point;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectionXYZRGBD(const Eigen::Vector3d point) : _point(point) {};

    //define error term computation
    virtual void computeError() override {
        const VertexPose* v = static_cast<const VertexPose*>(_vertices[0]);
        _error = _measurement - v->estimate() * _point;
    }

    //Jacobian
    virtual void linearizeOplus() override {
        const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d xyz_t = T * _point;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_t);
    }

    //dummy read/write function
    virtual bool read(std::istream& in) {};
    virtual bool write(std::ostream& out) const {};
};

void bundleAdjustment(const std::vector<cv::Point3f>& pts1,
    const std::vector<cv::Point3f>& pts2,
    cv::Mat& R, cv::Mat& t,
    Sophus::SE3d& pose) {

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  //pose -6, landmark(x,y,z) - 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    //vertex
    VertexPose* vertexPose = new VertexPose();
    vertexPose->setId(0);
    vertexPose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertexPose);

    //edges
    int index = 1;
    std::vector<EdgeProjectionXYZRGBD*> edges;
    for (size_t i = 0; i < pts1.size();i++) {
        EdgeProjectionXYZRGBD* edge = new EdgeProjectionXYZRGBD(
            Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(index);
        edge->setVertex(0, vertexPose);
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);
        optimizer.addEdge(edge);
        index++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    pose = vertexPose->estimate();
    std::cout << "Pose Estimate after BA: \n" << pose.matrix() << std::endl;

}

int main() {
    std::string fileName1 = "../data/1.png";
    std::string fileName2 = "../data/2.png";
    std::string fileName3 = "../data/1_depth.png";
    std::string fileName4 = "../data/2_depth.png";
    cv::Mat img1 = cv::imread(fileName1);
    cv::Mat img2 = cv::imread(fileName2);
    cv::Mat depth1 = cv::imread(fileName3);
    cv::Mat depth2 = cv::imread(fileName4);
    assert(img1.data != nullptr && img2.data != nullptr);
    assert(depth1.data != nullptr && depth2.data != nullptr);

    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::DMatch> matches;
    findFeatureMatches(img1, img2, kp1, kp2, matches);
    std::cout << "total matches: " << matches.size() << std::endl;

    //Camera intrinsics, TUM Freiburg
    cv::Point2d pp(325.1, 249.7); //principal point = (cx,cy);
    double fx = 520.9; //focal length = fx = fy;
    double fy = 521.0; //focal length = fx = fy;

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;  //fx
    K.at<double>(1, 1) = fy;  //fy
    K.at<double>(0, 2) = pp.x; //cx
    K.at<double>(1, 2) = pp.y; //cy

    std::vector<cv::Point3f> pts1, pts2;

    for (int i = 0; i < matches.size(); i++) {
        ushort d1 = depth1.ptr<unsigned short>(int(kp1[matches[i].queryIdx].pt.y))[int(kp1[matches[i].queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(kp2[matches[i].trainIdx].pt.y))[int(kp2[matches[i].trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) { //bad depth
            continue;
        }
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d1) / 5000.0;

        cv::Point2d p1 = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(kp2[matches[i].trainIdx].pt, K);
        pts1.emplace_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.emplace_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }
    std::cout << "3d-3d pairs: " << pts1.size() << std::endl;
    cv::Mat R, t;
    poseEstimation_3d3d(pts1, pts2, R, t);
    Sophus::SE3d pose;
    bundleAdjustment(pts1, pts2, R, t, pose);

    return 0;
}