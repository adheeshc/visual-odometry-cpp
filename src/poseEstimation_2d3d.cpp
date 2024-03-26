#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <g2o/core/base_vertex.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Function to convert std::vector<cv::Point3f> to VecVector3d
VecVector3d convertToEigen3d(const std::vector<cv::Point3f>& points_3d) {
    VecVector3d eigenPoints;
    eigenPoints.reserve(points_3d.size()); // Reserve space to avoid reallocation

    // Convert each cv::Point3f to Eigen::Vector3d
    for (const auto& point : points_3d) {
        Eigen::Vector3d eigenPoint(point.x, point.y, point.z);
        eigenPoints.push_back(eigenPoint);
    }
    return eigenPoints;
}

VecVector2d convertToEigen2d(const std::vector<cv::Point2f>& points_2d) {
    VecVector2d eigenPoints;
    eigenPoints.reserve(points_2d.size()); // Reserve space to avoid reallocation

    // Convert each cv::Point3f to Eigen::Vector3d
    for (const auto& point : points_2d) {
        Eigen::Vector2d eigenPoint(point.x, point.y);
        eigenPoints.push_back(eigenPoint);
    }
    return eigenPoints;
}

void bundleAdjustmentGaussNewton(
    const VecVector3d& points_3d,
    const VecVector2d& points_2d,
    const cv::Mat& K,
    Sophus::SE3d& pose
) {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 30;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);  //fx
    double fy = K.at<double>(1, 1);  //fy
    double cx = K.at<double>(0, 2); //cx
    double cy = K.at<double>(1, 2); //cy

    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        for (int i = 0;i < points_3d.size();i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double invZ = 1.0 / pc[2];
            double invZ2 = invZ * invZ;
            double u = fx * pc[0] / pc[2] + cx;
            double v = fy * pc[1] / pc[2] + cy;
            Eigen::Vector2d proj(u, v);
            Eigen::Vector2d e = points_2d[i] - proj;
            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J(0, 0) = -fx * invZ;
            J(0, 1) = 0;
            J(0, 2) = fx * pc[0] * invZ2;
            J(0, 3) = fx * pc[0] * pc[1] * invZ2;
            J(0, 4) = -(fx + fx * pc[0] * pc[0] * invZ2);
            J(0, 5) = fx * pc[1] * invZ;

            J(1, 0) = 0;
            J(1, 1) = -fy * invZ;
            J(1, 2) = fy * pc[1] * invZ2;
            J(1, 3) = fy + fy * pc[1] * pc[1] * invZ2;
            J(1, 4) = -fy * pc[0] * pc[1] * invZ2;
            J(1, 5) = -fy * pc[0] * invZ;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            std::cout << "result is nan";
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            //COST INCREASE
            // std::cout << "cost: " << cost << ", last cost: " << lastCost << std::endl;
            break;
        }

        //update estimate
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        // std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
        if (dx.norm() < 1e-6) {//convergenece    
            break;
        }
    }
    std::cout << "Pose Estimate by GaussNewton: \n" << pose.matrix() << std::endl;
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

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjection(const Eigen::Vector3d& pos, const Eigen::Matrix3d& K) : _pos3d(pos), _K(K) {};

    //define error term computation
    virtual void computeError() override {
        const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d posPixel = _K * (T * _pos3d);
        posPixel /= posPixel[2];
        _error = _measurement - posPixel.head<2>();
    }

    //Jacobian
    virtual void linearizeOplus() override {
        const VertexPose* v = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d posCam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = posCam[0];
        double Y = posCam[1];
        double Z = posCam[2];
        double Z2 = Z * Z;
        // _jacobianOplusXi << -fx / Z, 0, fx* X / Z2, fx* X* Y / Z2, -fx - fx * X * X / Z2, fx* Y / Z,
            // 0, -fy / Z, fy* Y / Z2, fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
        _jacobianOplusXi(0, 0) = -fx / Z;
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(0, 2) = fx * X / Z2;
        _jacobianOplusXi(0, 3) = fx * X * Y / Z2;
        _jacobianOplusXi(0, 4) = -fx - fx * X * X / Z2;
        _jacobianOplusXi(0, 5) = fx * Y / Z;

        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = -fy / Z;
        _jacobianOplusXi(1, 2) = fy * Y / Z2;
        _jacobianOplusXi(1, 3) = fy + fy * Y * Y / Z2;
        _jacobianOplusXi(1, 4) = -fy * X * Y / Z2;
        _jacobianOplusXi(1, 5) = -fy * X / Z;
    }

    //dummy read/write function
    virtual bool read(std::istream& in) {};
    virtual bool write(std::ostream& out) const {};
};

void bundleAdjustmentg2o(
    const VecVector3d& points_3d,
    const VecVector2d& points_2d,
    const cv::Mat& K,
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

    //convert K
    Eigen::Matrix3d KEigen;
    KEigen <<
        K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // //edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size();i++) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection* edge = new EdgeProjection(p3d, KEigen);
        edge->setId(index);
        edge->setVertex(0, vertexPose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    pose = vertexPose->estimate();
    std::cout << "Pose Estimate by g2o: \n" << pose.matrix() << std::endl;
}

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

void poseEstimation_2d3d(std::vector<cv::Point3f>& points_3d, std::vector<cv::Point2f>& points_2d, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& matches, const cv::Mat& K, cv::Mat& R, cv::Mat& t) {

    cv::Mat r;
    cv::solvePnP(points_3d, points_2d, K, cv::Mat(), r, t, false);
    cv::Rodrigues(r, R); //r is rotation vector, R is rotation matrix

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

int main() {

    // Load Data
    std::string fileName1 = "../data/1.png";
    std::string fileName2 = "../data/2.png";
    std::string fileName3 = "../data/1_depth.png";
    cv::Mat img1 = cv::imread(fileName1);
    cv::Mat img2 = cv::imread(fileName2);
    cv::Mat depth1 = cv::imread(fileName3);
    assert(img1.data != nullptr && img2.data != nullptr);
    assert(depth1.data != nullptr);

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

    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;

    for (int i = 0; i < matches.size(); i++) {
        ushort d1 = depth1.ptr<unsigned short>(int(kp1[matches[i].queryIdx].pt.y))[int(kp1[matches[i].queryIdx].pt.x)];
        if (d1 == 0)
            continue;
        float dd = d1 / 5000.0;
        cv::Point2d p1 = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        points_2d.emplace_back(kp2[matches[i].trainIdx].pt);
        points_3d.emplace_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
    }
    std::cout << "3d-2d pairs: " << points_3d.size() << std::endl;

    cv::Mat R, t;
    poseEstimation_2d3d(points_3d, points_2d, kp1, kp2, matches, K, R, t);
    


    Sophus::SE3d pose;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(convertToEigen3d(points_3d), convertToEigen2d(points_2d), K, pose);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Time Taken by GaussNewton: " << timeTaken.count() << " seconds" << std::endl;

    t1 = std::chrono::steady_clock::now();
    bundleAdjustmentg2o(convertToEigen3d(points_3d), convertToEigen2d(points_2d), K, pose);
    t2 = std::chrono::steady_clock::now();
    timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Time Taken by g2o: " << timeTaken.count() << " seconds" << std::endl;

    return 0;
}