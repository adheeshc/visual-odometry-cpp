#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

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

        }
    }

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

int main() {

    // Load Data
    std::string fileName1 = "../data/1.png";
    std::string fileName2 = "../data/2.png";
    std::string fileName3 = "../data/1_depth.png";
    cv::Mat img1 = cv::imread(fileName1);
    cv::Mat img2 = cv::imread(fileName2);
    cv::Mat d1 = cv::imread(fileName3);
    assert(img1.data != nullptr && img2.data != nullptr);

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
        ushort d = d1.ptr<unsigned short>(int(kp1[matches[i].queryIdx].pt.y))[int(kp1[matches[i].queryIdx].pt.x)];
        if (d == 0)
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        points_2d.emplace_back(kp2[matches[i].trainIdx].pt);
        points_3d.emplace_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
    }

    std::cout << "3d-2d pairs: " << points_3d.size() << std::endl;

    cv::Mat r, t;
    cv::solvePnP(points_3d, points_2d, K, cv::Mat(), r, t, false);
    cv::Mat R;
    cv::Rodrigues(r, R); //r is rotation vector, R is rotation matrix

    std::cout << "R: " << R << std::endl;
    std::cout << "t: " << t << std::endl;

    return 0;
}