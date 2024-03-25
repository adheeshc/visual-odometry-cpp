#include <iostream>
#include <opencv2/opencv.hpp>

void poseEstimation_2d2d(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& matches, const cv::Mat& K, cv::Mat& R, cv::Mat& t) {
    std::vector<cv::Point2f> points1, points2;
    //Convert the matches to vector<Point2f>
    for (int i = 0;i < matches.size();i++) {
        points1.emplace_back(kp1[matches[i].queryIdx].pt);
        points2.emplace_back(kp2[matches[i].trainIdx].pt);
    }

    //Calculate Fundamental Matrix
    cv::Mat fundamentalMatrix;
    fundamentalMatrix = cv::findFundamentalMat(points1, points2, fundamentalMatrix);
    std::cout << "fundamentalMatrix: \n" << fundamentalMatrix << std::endl;

    //Calculate Essential Matrix
    cv::Mat essentialMatrix;
    essentialMatrix = cv::findEssentialMat(points1, points2, K);
    std::cout << "essentialMatrix: \n" << essentialMatrix << std::endl;

    //Calculate Homography Matrix
    cv::Mat homographyMatrix;
    homographyMatrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "homographyMatrix: \n" << homographyMatrix << std::endl;

    //Recover R and t
    cv::recoverPose(essentialMatrix, points1, points2, K, R, t);
    std::cout << "R: \n" << R << std::endl;
    std::cout << "t: \n" << t << std::endl;

}

cv::Point2d pixel2cam(const cv::Point2d& p, cv::Mat& K) {
    double u = p.x - K.at<double>(0, 2) / K.at<double>(0, 0); // u = x - cx/fx
    double v = p.y - K.at<double>(1, 2) / K.at<double>(1, 1); // v = y - cy/fy

    return cv::Point2d(u, v);
}

void triangulate(
    std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<cv::DMatch>& matches,
    cv::Mat& K, cv::Mat& R, cv::Mat& t,
    std::vector<cv::Point3d>& points) {

    cv::Mat T1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat T2 = cv::Mat::eye(3, 4, CV_64F);
    T2.at<double>(0, 0) = R.at<double>(0, 0);
    T2.at<double>(0, 1) = R.at<double>(0, 1);
    T2.at<double>(0, 2) = R.at<double>(0, 2);
    T2.at<double>(1, 0) = R.at<double>(1, 0);
    T2.at<double>(1, 1) = R.at<double>(1, 1);
    T2.at<double>(1, 2) = R.at<double>(1, 2);
    T2.at<double>(2, 0) = R.at<double>(2, 0);
    T2.at<double>(2, 1) = R.at<double>(2, 1);
    T2.at<double>(2, 2) = R.at<double>(2, 2);

    T2.at<double>(0, 3) = t.at<double>(0, 0);
    T2.at<double>(1, 3) = t.at<double>(1, 0);
    T2.at<double>(2, 3) = t.at<double>(2, 0);

    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < matches.size(); i++) {
        points1.emplace_back(pixel2cam(kp1[matches[i].queryIdx].pt, K));
        points2.emplace_back(pixel2cam(kp2[matches[i].trainIdx].pt, K));
    }

    cv::Mat points4d;
    cv::triangulatePoints(T1, T2, points1, points2, points4d);

    // convert to Non-Homogenous coordinates
    for (int i = 0; i < points4d.cols; i++) {
        cv::Mat x = points4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.emplace_back(p);
    }
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
        if (match[i].distance <= std::max(2 * minDist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}


int main() {

    // Load Data
    std::string fileName1 = "../data/1.png";
    std::string fileName2 = "../data/2.png";
    cv::Mat img1 = cv::imread(fileName1);
    cv::Mat img2 = cv::imread(fileName2);
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

    cv::Mat R, t;
    poseEstimation_2d2d(kp1, kp2, matches, K, R, t);

    std::vector<cv::Point3d> points;
    triangulate(kp1, kp2, matches, K, R, t, points);

    for (int i = 0; i < matches.size(); i++) {
        //first camera
        cv::Point2f pt1Cam = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        cv::Point2f pt1Cam_3d(points[i].x / points[i].z, points[i].y / points[i].z);
        // std::cout << "point in first camera frame =" << pt1Cam << std::endl;
        // std::cout << "3d projected point =" << pt1Cam_3d << " , d = " << points[i].z << std::endl;

        //second camera
        cv::Point2f pt2Cam = pixel2cam(kp2[matches[i].trainIdx].pt, K);
        cv::Mat pt2_t = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_t /= pt2_t.at<double>(2, 0);
        // std::cout << "point in second camera frame =" << pt2Cam << std::endl;
        // std::cout << "3d projected point =" << pt2_t.t() << std::endl;
        // std::cout << std::endl;
    }

    std::cout << "Done!" << "\n";
    return 0;
}