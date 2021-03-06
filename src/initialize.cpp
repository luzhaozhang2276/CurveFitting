//
// Created by lu on 2021/1/7.
//

#include "include/pose/initialize.h"

using namespace std;
using namespace cv;

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

bool checkRT(
        const vector<Point3d> &points_3d1,
        const vector<Point3d> &points_3d2,
        const vector<Point2d> &points_2d2,
        const Mat &K,
        Sophus::SE3d &pose
)
{
    LOG(INFO) << "\nCheckRT";

    cv::Mat T;
    cv::eigen2cv(pose.matrix(), T);
    float fx = 520.9;
    float fy = 521.0;
    float cx = 325.1;
    float cy = 249.7;

    for (int i = 0; i < points_3d1.size(); i++) {
        cv::Mat pt_cam, pt_world;
        pt_cam.create(4, 1, CV_64FC1);
        pt_world.create(4, 1, CV_64FC1);

        pt_world.at<double>(0, 0) = points_3d1[i].x;
        pt_world.at<double>(1, 0) = points_3d1[i].y;
        pt_world.at<double>(2, 0) = points_3d1[i].z;
        pt_world.at<double>(3, 0) = 1;

        /// convert to camera coordinate
        pt_cam = T * pt_world;        // 根据相机外参,求解对应相机坐标系下的3D点

        Vec3d Xc;
        Xc[0] = pt_cam.at<double>(0, 0);
        Xc[1] = pt_cam.at<double>(1, 0);
        Xc[2] = pt_cam.at<double>(2, 0);

//        cout << i << ":" << endl;
//        cout << "3D1: " << pt_world.t() << endl;
//        cout << "3D:  " << pt_cam.t() << endl;
//        cout << "3D2: " << points_3d2[i] << endl;
//        cout << "2D:  [" << fx * Xc[0] / Xc[2] + cx << " , " << fy * Xc[1] / Xc[2] + cy << "]" << endl;
//        cout << "2D2: " << points_2d2[i] << endl;

//        Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
//        cout << endl;
    }
    return true;
}

bool generateData(const cv::Mat &K,
                  const std::string& img1, const std::string& img2,
                  const std::string& depth1, const std::string& depth2,
                  std::vector<cv::Point3d>& pts_3d, std::vector<cv::Point2d>& pts_2d)
{
    //-- 读取图像
    Mat img_1 = imread(img1, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(img2, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    LOG(INFO) << "一共找到了" << matches.size() << "组匹配点";

    // 建立3D点
    Mat d1 = imread(depth1, CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像

    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    Mat d2 = imread(depth2, CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    vector<Point3f> pts_3d_two;       // 第二帧的3D点数据
    for (DMatch m:matches) {
        ushort d = d2.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        pts_3d_two.push_back(Point3f(p2.x * dd, p2.y * dd, dd));
    }

    LOG(INFO) << "3d-2d pairs: " << pts_3d.size();
    return true;
}

bool transformData(const cv::Mat &R, const cv::Mat &t, Sophus::SE3d& pose,
                   std::vector<cv::Point3d>& pts_3d, std::vector<cv::Point2d>& pts_2d,
                   VecVector3d& pts_3d_eigen, VecVector2d& pts_2d_eigen)
{
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    // 赋值 初始解
    Eigen::Matrix3d eiR;
    Eigen::Vector3d t_ei;
    cv::cv2eigen(R, eiR);
    cv::cv2eigen(t, t_ei);
    Sophus::SE3d pose_gn(eiR, t_ei);
    pose = pose_gn;
    return true;
}