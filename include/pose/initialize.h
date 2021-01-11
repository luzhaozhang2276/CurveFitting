//
// Created by lu on 2021/1/7.
//

#ifndef CURVEFITTING_INITIALIZE_H
#define CURVEFITTING_INITIALIZE_H

#include "commom_include.h"

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

//vector<Point3f> pts_3d;       // 第一帧的3D点数据
//vector<Point2f> pts_2d;       // 第二帧的2D点数据

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void find_feature_matches(
        const cv::Mat &img_1, const cv::Mat &img_2,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::KeyPoint> &keypoints_2,
        std::vector<cv::DMatch> &matches);

bool checkRT(
        const std::vector<cv::Point3d> &points_3d1,
        const std::vector<cv::Point3d> &points_3d2,
        const std::vector<cv::Point2d> &points_2d2,
        const cv::Mat &K,
        Sophus::SE3d &pose);

bool generateData(const cv::Mat &K,
                  const std::string& img1, const std::string& img2,
                  const std::string& depth1, const std::string& depth2,
                  std::vector<cv::Point3d>& pts_3d, std::vector<cv::Point2d>& pts_2d);

bool transformData(const cv::Mat &R, const cv::Mat &t, Sophus::SE3d& pose,
                   std::vector<cv::Point3d>& pts_3d, std::vector<cv::Point2d>& pts_2d,
                   VecVector3d& pts_3d_eigen, VecVector2d& pts_2d_eigen);


#endif //CURVEFITTING_INITIALIZE_H
