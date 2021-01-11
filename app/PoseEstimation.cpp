//
// Created by lu on 2021/1/7.
//

#include <iostream>

#include "pose/initialize.h"
#include "pose/poseGN.h"
#include "pose/poseCeres.h"
#include "pose/poseG2o.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    char name[] = "log";                    // log file
    GLogHelper log(name);

    /// 初始化 数据生成
    LOG(INFO) << "------------------------------------------------------------------------------";
    string image1 = "./images/1.png";       // color image
    string image2 = "./images/2.png";
    string depth1 = "./images/1_depth.png"; // depth image
    string depth2 = "./images/2_depth.png";
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  // K
    vector<Point3d> pts_3d;                 // 第一帧的3D点数据
    vector<Point2d> pts_2d;                 // 第二帧的2D点数据
    generateData(K, image1, image2, depth1, depth2, pts_3d, pts_2d);    // 生成数据点
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    LOG(INFO) << "------------------------------------------------------------------------------";


    LOG(INFO);
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "calling solvePnP by opencv : ";
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t, R;
    // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cerr << fixed << setprecision(10) << "R= \n" << R << endl;
    cerr << fixed << setprecision(10) << "t= \n" << t << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // 矩阵转换输出
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv2eigen(R, R_eigen);
    cv2eigen(t, t_eigen);
    Sophus::SE3d SE3_Rt(R_eigen, t_eigen);
    cerr << fixed << setprecision(10) << "pose by opencv: \n" << SE3_Rt.matrix() << endl;
    LOG(INFO) << "opencv solve pnp in opencv cost time: " << time_used.count() << " seconds.";
    LOG(INFO) << "------------------------------------------------------------------------------";


    LOG(INFO);
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "calling bundle adjustment by gauss newton : ";
    // 初始解
    Sophus::SE3d pose_gn;
    Sophus::SE3d pose_gn_null;
    transformData(R, t, pose_gn, pts_3d, pts_2d, pts_3d_eigen, pts_2d_eigen);
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    LOG(INFO) << "------------------------------------------------------------------------------";


    LOG(INFO);
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "calling bundle adjustment by g2o : ";
    Sophus::SE3d pose_g2o;
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    LOG(INFO) << "------------------------------------------------------------------------------";



    LOG(INFO);
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "calling bundle adjustment by ceres : ";
    Sophus::SE3d pose_ceres;
    bundleAdjustmentCeres(pts_3d, pts_2d, K, pose_ceres);
    LOG(INFO) << "------------------------------------------------------------------------------";

    return 0;
}






