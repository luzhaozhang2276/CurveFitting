//
// Created by lu on 2021/1/7.
//

#ifndef CURVEFITTING_POSECERES_H
#define CURVEFITTING_POSECERES_H

#include "commom_include.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

//cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  // K
// 代价函数的计算模型
/// 自动求导模型
struct PoseCost
{
    PoseCost ( cv::Point2f uv,cv::Point3f xyz, cv::Mat K) : _uv(uv), _xyz(xyz), _K(K) {}
    // 残差的计算
    template <typename T>
    bool operator() (
            const T *const camera,  // 位姿参数，有6维 (前三维: 旋转向量, 后三维: 平移向量)
            T *residual ) const     // 残差
    {
        T predictions[2];
        T point[3];
        point[0]=T(_xyz.x);
        point[1]=T(_xyz.y);
        point[2]=T(_xyz.z);

        CamProjectionWithoutDistortion(camera, point, predictions, _K);

        residual[0] = predictions[0] - T(_uv.x);
        residual[1] = predictions[1] - T(_uv.y);

        return true;
    }

    template <typename T>
    static inline bool CamProjectionWithoutDistortion(const T *camera, const T *point, T *predictions, cv::Mat K) {
        T p[3];
        // 参数: const T angle_axis[3], const T pt[3], T result[3]
        // (angle_axis为三维数组,传入六维的camera数组也可,因为传入的是地址,仅读取前三维,即旋转向量)
        // (使用旋转向量进行旋转计算)
        ceres::AngleAxisRotatePoint(camera, point, p);              // 计算 RP
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];    // 计算 RP + t

        T xp = p[0]/p[2];
        T yp = p[1]/p[2];   // xp,yp是归一化坐标，深度为p[2]

        predictions[0] = xp*K.at<double>(0,0)+K.at<double>(0,2);
        predictions[1] = yp*K.at<double>(1,1)+K.at<double>(1,2);
    }

    static ceres::CostFunction* Create(const cv::Point2f uv,const cv::Point3f xyz, const cv::Mat K) {
        return (new ceres::AutoDiffCostFunction<PoseCost, 2, 6>(
                new PoseCost(uv, xyz, K)));
    }


    const cv::Point2f _uv;
    const cv::Point3f _xyz;
    const cv::Mat _K;
};



void bundleAdjustmentCeres(
        const std::vector<cv::Point3d>& points_3d,
        const std::vector<cv::Point2d>& points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose);

#endif //CURVEFITTING_POSECERES_H
