//
// Created by lu on 2021/1/7.
//

#include "pose/initialize.h"
#include "pose/poseCeres.h"

using namespace std;
using namespace cv;

void bundleAdjustmentCeres(
        const std::vector<cv::Point3d>& points_3d,
        const std::vector<cv::Point2d>& points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose)
{
    double camera[6] = {0, 1, 2, 0, 0, 0};

    ceres::Problem problem;
    for (int i = 0; i < points_2d.size(); ++i)
    {
        ceres::CostFunction* cost_function = PoseCost::Create(points_2d[i],points_3d[i], K);
//                new ceres::AutoDiffCostFunction<PoseCost, 2, 6>(
//                        new PoseCost(points_2d[i],points_3d[i]));
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 camera);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";

    Mat R_vec = (Mat_<double>(3,1) << camera[0],camera[1],camera[2]);//数组转cv向量
    Mat R_cvest;
    Rodrigues(R_vec,R_cvest);//罗德里格斯公式，旋转向量转旋转矩阵
    // cout<<"R_cvest="<<R_cvest<<endl;

    Eigen::Matrix3d R_est;
    cv2eigen(R_cvest,R_est);//cv矩阵转eigen矩阵
    // cout<<"R_est="<<R_est<<endl;

    Eigen::Vector3d t_est(camera[3],camera[4],camera[5]);
    // cout<<"t_est="<<t_est<<endl;
    Eigen::Isometry3d T(R_est);//构造变换矩阵与输出
    T.pretranslate(t_est);
    cout << fixed << setprecision(10) << T.matrix() << endl;

}
// 配置求解器
//ceres::Solver::Options options;
//options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//options.minimizer_progress_to_stdout = false;
//
//ceres::Solver::Summary summary;
//ceres::Solve(options, &problem, &summary);