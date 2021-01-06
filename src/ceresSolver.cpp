//
// Created by lu on 2021/1/5.
//
#include <iostream>
#include "ceresSolver.h"

void ceresSolver(const std::vector<double> &y_data,
                 const std::vector<double> &x_data,
                 int N, double ae, double be, double ce)
{
    // 开始ceres 迭代
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始ceres迭代 : ";
    double abc[3] = {ae, be, ce};

    // 构建最小二乘问题
    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        /// 自动求导
        ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<Auto_Diff_Curve_Cost, 1, 3>(
                        new Auto_Diff_Curve_Cost(x_data[i], y_data[i]));

        /// 数值求导
        ceres::CostFunction* numeric_cost_function =
                new ceres::NumericDiffCostFunction<Numeric_Diff_Curve_Cost, ceres::CENTRAL,1 ,3>(
                        new Numeric_Diff_Curve_Cost(x_data[i], y_data[i]));

        /// 解析求导
        ceres::CostFunction* analytic_cost_function = new AnalyticCostFunctor(x_data[i], y_data[i]);

        /// 添加残差项 (替换误差函数即可)
        problem.AddResidualBlock(analytic_cost_function, new ceres::CauchyLoss(24.0), abc);
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // LOG(INFO) << summary.BriefReport();
    // LOG(INFO) << summary.final_cost;
    // LOG(INFO) << summary.num_successful_steps;

    std::ofstream file("./scripts/error-ceres.txt", std::ios::trunc);
    int iter = 1;
    for (const auto &p:summary.iterations)
    {
        file << iter << "," << p.cost << std::endl;
        iter++;
    }
    file.close();

    LOG(INFO) << "Approach to ceres parameter converge, itera_count = "
              << summary.num_successful_steps << "\tcost = " << summary.final_cost;
    LOG(INFO) << "ae = " << abc[0] << "\tbe = " << abc[1] << "\tce = " << abc[2];
    LOG(INFO) << "------------------------------------------------------------------------------";
}