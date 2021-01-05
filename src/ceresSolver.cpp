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
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CERES_CURVE_COST, 1, 3>(
                        new CERES_CURVE_COST(x_data[i], y_data[i])),
                nullptr, abc
        );
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