//
// Created by lu on 2021/1/6.
//

#include "g2oSolver.h"

void g2oSolver(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce)
{
    double w_sigma = 1.0;   // 噪声Sigma值, 用于计算协方差矩阵(传递误差)

    // 构建图优化,先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;  // 每个误差项优化变量维度为3,误差值维度为1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

    // 梯度下降方法: GN,LM,DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;         // 图模型
    optimizer.setAlgorithm(solver);         // 设置求解器
    optimizer.setVerbose(true);     // 打开调试输出

    // 往图中增加顶点
    auto *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    // 往图中增加边
    for (int i = 0; i < N; ++i) {
        auto *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0,v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));    // 信息矩阵: 协方差矩阵之逆
        optimizer.addEdge(edge);
    }

    /// 执行优化
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始g2o迭代 : ";
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // 输出优化值
    LOG(INFO) << "Approach to g2o parameter converge, itera_count = "
              << 5 << "\tcost = " << optimizer.activeRobustChi2();
    LOG(INFO) << "ae = " << v->estimate()[0] << "\tbe = " << v->estimate()[1] << "\tce = " << v->estimate()[2];
    LOG(INFO) << "------------------------------------------------------------------------------";
}