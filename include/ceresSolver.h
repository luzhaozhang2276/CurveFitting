//
// Created by lu on 2021/1/5.
//

#ifndef CURVEFITTING_CERESSOLVER_H
#define CURVEFITTING_CERESSOLVER_H

#include "commom_include.h"
#include <ceres/ceres.h>


// 代价函数的计算模型
/// 自动求导模型
struct Auto_Diff_Curve_Cost {
    Auto_Diff_Curve_Cost(double x, double y) : _x(x), _y(y) {}

    // 残差的计算
    template<typename T>
    bool operator()(const T *const abc, T *residual ) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

/// 数值求导模型
struct Numeric_Diff_Curve_Cost {
    Numeric_Diff_Curve_Cost(double x, double y) : _x(x), _y(y) {}

    // 残差的计算
    bool operator()(const double *const abc, double *residual) const {
        residual[0] = _y - ceres::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        return true;
    }

    const double _x, _y;
};

/// 解析求导模型
class AnalyticCostFunctor : public ceres::SizedCostFunction<1, 3> {
public:
    AnalyticCostFunctor(double x, double y) : _x(x), _y(y) {}
    virtual ~AnalyticCostFunctor() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double ae = parameters[0][0];
        const double be = parameters[0][1];
        const double ce = parameters[0][2];
        residuals[0] = _y - exp(ae * _x * _x + be * _x + ce);

        // Compute the Jacobian if asked for.
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -_x * _x * exp(ae * _x * _x + be * _x + ce);  // de/da
            jacobians[0][1] = -_x * exp(ae * _x * _x + be * _x + ce);  // de/db
            jacobians[0][2] = -exp(ae * _x * _x + be * _x + ce);  // de/dc
        }
        return true;
    }


private:
    const double _x, _y;
};




void ceresSolver(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);


#endif //CURVEFITTING_CERESSOLVER_H
