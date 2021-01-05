//
// Created by lu on 2021/1/5.
//

#ifndef CURVEFITTING_CERESSOLVER_H
#define CURVEFITTING_CERESSOLVER_H

#include "commom_include.h"
#include <ceres/ceres.h>


// 代价函数的计算模型
struct CERES_CURVE_COST {
    CERES_CURVE_COST(double x, double y) : _x(x), _y(y) {}

    // 残差的计算
    template<typename T>
    bool operator()(const T *const abc, T *residual ) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

void ceresSolver(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);


#endif //CURVEFITTING_CERESSOLVER_H
