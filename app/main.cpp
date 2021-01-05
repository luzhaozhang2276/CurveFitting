#include <iostream>
#include <vector>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "optimization.h"
#include "ceresSolver.h"

using namespace std;
using namespace Eigen;


int main(int argc, char **argv)
{
    char name[] = "log";
    GLogHelper log(name);


    double ar = 1.0, br =  2.0, cr = 1.0;   // 真实参数值
    // double ae = 2.0, be = -1.0, ce = 5.0;   // 初始估计参数值
    double ae = 0.9, be = 1.9, ce = 0.9;   // 初始估计参数值
    int N = 1000;                           // 数据点
    double w_sigma = 1.0;                   // 噪声sigma
    cv::RNG rng;                            // Opencv随机数生成器

    /// 生成测试数据并保存
    ofstream filePoints("./scripts/data.txt", ios::trunc);
    vector<double> x_data, y_data;
    for (int i = 0; i < N;  i++ )
    {
        double x = i / 1000.0;
        x_data.push_back(x);

        double y = exp(ar * x * x + br * x + cr)  + rng.gaussian(w_sigma * w_sigma);
        // outliers
        if (i == 20000)
        {
            // LOG(INFO) << "add outlier: (0.2, 3000)";
            y_data.push_back(3000.00);
            filePoints << x << "," << 3000.00 << endl;
        }
        else
        {
            y_data.push_back(y);
            filePoints << x << "," << y << endl;
        }
    }
    filePoints.close();



    ofstream fileResult("./scripts/result.txt", ios::trunc);
    fileResult << ar << ',' << br << ',' << cr;

    /// Gauss-Newton
    GN(y_data, x_data, N, ae, be, ce);
    fileResult << ae << ',' << be << ',' << ce;
    LOG(INFO);

    /// robust-Gauss-Newton
    R_GN(y_data, x_data, N, ae, be, ce);
    fileResult << ae << ',' << be << ',' << ce;
    LOG(INFO);

    /// Levenberg-Marquardt
    LM(y_data, x_data, N, ae, be, ce);
    fileResult << ae << ',' << be << ',' << ce;
    LOG(INFO);

    /// Levenberg-Marquardt-Nielsen
    LM_Nielsen(y_data, x_data, N, ae, be, ce);
    fileResult << ae << ',' << be << ',' << ce;
    LOG(INFO);

    /// robust-Levenberg-Marquardt
    R_LM(y_data,x_data,N, ae ,be, ce);
    fileResult << ae << ',' << be << ',' << ce;
    LOG(INFO);

    /// ceres
    ceresSolver(y_data, x_data, N, ae, be, ce);
    fileResult << ae << ',' << be << ',' << ce;
    LOG(INFO);

    fileResult.close();

    // LOG(INFO) << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
    system("bash ./scripts/display.sh");
    return 0;
}