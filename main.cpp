#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
//#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "GLogHelper.h"

// #define random(x)  rand( x)/RAND_MAX

using namespace std;
using namespace Eigen;

void GN(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce);
void LM(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce);
void R_GN(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce);
void LM_Nielsen(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce);
void R_LM(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce);
double getRand();

const double function_tolerance = 1e-5;
const double max_trans_epsilon = 1e-5;
const double parameter_tolerance = 1e-5;
const int iterations = 1000;

int main(int argc, char **argv)
{
    char name[] = "log";
    GLogHelper log(name);

    double ar = 1.0, br =  2.0, cr = 1.0;   // 真实参数值
//    double ae = 2.0, be = -1.0, ce = 5.0;   // 初始估计参数值
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
        if (i == 200)
        {
            LOG(INFO) << "add outlier: (0.2, 3000)";
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

    GN( y_data, x_data, N, ae, be, ce);
    LOG(INFO);
    R_GN(y_data, x_data, N, ae, be, ce);
    LOG(INFO);
    LM(y_data, x_data, N, ae, be, ce);
    LOG(INFO);
    LM_Nielsen(y_data, x_data, N, ae, be, ce);
    LOG(INFO);
    R_LM(y_data,x_data,N, ae ,be, ce);
    LOG(INFO);

//    ofstream fileResult("./scripts/result.txt", ios::trunc);
//    fileResult << ae << ',' << be << ',' << ce;
//    fileResult.close();

    // outliers
//    y_data[988] = 0.00;
//    y_data[998] = 0.00;

//    LOG(INFO) << "手动 增加 outliers : "<<std::endl;
//    LOG(INFO) << "令　y[988] = 0.0 ，y[998] = 0.0  　再次求解: ";

    LOG(INFO) << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
    system("bash ./scripts/display.sh");
    return 0;
}

//  得到 0 -1 之间的随机数
double getRand()
{
    const int n = 99;
    srand( time(nullptr));
    return 2.0* ( rand()%(n+1) / (double) (n+1) ) -1.0 ;
}


/**
 * @brief Gauss-Newton
 */
void GN ( const vector<double>& y_data , const vector<double>& x_data , int N ,double ae ,double be ,double ce)
{
    // 开始Gauss-Newton迭代
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始Gauss-Newton迭代 : ";

    double cost, lastCost = 1e8;  // 本次迭代的cost和上一次迭代的cost

    ofstream file("./scripts/error-GN.txt", ios::trunc);
    for ( int iter = 0; iter < iterations; iter++ )
    {
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        cost = 0;

        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i];
            double error= yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += J * J.transpose();
            b += - error * J;
            cost += error * error;
        } // for

        // 求解线性方程 Hx=b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0]))
        {
            LOG(INFO) << "result is nan!";
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            LOG(INFO) << "Approach to GN parameter converge, itera_count = " << iter + 1 << "\tcost = " << cost;
            LOG(INFO) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
            LOG(INFO) << "------------------------------------------------------------------------------";
            break;
        }

        // update
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        lastCost = cost;
        cost =0;
        // Calculation
//        for (int i = 0; i < N; i++)
//        {
//            double xi = x_data[i], yi = y_data[i];
//            double error = yi - exp(ae * xi * xi + be * xi + ce);
//            cost += error * error;
//        } // for
//        LOG(INFO) << "dx = " << dx[0] << "\t," << dx[1] << "\t," << dx[2];
//        LOG(INFO) << "iter: " << iter << "  ,cost: " << lastCost << " ae = "<< ae << "  be = " <<  be << "  ce = " << ce;
        file << iter + 1 << "," << lastCost << endl;
    } // for
    file.close();
}

/**
 * @brief LM
 */
void LM ( const vector<double>& y_data , const vector<double>& x_data , int N ,double ae ,double be ,double ce)
{
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始 LM 迭代 :";
    // 开始迭代
    double u = 0.0 ;
    ofstream file("./scripts/error-LM.txt", ios::trunc);
    for ( int iter = 0 ; iter < iterations ; iter ++ )
    {
        // LOG(INFO) << "iter: " << iter;
        double cost = 0.0, newcost = 0.0;
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        // 计算 H b
        for(int i = 0 ; i < N ; i ++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error= yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += J * J.transpose();
            b += - error * J;
            cost += error * error;
        }

        if (iter == 0) {
            u = (1e-8) * max(H(0, 0), max(H(1, 1), H(2, 2)));
//            LOG(WARNING) << "u0 = " << u;
        }

//        LOG(WARNING) << "u = " << u;
        H = H + ( u * Eigen::MatrixXd::Identity(3, 3));
        //  solver update
        Eigen::Vector3d dx = H.ldlt().solve(b);

        // Matrix<double ,1,1> L ;
        Eigen::Matrix<double,1,1> L;    // 分母
        L(0,0) = 0.5 * dx.transpose() * (u * dx + b);

        // try to update
        double new_ae = ae + dx[0];
        double new_be = be + dx[1];
        double new_ce = ce + dx[2];

        // 重新计算cost
        newcost = 0;
        for(int i = 0 ; i < N ; i++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error = 0;
            error = yi - exp(new_ae * xi * xi + new_be * xi + new_ce);
            newcost += error * error;
        }

        // 没达到收敛条件则继续迭代
        //      cost - newcost
        // p = -------------
        //        L(0,0)
        double p = (cost - newcost) / L(0, 0) ;
        if (p < 0.25){
            u = 2 * u;
        } else if (p > 0.75) {
            u = u / 3;
        }

        // update parameter
        if (p > 0)
        {
            ae = new_ae;
            be = new_be;
            ce = new_ce;
        }

        // 判断收敛
        if (iter > 0 &&
            sqrt(pow(dx[0], 2) + pow(dx[1], 2) + pow(dx[2], 2)) <=
            (sqrt(pow(new_ae, 2) + pow(new_ce, 2) + pow(new_be, 2)) + parameter_tolerance) * parameter_tolerance
                )
        {
            LOG(INFO) << "Approach to LM parameter converge, iterations count = " << iter + 1 << "\tcost = " << cost;
            LOG(INFO) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
            LOG(INFO) << "------------------------------------------------------------------------------";
            break;
        }
        //  达到迭代次数
        if (iter == iterations - 1)
        {
            LOG(INFO) << "LM Have reach the maximum iteration num " << iter;
            break;
        }
        file << iter + 1 << "," << newcost << endl;
    }// for
    file.close();

}

/**
 * @brief Robust-Gauss-Newton
 */
void R_GN(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce)
{
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始Robust-Gauss-Newton迭代 : ";
    double cost = 0, lastCost = 0; // 本次迭代的Fx和上一次迭代的Fx
    // 开始rgn迭代
    ofstream file("./scripts/error-RGN.txt", ios::trunc);
    for (int iter = 0; iter < iterations; iter++)
    {
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        cost = 0;
        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i]; // xi & yi
            double error = 0;
            error = yi - exp(ae * xi * xi + be * xi + ce); // 使用上一步的估计值,计算 error (fx)
            double error2 = pow(error,2);
            double p1 = 1 / ( error2 + 1 );
            double p2 = - pow(p1,2);
            Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc
            Eigen::Matrix<double,1,3> JT = J.transpose();
            double w =  p1 + 2*p2*error2;
            if ( w >= 0 )       // 维度为 1,因此判断半正定只需判断符号
            {
                H += J * w * JT;
            }else{
                H += J * p1* JT;    // 负定时,使用一阶导近似
            }
            b += p1 * (-error * J) ;
            cost += error2;
        } // for
        // 求解线性方程 Hx=b
        Vector3d dx = H.ldlt().solve(b);

        if (isnan(dx[0]))
        {
            LOG(INFO) << "result is nan!";
            break;
        }
        // update
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        lastCost = cost;
        file << iter + 1 << "," << lastCost << endl;

        // 判断收敛
        if (iter > 0 &&
            sqrt(pow(dx[0], 2) + pow(dx[1], 2) + pow(dx[2], 2)) <=
            (sqrt(pow(ae, 2) + pow(be, 2) + pow(ce, 2)) + parameter_tolerance) * parameter_tolerance
                )
        {
            LOG(INFO) << "Approach to R-GN parameter converge, itera_count = " << iter + 1 << "\tcost = " << cost;
            LOG(INFO) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
            LOG(INFO) << "------------------------------------------------------------------------------";
            break;
        }
        // 达到最大迭代次数
        if (iter == iterations - 1)
        {
            LOG(INFO) << "R_GN Have reach the maximum iteration num " << iter;
            LOG(INFO) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
            LOG(INFO) << "---------------------------------------------------------";
            break;
        }
    } // for
    file.close();
}

/**
 * @brief LM_Nielsen
 */
void LM_Nielsen(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce)
{
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始 LM_Nielsen 迭代 : ";
    // 开始迭代
    double v = 2.0;
    double u = 0.0 ;
    double cost, newCost;  // 本次迭代的cost和新迭代的cost
    ofstream file("./scripts/error-LM-N.txt", ios::trunc);
    for ( int iter = 0 ; iter < iterations ; iter ++ )
    {
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        // 计算 H b
        cost = 0;
        for(int i = 0 ; i < N ; i ++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error= yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += J * J.transpose();
            b += - error * J;
            cost += error * error;
        }

        if (iter == 0)
            u = (1e-8)*max( H(0, 0), max(H(1, 1), H(2, 2)));

        //LOG(WARNING) << "u = " << u;
        H = H + ( u * Eigen::MatrixXd::Identity(3, 3));
        //  solver update
        Eigen::Vector3d dx = H.ldlt().solve(b);

        Eigen::Matrix<double,1,1> L;    // 分母
        L(0,0) = 0.5 * dx.transpose() * (u * dx + b);

        // try to update
        double new_ae = ae + dx[0];
        double new_be = be + dx[1];
        double new_ce = ce + dx[2];
        //LOG(WARNING) << "dx = " << dx[0] << "\t," << dx[1] << "\t," << dx[2];

        // 重新计算Fx
        newCost = 0;
        for(int i = 0 ; i < N ; i++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error = 0;
            error = yi - exp(new_ae * xi * xi + new_be * xi + new_ce);
            newCost += error * error;
        }

        // 没达到收敛条件则继续迭代
        //      Fx - newFx
        // p = -------------
        //        L(0,0)
        double p = (cost - newCost) / L(0,0) ;
//        LOG(INFO) << "p = " << p << "\tF = " << (cost - newCost) << "\tL = " << L;
        if (p <= 0){
            LOG(INFO) << "Refuse the update !!! ";
            u = v * u;
            v = 2*v;
            LOG(INFO) << "u = " << u;
        }else{
            u = u * max( 1.0/3.0, 1 - pow((2.0*p - 1), 3));
            v = 2.0;
            ae = new_ae;
            be = new_be;
            ce = new_ce;
//            LOG(WARNING) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
//            LOG(INFO) << "new cost = " << newCost;
        }

        // 判断收敛
        if (iter > 0 &&
            sqrt(pow(dx[0], 2) + pow(dx[1], 2) + pow(dx[2], 2)) <=
            (sqrt(pow(new_ae, 2) + pow(new_ce, 2) + pow(new_be, 2)) + parameter_tolerance) * parameter_tolerance
                )
        {
            LOG(INFO) << "Approach to LM-Nielsen parameter converge, iterations count = " << iter + 1 << "\tcost = " << cost;
            LOG(INFO) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
            LOG(INFO) << "------------------------------------------------------------------------------";
            break;
        }
        //  达到迭代次数
        if (iter == iterations - 1)
        {
            LOG(INFO) << "LM-Nielsen Have reach the maximum iteration num " << iter;
            break;
        }
        file << iter + 1 << "," << newCost << endl;
    }// for
    file.close();
}

/**
 * @brief Robust-LM
 */
void R_LM(const vector<double> &y_data, const vector<double> &x_data, int N, double ae, double be, double ce)
{
    LOG(INFO) << "------------------------------------------------------------------------------";
    LOG(INFO) << "开始 RLM 迭代 :";
    // 开始迭代
    double u = 0.0 ;
    ofstream file("./scripts/error-RLM.txt", ios::trunc);
    for ( int iter = 0 ; iter < iterations ; iter ++ )
    {
        // LOG(INFO) << "iter: " << iter;
        double cost = 0.0, newcost = 0.0;
        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero();             // bias
        // 计算 H b
        for(int i = 0 ; i < N ; i ++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error= yi - exp(ae * xi * xi + be * xi + ce);
            double error2 = pow(error, 2);
            double p1 = 1 / (error2 + 1);
            double p2 = -pow(p1, 2);

            Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            double w = p1 + 2 * p2 * error2;
            if (w >= 0)
                H += J * w * J.transpose();
            else
                H += J * p1 * J.transpose();

            b += p1 * (-error * J);
            cost += error2;
        }

        if (iter == 0)
            u = (1e-8) * max(H(0, 0), max(H(1, 1), H(2, 2)));

        // LOG(WARNING) << "u = " << u;
        H = H + ( u * Eigen::MatrixXd::Identity(3, 3));
        // solver update
        Eigen::Vector3d dx = H.ldlt().solve(b);

        // Matrix<double ,1,1> L ;
        Eigen::Matrix<double,1,1> L;    // 分母
        L(0,0) = 0.5 * dx.transpose() * (u * dx + b);

        // try to update
        double new_ae = ae + dx[0];
        double new_be = be + dx[1];
        double new_ce = ce + dx[2];

        // 重新计算cost
        newcost = 0;
        for(int i = 0 ; i < N ; i++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error = 0;
            error = yi - exp(new_ae * xi * xi + new_be * xi + new_ce);
            newcost += error * error;
        }

        // 没达到收敛条件则继续迭代
        //      cost - newcost
        // p = -------------
        //        L(0,0)
        double p = (cost - newcost) / L(0, 0) ;
        if (p < 0.25){
            u = 2 * u;
        } else if (p > 0.75) {
            u = u / 3;
        }

        // update parameter
        if (p > 0)
        {
            ae = new_ae;
            be = new_be;
            ce = new_ce;
        }

        // 判断收敛
        if (iter > 0 &&
            sqrt(pow(dx[0], 2) + pow(dx[1], 2) + pow(dx[2], 2)) <=
            (sqrt(pow(new_ae, 2) + pow(new_ce, 2) + pow(new_be, 2)) + parameter_tolerance) * parameter_tolerance
                )
        {
            LOG(INFO) << "Approach to R-LM parameter converge, iterations count = " << iter + 1 << "\tcost = " << cost;
            LOG(INFO) << "ae = " << ae << "\tbe = " << be << "\tce = " << ce;
            LOG(INFO) << "------------------------------------------------------------------------------";
            break;
        }
        //  达到迭代次数
        if (iter == iterations - 1)
        {
            LOG(INFO) << "LM Have reach the maximum iteration num " << iter;
            break;
        }
        file << iter + 1 << "," << newcost << endl;
    }// for
    file.close();
}