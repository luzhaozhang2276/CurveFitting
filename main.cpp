#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

// #define random(x)  rand( x)/RAND_MAX

using namespace std;
using namespace Eigen;

void GN(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be);
void R_GN(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be);
void LM_Nielsen(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be);
void R_LM(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be);
double getRand();

const double function_tolerance = 1e-5;
const double max_trans_epsilon = 1e-5;
const double parameter_tolerance = 1e-5;
const int iterations = 20;

int main(int argc, char **argv)
{
    double ar = 2.0, br = 0.0;  // 真实参数值
    double ae = -8, be = -9.0 ;   // 初始估计参数值
    int N = 1000 ;                                // 数据点

    // 测试数据
    vector<double> x_data, y_data;
    for (int i = 0; i < N;  i++ )
    {
        x_data.push_back(double(i));
        y_data.push_back( 2 *x_data[i]  + 0.001 * getRand() );
    }

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<< std::endl;
    std::cout << "给定-" << N << "-组点对" << " -数据带有[-0.01,0.01]均匀分布的噪声"<< std::endl;
    std::cout << "真实参数值：" <<"ar = "  << ar << " " << "br = "  << br  << std::endl;
    std::cout << "初始估计值：" <<"ae = " << ae << " " << "be = " << be  << std::endl;
    std::cout << std::endl;
    std::cout << "数据状况良好的情况下：" << std::endl;

    GN( y_data, x_data, N, ae, be) ;
    LM_Nielsen(y_data, x_data, N, ae, be);
    R_GN(y_data, x_data, N, ae, be);
    R_LM(y_data,x_data,N, ae ,be);

    // outliers
    y_data[988] = 0.00;
    y_data[998] = 0.00;

    std::cout << std::endl;
    std::cout << "手动 增加 outliers : "<<std::endl;
    std::cout << "令　y[988] = 0.0 ，y[998] = 0.0  　再次求解: " << std::endl;
    GN(y_data, x_data, N, ae, be);
    LM_Nielsen(y_data, x_data, N, ae, be);
    R_GN(y_data, x_data, N, ae, be);
    R_LM(y_data, x_data, N, ae, be);
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    return 0;
}

//  得到 0 -1 之间的随机数
double getRand()
{
    const int n = 99;
    srand( time(NULL));
    return 2.0* ( rand()%(n+1) / (double) (n+1) ) -1.0 ;
}

/*
** Guassian-Newton
**
*/
void GN ( const vector<double>& y_data , const vector<double>& x_data , const int N ,double ae ,double be )
{
    // 开始Gauss-Newton迭代
    std::cout << std::endl;
    std::cout << " 开始Gauss-Newton迭代 : " << std::endl;

    double Fx = 0, lastFx = 0; // 本次迭代的Fx和上一次迭代的Fx

    for ( int iter = 0; iter < iterations; iter++ )
    {
        Matrix2d H = Matrix2d::Zero();
        Vector2d b = Vector2d::Zero();
        Fx = 0;

        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i];
            double error = 0;
            error = yi - (ae * xi + be);
            Vector2d J;
            J[0] = double(-xi);
            J[1] = double(-1);
            H +=  J * J.transpose() ;
            b += -error * J  ;
            Fx += error * error;
        } // for

        // 求解线性方程 Hx=b
        Vector2d dx = H.ldlt().solve(b);
        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }
        // update
        ae += dx[0];
        be += dx[1];
        lastFx = Fx;
        Fx =0;
        // Calculation
        for (int i = 0; i < N; i++)
        {
            double error = 0;
            double xi = x_data[i], yi = y_data[i];
            error = yi - (ae * xi + be);
            Fx += error * error;

        } // for
        std::cout << " ae = "<< ae << "   " << "be = " <<  be  <<  std::endl;
        //
        // if (fabs((Fx - lastFx) / lastFx) < function_tolerance)
        // {
        //     std::cout << "Approach to GN Fx converge, itera_count = " << iter + 1 << std::endl;
        //     // std::cout << " ae = "<< ae << "   " << "be = " <<  be  <<  std::endl;
        //     std::cout << "---------------------------------------------------------" << std::endl;
        //     break;
        // }
        //
        if ( sqrt(pow(dx[0], 2) + pow(dx[1], 2)) <= (sqrt(pow(ae, 2) + pow(be, 2)) + parameter_tolerance) * parameter_tolerance)
        {
            // std::cout << "+1" << std::endl;
            std::cout << "Approach to GN parameter converge, itera_count = " << iter + 1 << std::endl;
            std::cout << "---------------------------------------------------------" << std::endl;
            break;
        }
        // 达到最大迭代次数
        if (iter == iterations - 1)
        {
            std::cout << "GN Have reach the maximum iteration num " << iter << std::endl;
            break;
        }
        lastFx = Fx;
    } // for
}

/*
**  Robustied-Guassian-Newton
**
*/
void R_GN(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be)
{
    std::cout << std::endl;
    std::cout << " 开始Robust-Gauss-Newton迭代 : " << std::endl;
    double Fx = 0, lastFx = 0; // 本次迭代的Fx和上一次迭代的Fx
    // 开始rgn迭代
    for (int iter = 0; iter < iterations; iter++)
    {
        Matrix2d H = Matrix2d::Zero();
        Vector2d b = Vector2d::Zero();
        Fx = 0;
        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i]; // xi & yi
            double error = 0;
            error = yi - (ae * xi + be); // 使用上一步的估计值,计算 error (fx)
            double error2 = pow(error,2);
            double p1 = 1 / ( error2 + 1 );
            double p2 = - pow(p1,2);
            Vector2d J;                  //
            J[0] = double(-xi);
            J[1] = double(-1);
            Eigen::Matrix<double,1,2> JT = J.transpose();
            double w =  p1 + 2*p2*error2;
            if ( w >= 0 )
            {
                H += J * w * JT;
            }else{
                H += J * p1* JT;
            }
            b += p1 * (-error * J) ;
            Fx += error2;
        } // for
        // 求解线性方程 Hx=b
        Vector2d dx = H.ldlt().solve(b);

        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }
        // update
        ae += dx[0];
        be += dx[1];
        lastFx = Fx;
        Fx = 0;
        // reCalculation
        for (int i = 0; i < N; i++)
        {
            double error = 0;
            double xi = x_data[i], yi = y_data[i];
            error = yi - (ae * xi + be);
            Fx += error * error;
        } // for
        std::cout << " ae = " << ae << "   " << "be = " << be << std::endl;
        lastFx = Fx;

        // if (fabs((Fx - lastFx) / lastFx) < function_tolerance)
        // {
        //     std::cout << "Approach to R-GN Fx converge, itera_count = " << iter + 1 << std::endl;
        //     // std::cout << " ae = "<< ae << "   " << "be = " <<  be  <<  std::endl;
        //     std::cout << "---------------------------------------------------------" << std::endl;
        //     break;
        // }
        //
        if (iter >=0 && sqrt(pow(dx[0], 2) + pow(dx[1], 2)) <= (sqrt(pow(ae, 2) + pow(be, 2)) + parameter_tolerance) * parameter_tolerance)
        {
            std::cout << "Approach to R-GN parameter converge, itera_count = " << iter + 1 << std::endl;
            std::cout << "---------------------------------------------------------" << std::endl;
            break;
        }
        // 达到最大迭代次数
        if (iter == iterations - 1)
        {
            std::cout << "R_GN Have reach the maximum iteration num " << iter << std::endl;
            break;
        }

    } // for
    cout << "estimated ab = " << ae << ", " << be << endl;
}

/*
    LM-Nielson
*/
void LM_Nielsen(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be)
{
    std::cout << std::endl;
    std::cout << " 开始 LM_Nielsen 迭代 : " << std::endl;
    // 开始迭代
    double v = 2.0;
    double u = 0.0 ;
    // double p = 0.0;
    for ( int iter = 0 ; iter < iterations ; iter ++ )
    {
        double Fx = 0.0, newFx = 0.0;
        Matrix2d H = Matrix2d::Zero();
        Vector2d b = Vector2d::Zero();
        // 计算 H b
        for(int i = 0 ; i < N ; i ++ )
        {
            double xi = x_data[i];
            double yi = y_data[i];
            Vector2d J;
            J[0] = double(-xi);
            J[1] = double(-1);
            Eigen::Matrix<double, 2, 2> JTJ = J * J.transpose();    // JTJ
            double error = 0.0; //  残差函数f
            error = yi - (ae * xi + be);
            H += JTJ ;
            b += -error * J;
            // 计算损失函数
            Fx += error*error;
        }

        if (iter == 0)
            u = (1e-8)*max( H(0, 0), H(1, 1) );
        // u = (1) * abs(max(H(0, 0), H(1, 1)));

        H = H + ( u * Eigen::MatrixXd::Identity(2, 2));
        //  solver update
        Eigen::Vector2d dx = H.ldlt().solve(b);

        // Matrix<double ,1,1> L ;
        Eigen::Matrix<double,1,1> L;
        L(0,0) = 0.5 * dx.transpose() * (u * dx + b);

        // try to update
        double new_ae = ae + dx[0];
        double new_be = be + dx[1];


        // 重新计算Fx

        for(int i = 0 ; i < N ; i++ )
        {
            double xi = x_data[i], yi = y_data[i];
            double error = 0;
            error = yi - (new_ae * xi + new_be);
            newFx += error * error;
        }

        // 没达到收敛条件则继续迭代
        double p = (Fx - newFx) / L(0,0) ;
        // std::cout << Fx - newFx << std::endl;
        // std::cout << L(0,0) << std::endl;
        if (p <= 0){
            std::cout << "Refuse the update !!! " << std::endl;
            u = v * u;
            v = 2*v;
        }else{
            u = u * max( 1.0/3.0, 1 - pow((2.0*p - 1), 3));
            v = 2.0;
            ae = new_ae;
            be = new_be;
            // Fx = newFx;
            std::cout << " ae = " << ae << "   " << "be = " << be << std::endl;
        }

        // 判断收敛 1
        if (iter > 0 && sqrt(pow(dx[0], 2) + pow(dx[1], 2)) <= (sqrt(pow(new_ae, 2) + pow(new_be, 2)) + parameter_tolerance) * parameter_tolerance)
        {
            std::cout << "Approach to LM parameter converge, itera_count = " << iter + 1 << std::endl;
            std::cout << " ae = " << ae << "   "
                      << "be = " << be << std::endl;
            std::cout << "---------------------------------------------------------" << std::endl;
            break;
        }
        //  达到迭代次数
        if (iter == iterations - 1)
        {
            std::cout << "LM Have reach the maximum iteration num " << iter << std::endl;
            break;
        }
    }// for
}

/*
**  Robustied-LM-Nielsen
*/
void R_LM(const vector<double> &y_data, const vector<double> &x_data, const int N, double ae, double be)
{
    std::cout << std::endl;
    std::cout << " 开始Robust-LM 迭代 : " << std::endl;
    double Fx = 0, lastFx = 0; // 本次迭代的Fx和上一次迭代的Fx
    // 开始rgn迭代
    double u = 0.0;
    double v = 2.0;
    for (int iter = 0; iter < iterations; iter++)
    {
        Matrix2d H = Matrix2d::Zero();
        Vector2d b = Vector2d::Zero();
        Fx = 0;
        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i]; // xi & yi
            double error = 0;
            error = yi - (ae * xi + be); // 使用上一步的估计值,计算 error (fx)
            double error2 = pow(error, 2);
            double p1 = 1 / (error2 + 1);
            double p2 = -pow(p1, 2);
            Vector2d J; //
            J[0] = double(-xi);
            J[1] = double(-1);
            Eigen::Matrix<double, 1, 2> JT = J.transpose();
            double w = p1 + 2 * p2 * error2;
            if (w >= 0)
            {
                H += J * w * JT;
            }
            else
            {
                H += J * p1 * JT;
            }
            b += p1 * (-error * J);
            Fx += error2;
        } // for

        if (iter == 0 )
        {
            u = (1e-4)*std::max(H(0,0),H(1,1));
        }
        H = H + u * Eigen::Matrix2d::Identity(2,2);

        // 求解线性方程 Hx=b
        Vector2d dx = H.ldlt().solve(b);

        double new_ae = ae + dx[0];
        double new_be = be + dx[1];

        std::cout << "dx = " << dx[0] <<"," << dx[1] << std::endl;

        Vector2d diff = H * dx - b;

        //  add a diff  converge
        std::cout << "diff = " << diff[0] << " , " << diff[1] << std::endl;
        if ( fabs( diff[0] ) <= 1e-30 &&  fabs(diff[1])<= 1e-30 )
        {
            std::cout << "Approach to R-LM diff converge, iter_count = " << iter + 1 << std::endl;
            std::cout << "  ae = " << new_ae << "   " << "be = " << new_be << std::endl;
            break;
        }

        Eigen::Matrix<double,1,1> L ;
        L(0, 0) = 0.5 * dx.transpose() * (u * dx + b);
        lastFx = Fx;
        Fx = 0;
        // reCalculation
        for (int i = 0; i < N; i++)
        {
            double error = 0;
            double xi = x_data[i], yi = y_data[i];
            error = yi - (new_ae * xi + new_be);
            Fx += error * error;
        } // for

        double p = (lastFx - Fx)/L(0,0);

        if (p > 0)
        {
            u = u * max((1.0 / 3.0), 1 - pow((2.0 * p - 1), 3));
            v = 2.0;
            ae = new_ae;
            be = new_be;
            std::cout << "Consent  the update !!! " << ", lastFx-Fx = " << lastFx - Fx << std::endl;
            std::cout << "  ae = " << ae << "   "<< "be = " << be << std::endl;
            std::cout << std::endl;
        }
        if ( p <= 0)
        {
            std::cout << "  new_ae = " << new_ae << "   "<< "new_be = " << new_be << std::endl;
            std::cout << "Refuse the update !!! "<< ", lastFx-Fx = " << lastFx - Fx << std::endl;
            std::cout << "  ae = " << ae << "   "<< "be = " << be << std::endl;
            std::cout << std::endl;
            u = v * u;
            v = 2 * v;
        }


        // std::cout << b[0] << "," <<b[1] << std::endl;
        //  收敛条件1
        if (iter >= 0 && sqrt(pow(dx[0], 2) + pow(dx[1], 2)) <= (sqrt(pow(ae, 2) + pow(be, 2)) + parameter_tolerance) * parameter_tolerance)
        {
            std::cout << "Approach to R-LM parameter converge, itera_count = " << iter + 1 << std::endl;
            cout << "estimated ab = " << ae << ", " << be << endl;
            break;
        }
        //  收敛条件2
        // if ( iter >= 0 && fabs(b[0]) <= 1e-2 && fabs(b[1]) <= 1e-2)
        // {
        //     std::cout << "Approach to R-LM diff converge, iter_count = " << iter + 1 << std::endl;
        // }
        // 达到最大迭代次数
        if (iter == iterations - 1)
        {
            std::cout << "R_LM Have reach the maximum iteration num " << iter << std::endl;
            break;
        }
    } // for
    std::cout << "Find  the x*" << std::endl;
}