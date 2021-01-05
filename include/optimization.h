//
// Created by lu on 2021/1/5.
//

#ifndef CURVEFITTING_OPTIMIZATION_H
#define CURVEFITTING_OPTIMIZATION_H

#include "commom_include.h"


//const double function_tolerance = 1e-5;
//const double max_trans_epsilon = 1e-5;
const double parameter_tolerance = 1e-5;
const int iterations = 1000;

void GN(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);
void LM(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);
void R_GN(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);
void LM_Nielsen(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);
void R_LM(const std::vector<double> &y_data, const std::vector<double> &x_data, int N, double ae, double be, double ce);
double getRand();


#endif //CURVEFITTING_OPTIMIZATION_H
