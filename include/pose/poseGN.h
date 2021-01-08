//
// Created by lu on 2021/1/7.
//

#ifndef CURVEFITTING_POSEGN_H
#define CURVEFITTING_POSEGN_H

#include "pose/initialize.h"

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &K,
        Sophus::SE3d &pose
);


#endif //CURVEFITTING_POSEGN_H
