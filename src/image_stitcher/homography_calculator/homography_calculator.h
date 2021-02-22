#ifndef PANORAMA_VIEW_HOMOGRAPHY_CALCULATOR_H
#define PANORAMA_VIEW_HOMOGRAPHY_CALCULATOR_H

#include <Eigen/Dense>

#include "../image_stitcher.h"

namespace image_stitcher::homography_calculator {
  typedef double (*transformCost)(const Eigen::MatrixXf& pts1, const Eigen::MatrixXf& pts2);

  double MSE(const Eigen::MatrixXf& pts1, const Eigen::MatrixXf& pts2);

  Eigen::Matrix3f calcHomography(const Eigen::MatrixXf& src_pts, const Eigen::MatrixXf& dst_pts);

  Eigen::Matrix3f RANSAC(const Eigen::MatrixXf& src_pts, const Eigen::MatrixXf& dst_pts,
                         transformCost cost = MSE, std::size_t num_iter = 100);
}

#endif //PANORAMA_VIEW_HOMOGRAPHY_CALCULATOR_H
