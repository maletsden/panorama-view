#ifndef PANORAMA_VIEW_HOMOGRAPHY_CALCULATOR_H
#define PANORAMA_VIEW_HOMOGRAPHY_CALCULATOR_H

#include <Eigen/Dense>
#include <random>

#include "../image_stitcher.h"

namespace image_stitcher::homography_calculator {
  /**
   * Cost function for transformation from source points to destination points
   *
   * @param pts1 - set of source points
   * @param pts2 - set of destination points
   *
   * @return - error between src and dest points
   */
  typedef double (*transformCost)(const Eigen::MatrixXf& pts1, const Eigen::MatrixXf& pts2);

  /**
   * Mean Square Error
   *
   * @param pts1 - set of source points
   * @param pts2 - set of destination points
   *
   * @return - error between src and dest points
   */
  double MSE(const Eigen::MatrixXf& pts1, const Eigen::MatrixXf& pts2);

  /**
   * Huber loss
   * (take into account that the data can be polluted with outliers and give them smaller error score)
   *
   * @param pts1 - set of source points
   * @param pts2 - set of destination points
   *
   * @return - error between src and dest points
   */
  double Huber(const Eigen::MatrixXf& pts1, const Eigen::MatrixXf& pts2);

  /**
   * Calculates 3x3 homography transformation matrix
   * (using QR solver)
   *
   * @param src_pts - set of source points
   * @param dst_pts - set of destination points
   *
   * @return - 3x3 homography transformation matrix
   */
  Eigen::Matrix3f calcHomography(const Eigen::MatrixXf& src_pts, const Eigen::MatrixXf& dst_pts);

  /**
   * Runs RANSAC algorithm to find homography matrix for given pairs of points
   *
   * @param points_pair - pair of source and destination points
   * @param cost - transformation cost function
   * @param num_iter - num of iterations
   *
   * @return - 3x3 homography transformation matrix
   */
  Eigen::Matrix3f RANSAC(const std::pair<Eigen::MatrixXf, Eigen::MatrixXf>& points_pair,
                         transformCost cost = MSE, std::size_t num_iter = 100);
}

#endif //PANORAMA_VIEW_HOMOGRAPHY_CALCULATOR_H
