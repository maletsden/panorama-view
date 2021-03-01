#ifndef PANORAMA_VIEW_KNN_FINDER_H
#define PANORAMA_VIEW_KNN_FINDER_H

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "../image_stitcher.h"

namespace image_stitcher::knn_finder {
  image_stitcher::Matches bruteForce(const Eigen::MatrixXf& points1, const Eigen::MatrixXf& points2, std::size_t k = 1);
  image_stitcher::Matches kDTree(const Eigen::MatrixXf& points1, const Eigen::MatrixXf& points2, std::size_t k = 1);
}

#endif //PANORAMA_VIEW_KNN_FINDER_H
