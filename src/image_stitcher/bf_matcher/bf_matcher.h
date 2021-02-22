#ifndef PANORAMA_VIEW_BF_MATCHER_H
#define PANORAMA_VIEW_BF_MATCHER_H

#include <cstdlib>
#include <vector>
#include <Eigen/Dense>
#include "../image_stitcher.h"

namespace image_stitcher::bf_matcher {
  image_stitcher::Matches kNN(const Eigen::MatrixXf& points1, const Eigen::MatrixXf& points2, std::size_t k = 2);

  // TODO: add k-d tree
}

#endif //PANORAMA_VIEW_BF_MATCHER_H
