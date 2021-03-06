#ifndef PANORAMA_VIEW_KNN_FINDER_H
#define PANORAMA_VIEW_KNN_FINDER_H

#include <cstdlib>
#include <vector>
#include <set>
#include <algorithm>
#include <Eigen/Dense>
#include "../image_stitcher.h"

namespace image_stitcher::knn_finder
{
  /**
   * Find the k nearest neighbours for each source point
   * using brute-force algorithm
   *
   * @param pts1 - set of source points
   * @param pts2 - set of destination points
   * @param k - amount of kNN that need to be found for each src point
   *
   * @return - vector of best matches (of size k) for each point
   */
  std::vector<std::vector<image_stitcher::Match>> bruteForce(
    const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, std::size_t k = 1
  );

  /**
   * Find the approximate k nearest neighbours for each source point
   * using a forest consisting of `trees_num` KD trees
   *
   * @param pts1 - set of source points
   * @param pts2 - set of destination points
   * @param k - amount of kNN that need to be found for each src point
   * @param trees_num - the number of KD trees that need to used
   * @param kd_tree_dims_num - the number of dimensions with top largest variance that will be used in each tree
   * @param kd_tree_leaves_percent - the amount of points that will be left on leaves of KD tree
   *                                 (in percentage of total destination points number)
   *
   * @return - vector of best matches (of size k) for each point
   */
  std::vector<std::vector<image_stitcher::Match>> randomKDTreeForest(
    const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, std::size_t k = 1,
    std::size_t trees_num = 5, std::size_t kd_tree_dims_num = 5, float kd_tree_leaves_percent = 0.02
  );
}

#endif //PANORAMA_VIEW_KNN_FINDER_H
