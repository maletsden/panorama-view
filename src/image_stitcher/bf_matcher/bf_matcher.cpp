#include <iostream>
#include "bf_matcher.h"
#include <algorithm>

// TODO: run in parallel
image_stitcher::Matches image_stitcher::bf_matcher::kNN(
    const Eigen::MatrixXf& points1, const Eigen::MatrixXf& points2, std::size_t k
) {
  // Calculate distance matrix
  const std::size_t N = points2.rows();
  const std::size_t K = points1.rows();

  // Allocate parts of the expression
  Eigen::MatrixXf XX{N, 1}, YY{1, K}, XY{N, K}, D{N, K};

  // Compute norms
  XX = points2.array().square().rowwise().sum();
  YY = points1.array().square().rowwise().sum().transpose();
  XY = (2 * points2) * points1.transpose();

  // Compute final expression
  D = XX * Eigen::MatrixXf::Ones(1, K);
  D = D + Eigen::MatrixXf::Ones(N, 1) * YY;
  D = D - XY;
  D = D.cwiseSqrt();

  // Find closest neighbours
  image_stitcher::Matches nn;

  float *data = D.data();
  std::vector<std::size_t> indexes_range;
  indexes_range.reserve(K);
  for (std::size_t i = 0; i < K; ++i) {
    indexes_range.push_back(i);
  }

  for (std::size_t i = 0; i < K; ++i) {
    auto data_start = data + N * i;
    std::vector<std::size_t> indexes{indexes_range};

    // get sorted indexes of column values
    std::sort(
      indexes.begin(), indexes.end(),
      [data_start](std::size_t i1, std::size_t i2){ return data_start[i1] < data_start[i2]; }
    );

    // TODO: add check whether the point is not too far from the reference point
    for (std::size_t k_i = 0; k_i < std::min(k, N); ++k_i) {
      nn.emplace_back(i, indexes[k_i], D(indexes[k_i], i));
    }
  }

  return nn;
}