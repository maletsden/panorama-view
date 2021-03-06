#include "knn_finder.h"
#include "kd_tree.h"

std::vector<std::vector<image_stitcher::Match>> image_stitcher::knn_finder::bruteForce(
  const Eigen::MatrixXf& points1, const Eigen::MatrixXf& points2, std::size_t k
)
{
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

  // Find closest neighbours
  std::vector<std::vector<image_stitcher::Match>> nn;

  float *data = D.data();
  std::vector<std::size_t> indexes_range;
  indexes_range.reserve(N);
  for (std::size_t i = 0; i < N; ++i)
  {
    indexes_range.push_back(i);
  }

  for (std::size_t i = 0; i < K; ++i)
  {
    auto data_start = data + N * i;
    std::vector<std::size_t> indexes{indexes_range};

    // get sorted indexes of column values
    std::sort(
      indexes.begin(), indexes.end(),
      [&data_start](std::size_t i1, std::size_t i2)
      {
        return data_start[i1] < data_start[i2];
      }
    );

    // save only `k` best matches
    nn.emplace_back();
    for (std::size_t k_i = 0; k_i < std::min(k, N); ++k_i)
    {
      nn.back().emplace_back(i, indexes[k_i], D(indexes[k_i], i));
    }
  }

  return nn;
}

double squareDistance(const Eigen::VectorXf& point1, const Eigen::VectorXf& point2)
{
  return static_cast<double>((point1 - point2).norm());
}

std::vector<std::vector<image_stitcher::Match>> image_stitcher::knn_finder::randomKDTreeForest(
    const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, std::size_t k,
    std::size_t trees_num, std::size_t kd_tree_dims_num, float kd_tree_leaves_percent
)
{
  std::vector<std::vector<image_stitcher::Match>> nn;
  nn.reserve(points1.size());

  // create random decision forest
  std::vector<RandomKDTree> forest;
  forest.reserve(trees_num);
  for (std::size_t i = 0; i < trees_num; ++i)
  {
    forest.emplace_back(points2, k, kd_tree_dims_num, kd_tree_leaves_percent);
  }

  std::vector<image_stitcher::Match> matches;
  matches.reserve(trees_num * (points2.size() * kd_tree_leaves_percent));

  std::vector<std::size_t> knn_indexes;
  knn_indexes.reserve(trees_num * (points2.size() * kd_tree_leaves_percent));

  for (std::size_t i = 0; i < static_cast<std::size_t>(points1.rows()); ++i)
  {
    auto& point = points1.row(i);

    // get all possible knn points indexes
    knn_indexes.clear();
    for (const auto& kd_tree: forest)
    {
      auto knn_idx = kd_tree.findKNN(point);
      knn_indexes.insert(knn_indexes.end(), knn_idx.begin(), knn_idx.end());
    }

    // sort matches of possible knn points based on their distances
    matches.clear();
    for (auto idx: knn_indexes)
    {
      matches.emplace_back(i, idx, squareDistance(point, points2.row(idx)));
    }

    std::sort(matches.begin(), matches.end(), [] (const image_stitcher::Match& m1, const image_stitcher::Match& m2)
    {
      return m1.distance < m2.distance;
    });

    // save only `k` best matches
    nn.emplace_back();
    nn.back().reserve(std::min(k, matches.size()));

    for (std::size_t j = 0; j < std::min(k, matches.size()); ++j)
    {
      nn.back().emplace_back(matches[j]);
    }
  }

  return nn;
}
