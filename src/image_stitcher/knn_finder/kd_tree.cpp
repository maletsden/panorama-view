#include <iostream>
#include "kd_tree.h"


RandomKDTree::RandomKDTree(
  const Eigen::MatrixXf& points_, std::size_t k_,
  std::size_t kd_tree_dims_num, float kd_tree_leaves_percent
):  points(points_), k(k_)
{
  std::vector<std::size_t> indexes;
  indexes.reserve(points_.rows());
  for (long i = 0; i < points_.rows(); ++i)
  {
    indexes.push_back(i);
  }

  // sort dimension indexes by its variance
  std::vector<std::size_t> dimensions_indexes;
  dimensions_indexes.reserve(points_.cols());
  for (std::size_t i = 0; i < static_cast<std::size_t>(points_.cols()); ++i)
  {
    dimensions_indexes.push_back(i);
  }

  Eigen::VectorXf dimension_var = (
    (points_.rowwise() - points_.colwise().mean()).colwise().norm()) / static_cast<double>(points_.rows()
  );

  std::sort(dimensions_indexes.begin(), dimensions_indexes.end(), [&dimension_var](std::size_t i, std::size_t j)
  {
    return dimension_var(i) > dimension_var(j);
  });

  // get just first `kd_tree_dims_num` dimension indexes
  dimensions_idx = std::vector<std::size_t>{
      dimensions_indexes.begin(),
      dimensions_indexes.begin() + std::min(dimensions_indexes.size(), kd_tree_dims_num)
  };

  // pad dimensions_idx with these 'top variance' dimensions
  auto max_tree_height = std::log2(points_.rows() / (points_.rows() * kd_tree_leaves_percent)) + 1;
  for (std::size_t i = 0; i < max_tree_height - dimensions_idx.size(); ++i)
  {
    dimensions_idx.push_back(dimensions_idx[i % dimensions_idx.size()]);
  }

  std::shuffle(dimensions_idx.begin(), dimensions_idx.end(), std::mt19937(std::random_device()()));

  tree = std::make_unique<KDTreeNode>(std::move(indexes));

  // start constructing the tree
  construct(tree.get());
}

void RandomKDTree::construct(KDTreeNode* subtree, std::size_t dimension_i) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::bernoulli_distribution distrib(0.5);

  auto& indexes = subtree->m_points_indexes;

  if (indexes.size() <= std::max(k * 2 - 1, static_cast<std::size_t>(points.rows() * 0.02)))
  {
    subtree->m_is_leaf = true;
    return;
  }

  const std::size_t dim = dimensions_idx[dimension_i % dimensions_idx.size()];
  std::sort(indexes.begin(), indexes.end(), [&](std::size_t a, std::size_t b)
  {
    return points(a, dim) < points(b, dim);
  });

  // split into random parts to add additional randomization
  float split_portion = distrib(gen) ? .33 : .75;

  // due to http://www.cs.cmu.edu/~agray/approxnn.pdf random splits and
  // additional overlap between cells (in our case 5% of cell size) leads to
  // better performance and lower error rate
  auto left_end = indexes.begin() + (indexes.size() * split_portion);
  std::vector<std::size_t> left_idx{indexes.begin(), std::min(left_end + 0.05 * indexes.size(), indexes.end())},
      right_idx{std::max(left_end - 0.05 * indexes.size(), indexes.begin()), indexes.end()};

  subtree->value = points(*left_end, dim);
  subtree->left = std::make_unique<KDTreeNode>(std::move(left_idx));
  subtree->right = std::make_unique<KDTreeNode>(std::move(right_idx));
  construct(subtree->left.get(), dimension_i + 1);
  construct(subtree->right.get(), dimension_i + 1);
}

std::vector<std::size_t> RandomKDTree::findKNN(const Eigen::VectorXf& point) const {
  auto subtree = tree.get();
  std::size_t dimension_i = 0;

  while (true) {
    if (subtree->m_is_leaf)
    {
      return subtree->m_points_indexes;
    }

    if (point(dimensions_idx[dimension_i++ % dimensions_idx.size()]) > subtree->value)
    {
      subtree = subtree->right.get();
    }
    else
    {
      subtree = subtree->left.get();
    }
  }
}
