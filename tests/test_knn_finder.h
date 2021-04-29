#ifndef PANORAMA_VIEW_TEST_KNN_FINDER_H
#define PANORAMA_VIEW_TEST_KNN_FINDER_H

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "../src/image_stitcher/image_stitcher.h"

void compareNN(
  const std::vector<std::vector<image_stitcher::Match>>& expected_nn,
  const std::vector<std::vector<image_stitcher::Match>>& nn
)
{
  ASSERT_EQ(expected_nn.size(), nn.size());
  for (std::size_t i = 0; i < expected_nn.size(); ++i) {
    for (std::size_t j = 0; j < expected_nn[i].size(); ++j) {
      ASSERT_EQ(expected_nn[i][j].pt1, nn[i][j].pt1);
      ASSERT_EQ(expected_nn[i][j].pt2, nn[i][j].pt2);
      ASSERT_NEAR(expected_nn[i][j].distance, nn[i][j].distance, 1e-5);
    }
  }
}

TEST(kNNFinder, BF_SIMPLE_TEST) {
  Eigen::Matrix2f pt1, pt2;

  pt1 << 1, 2,
         3, 4;
  pt2 << 5, 6,
         7, 8;

  auto nn = image_stitcher::knn_finder::bruteForce(pt1, pt2, 2);

  std::vector<std::vector<image_stitcher::Match>> expected_nn {
    {
      image_stitcher::Match{0, 0, 32},
      image_stitcher::Match{0, 1, 72},
    },
    {
      image_stitcher::Match{1, 0, 8},
      image_stitcher::Match{1, 1, 32},
    }
  };

  compareNN(expected_nn, nn);
}

TEST(kNNFinder, BF_ZEROS) {
  Eigen::Matrix2f pt1{Eigen::Matrix2f::Zero()}, pt2{Eigen::Matrix2f::Zero()};

  auto nn = image_stitcher::knn_finder::bruteForce(pt1, pt2, 2);

  std::vector<std::vector<image_stitcher::Match>> expected_nn {
    {
      image_stitcher::Match{0, 0, 0},
      image_stitcher::Match{0, 1, 0},
    },
    {
      image_stitcher::Match{1, 0, 0},
      image_stitcher::Match{1, 1, 0},
    }
  };


  compareNN(expected_nn, nn);
}

TEST(kNNFinder, BF_ONES) {
  Eigen::Matrix2f pt1{Eigen::Matrix2f::Ones()}, pt2{Eigen::Matrix2f::Ones()};

  auto nn = image_stitcher::knn_finder::bruteForce(pt1, pt2, 2);

  std::vector<std::vector<image_stitcher::Match>> expected_nn {
    {
      image_stitcher::Match{0, 0, 0},
      image_stitcher::Match{0, 1, 0},
    },
    {
      image_stitcher::Match{1, 0, 0},
      image_stitcher::Match{1, 1, 0},
    }
  };

  compareNN(expected_nn, nn);
}
std::ostream& operator<<(std::ostream& os, const image_stitcher::Match& m) {
  os << "Match<" << m.pt1 << ", " << m.pt2 << ", " << m.distance << ">";

  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
  os << '[';
  for (int i = 0; i < static_cast<int>(vector.size()) - 1; ++i) {
//  for (int i = 0; i < 20; ++i) {
    os << vector[i] << ", ";
  }
  os << vector.back() << ']';

  return os;
}
TEST(kNNFinder, KDTREE_SIMPLE_TEST) {
  Eigen::Matrix2f pt1, pt2;

  pt1 << 0, 4,
         4, 4;
  pt2 << 0, 4,
         0, 0;

  auto nn = image_stitcher::knn_finder::randomKDTreeForest(pt1, pt2, 2);

  std::vector<std::vector<image_stitcher::Match>> expected_nn {
      {
          image_stitcher::Match{0, 0, 0},
          image_stitcher::Match{0, 1, 16},
      },
      {
          image_stitcher::Match{1, 0, 16},
          image_stitcher::Match{1, 1, 32},
      }
  };


  compareNN(expected_nn, nn);
}
#endif //PANORAMA_VIEW_TEST_KNN_FINDER_H
