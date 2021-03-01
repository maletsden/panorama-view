#ifndef PANORAMA_VIEW_TEST_KNN_FINDER_H
#define PANORAMA_VIEW_TEST_KNN_FINDER_H

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "../src/image_stitcher/image_stitcher.h"

void compareNN(const image_stitcher::Matches& expected_nn, const image_stitcher::Matches& nn) {
  ASSERT_EQ(expected_nn.size(), nn.size());
  for (int i = 0; i < expected_nn.size(); ++i) {
    ASSERT_EQ(expected_nn[i].pt1, nn[i].pt1);
    ASSERT_EQ(expected_nn[i].pt2, nn[i].pt2);
    ASSERT_NEAR(expected_nn[i].distance, nn[i].distance, 1e-5);
  }
}

TEST(kNNFinder, BF_SIMPLE_TEST) {
  Eigen::Matrix2f pt1, pt2;

  pt1 << 1, 2,
         3, 4;
  pt2 << 5, 6,
         7, 8;

  auto nn = image_stitcher::knn_finder::bruteForce(pt1, pt2, 2);

  image_stitcher::Matches expected_nn{
      image_stitcher::Match{0, 0, 32},
      image_stitcher::Match{0, 1, 72},
      image_stitcher::Match{1, 0, 8},
      image_stitcher::Match{1, 1, 32},
  };

  compareNN(expected_nn, nn);
}

TEST(kNNFinder, BF_ZEROS) {
  Eigen::Matrix2f pt1{Eigen::Matrix2f::Zero()}, pt2{Eigen::Matrix2f::Zero()};

  auto nn = image_stitcher::knn_finder::bruteForce(pt1, pt2, 2);

  image_stitcher::Matches expected_nn{
      image_stitcher::Match{0, 0, 0},
      image_stitcher::Match{0, 1, 0},
      image_stitcher::Match{1, 0, 0},
      image_stitcher::Match{1, 1, 0},
  };

  compareNN(expected_nn, nn);
}

TEST(kNNFinder, BF_ONES) {
  Eigen::Matrix2f pt1{Eigen::Matrix2f::Ones()}, pt2{Eigen::Matrix2f::Ones()};

  auto nn = image_stitcher::knn_finder::bruteForce(pt1, pt2, 2);

  image_stitcher::Matches expected_nn{
      image_stitcher::Match{0, 0, 0},
      image_stitcher::Match{0, 1, 0},
      image_stitcher::Match{1, 0, 0},
      image_stitcher::Match{1, 1, 0},
  };

  compareNN(expected_nn, nn);
}

//TEST(kNNFinder, KDTREE_SIMPLE_TEST) {
//  Eigen::Matrix2f pt1, pt2;
//
//  pt1 << 0, 4,
//         4, 4;
//  pt2 << 0, 4,
//         0, 0;
//
//  auto nn = image_stitcher::knn_finder::kDTree(pt1, pt2, 2);
//
//  image_stitcher::Matches expected_nn{
//      image_stitcher::Match{0, 0, 4},
//      image_stitcher::Match{0, 1, 2},
//      image_stitcher::Match{1, 0, 2},
//      image_stitcher::Match{1, 1, 4},
//  };
//
//  compareNN(expected_nn, nn);
//}
#endif //PANORAMA_VIEW_TEST_KNN_FINDER_H
