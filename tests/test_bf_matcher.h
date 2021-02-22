#ifndef PANORAMA_VIEW_TEST_BF_MATCHER_H
#define PANORAMA_VIEW_TEST_BF_MATCHER_H

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "../src/image_stitcher/image_stitcher.h"
#include "../src/image_stitcher/bf_matcher/bf_matcher.h"

void compareNN(const image_stitcher::Matches& expected_nn, const image_stitcher::Matches& nn) {
  ASSERT_EQ(expected_nn.size(), nn.size());
  for (int i = 0; i < expected_nn.size(); ++i) {
    ASSERT_EQ(expected_nn[i].pt1, nn[i].pt1);
    ASSERT_EQ(expected_nn[i].pt2, nn[i].pt2);
    ASSERT_NEAR(expected_nn[i].distance, nn[i].distance, 1e-5);
  }
}

TEST(BFMatcher, kNN_SIMPLE_TEST) {
  Eigen::Matrix2f pt1, pt2;

  pt1 << 1, 2, 3, 4;
  pt2 << 5, 6, 7, 8;

  auto nn = image_stitcher::bf_matcher::kNN(pt1, pt2, 2);

  image_stitcher::Matches expected_nn{
      image_stitcher::Match{0, 0, 5.65685},
      image_stitcher::Match{0, 1, 8.48528},
      image_stitcher::Match{1, 0, 2.82843},
      image_stitcher::Match{1, 1, 5.65685},
  };

  compareNN(expected_nn, nn);
}

TEST(BFMatcher, kNN_ZEROS) {
  Eigen::Matrix2f pt1{Eigen::Matrix2f::Zero()}, pt2{Eigen::Matrix2f::Zero()};

  auto nn = image_stitcher::bf_matcher::kNN(pt1, pt2, 2);

  image_stitcher::Matches expected_nn{
      image_stitcher::Match{0, 0, 0},
      image_stitcher::Match{0, 1, 0},
      image_stitcher::Match{1, 0, 0},
      image_stitcher::Match{1, 1, 0},
  };

  compareNN(expected_nn, nn);
}

TEST(BFMatcher, kNN_ONES) {
  Eigen::Matrix2f pt1{Eigen::Matrix2f::Ones()}, pt2{Eigen::Matrix2f::Ones()};

  auto nn = image_stitcher::bf_matcher::kNN(pt1, pt2, 2);

  image_stitcher::Matches expected_nn{
      image_stitcher::Match{0, 0, 0},
      image_stitcher::Match{0, 1, 0},
      image_stitcher::Match{1, 0, 0},
      image_stitcher::Match{1, 1, 0},
  };

  compareNN(expected_nn, nn);
}
#endif //PANORAMA_VIEW_TEST_BF_MATCHER_H
