#ifndef PANORAMA_VIEW_TEST_HOMOGRAPHY_CALCULATOR_H
#define PANORAMA_VIEW_TEST_HOMOGRAPHY_CALCULATOR_H
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "../src/image_stitcher/image_stitcher.h"

TEST(HomographyCalculator, HUBER_SIMPLE_TEST) {
  Eigen::MatrixXf source_pts{3, 4}, target_pts{3, 4};

  source_pts << 1, 3, 5, 7,
                2, 4, 6, 8,
                1, 1, 1, 1;
  target_pts << 2, 4, 6, 8,
                2, 4, 6, 8,
                1, 1, 1, 1;

  double error = image_stitcher::homography_calculator::Huber(source_pts, target_pts);

  // since standard deviation of distances is just 0
  double expected_error = 0.0;

  ASSERT_DOUBLE_EQ(expected_error, error);
}


TEST(HomographyCalculator, MSE_SIMPLE_TEST) {
  Eigen::MatrixXf source_pts{3, 4}, target_pts{3, 4};

  source_pts << 1, 3, 5, 7,
                2, 4, 6, 8,
                1, 1, 1, 1;
  target_pts << 2, 4, 6, 8,
                2, 4, 6, 8,
                1, 1, 1, 1;

  double error = image_stitcher::homography_calculator::MSE(source_pts, target_pts);
  double expected_error = 4.0;

  ASSERT_DOUBLE_EQ(expected_error, error);
}

TEST(HomographyCalculator, CALC_HOMOGRAPHY_SIMPLE_TEST) {
  Eigen::MatrixXf source_pts{3, 4}, target_pts{3, 4};

  source_pts << 1, 1, 3, 3,
                4, 2, 4, 2,
                1, 1, 1, 1;

  // shift by 4 along Ox and by 2 along Oy
  target_pts << 5, 5, 7, 7,
                6, 4, 6, 4,
                1, 1, 1, 1;

  auto homography = image_stitcher::homography_calculator::calcHomography(source_pts, target_pts);

  Eigen::Matrix3f expected_homography;
  expected_homography << 1, 0, 4,
                         0, 1, 2,
                         0, 0, 1;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(expected_homography(i, j), homography(i, j), 1e-5);
    }
  }
}

TEST(HomographyCalculator, RANSAC_SIMPLE_TEST) {
  std::pair<Eigen::MatrixXf, Eigen::MatrixXf> points{
      Eigen::MatrixXf{3, 8}, Eigen::MatrixXf{3, 8}
  };
  std::get<0>(points) << 1, 1, 3, 3, 5, 5, 7, 7,
                            4, 2, 4, 2, 4, 2, 4, 2,
                            1, 1, 1, 1, 1, 1, 1, 1;

  // shift by 4 along Ox and by 2 along Oy
  std::get<1>(points) << 5, 5, 7, 7, 9, 9, 11, 11,
                            6, 4, 6, 4, 6, 4,  6,  4,
                            1, 1, 1, 1, 1, 1,  1,  1;
  auto homography = image_stitcher::homography_calculator::RANSAC(
    points, image_stitcher::homography_calculator::MSE, 5
  );

  Eigen::Matrix3f expected_homography;
  expected_homography << 1, 0, 4,
                         0, 1, 2,
                         0, 0, 1;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(expected_homography(i, j), homography(i, j), 1e-5);
    }
  }
}

TEST(HomographyCalculator, RANSAC_NONPARALLEL_TEST) {
  std::pair<Eigen::MatrixXf, Eigen::MatrixXf> points{
      Eigen::MatrixXf{3, 4}, Eigen::MatrixXf{3, 4}
  };
  std::get<0>(points) << 0, 0, 4, 4,
                            0, 4, 0, 4,
                            1, 1, 1, 1;

  std::get<1>(points) << 0, 1, 4, 3,
                            0, 4, 0, 4,
                            1, 1, 1, 1;

  auto homography = image_stitcher::homography_calculator::RANSAC(
      points, image_stitcher::homography_calculator::MSE, 5
  );

  Eigen::Matrix3f expected_homography;
  expected_homography << 1,  0.5, 0,
                         0,    2, 0,
                         0, 0.25, 1;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(expected_homography(i, j), homography(i, j), 1e-6);
    }
  }
}

#endif //PANORAMA_VIEW_TEST_HOMOGRAPHY_CALCULATOR_H
