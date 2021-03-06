#ifndef PANORAMA_VIEW_TEST_IMAGE_TRANSFORMATION_H
#define PANORAMA_VIEW_TEST_IMAGE_TRANSFORMATION_H

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "../src/image_stitcher/image_stitcher.h"

TEST(IMAGE_TRANSFORMATION, APPLY_HOMOGRAPHY) {
  const int image_size = 10;
  const int image_n_channels = 3;
  image_stitcher::Image original_img{image_size, image_size, image_n_channels};

  // set right line
  for (int i = 0; i < image_size; ++i) {
    original_img(i, i, 0) = 255;
  }

  Eigen::Matrix3f homography;
  homography << 1, 0, 4,
                0, 1, 2,
                0, 0, 1;

  image_stitcher::Image expected_img{image_size, image_size, image_n_channels};
  for (int i = 0; i < image_size - 4; ++i) {
    expected_img(i + 4, (i + 2), 0) = 255;
  }

  image_stitcher::Image output_img{image_size, image_size, image_n_channels};
  image_stitcher::image_transformation::applyHomography(original_img, output_img, homography);

  for (int i = 0; i < image_size; ++i) {
    for (int j = 0; j < image_size; ++j) {
      ASSERT_EQ(expected_img(i, j, 0), output_img(i, j, 0));
      ASSERT_EQ(expected_img(i, j, 1), output_img(i, j, 1));
      ASSERT_EQ(expected_img(i, j, 2), output_img(i, j, 2));
    }
  }
}

TEST(IMAGE_TRANSFORMATION, APPLY_HOMOGRAPHY_PTP) {
  Eigen::MatrixXf source_pts{3, 4}, target_pts{3, 4};

  source_pts << 0, 0, 4, 4,
                0, 4, 0, 4,
                1, 1, 1, 1;
  target_pts << 0, 1, 4, 3,
                0, 4, 0, 4,
                1, 1, 1, 1;

  auto homography = image_stitcher::homography_calculator::calcHomography(source_pts, target_pts);

  auto transformed_pts = image_stitcher::image_transformation::applyHomography(homography, source_pts);

  for (int i = 0; i < target_pts.rows(); ++i) {
    for (int j = 0; j < target_pts.cols(); ++j) {
      ASSERT_NEAR(target_pts(i, j), transformed_pts(i, j), 1e-5);
    }
  }
}

#endif //PANORAMA_VIEW_TEST_IMAGE_TRANSFORMATION_H
