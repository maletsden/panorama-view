#ifndef PANORAMA_VIEW_TEST_IMAGE_TRANSFORMATION_H
#define PANORAMA_VIEW_TEST_IMAGE_TRANSFORMATION_H

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "../src/image_stitcher/image_stitcher.h"

TEST(IMAGE_TRANSFORMATION, APPLY_HOMOGRAPHY) {
  const int image_size = 10;
  const int image_n_channels = 3;
  image_stitcher::Image original_img{image_size, image_size * image_n_channels};

  // set right line
  for (int i = 0; i < image_size; ++i) {
    original_img(i, i * 3) = 255;
  }

  Eigen::Matrix3f homography;
  homography << 1, 0, 4,
                0, 1, 2,
                0, 0, 1;

  image_stitcher::Image expected_img{image_stitcher::Image::Zero(image_size, image_size * image_n_channels)};
  for (int i = 0; i < image_size - 4; ++i) {
    expected_img(i + 4, (i + 2) * 3) = 255;
  }

  image_stitcher::Image output_img{image_stitcher::Image::Zero(image_size, image_size * image_n_channels)};
  image_stitcher::image_transformation::applyHomography(original_img, output_img, homography);

  std::cout << expected_img.cast<int>() << std::endl << std::endl;
  std::cout << output_img.cast<int>() << std::endl;
  for (int i = 0; i < image_size; ++i) {
    for (int j = 0; j < image_size; ++j) {
      ASSERT_EQ(expected_img(i, j), output_img(i, j));
    }
  }
}

#endif //PANORAMA_VIEW_TEST_IMAGE_TRANSFORMATION_H
