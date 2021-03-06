#ifndef PANORAMA_VIEW_IMAGE_TRANSFORMATION1_H
#define PANORAMA_VIEW_IMAGE_TRANSFORMATION1_H

#include <Eigen/Dense>
#include "../image_stitcher.h"

namespace image_stitcher::image_transformation {
  /**
   * Apply homography transformation to source image
   * project it onto destination image
   *
   * @param src_img - source image
   * @param dst_img - destination image
   * @param homography - homography transformation matrix
   * @param shift_by_rows - number of rows that need to be shifted from start of destination image
   * @param shift_by_cols - number of cols that need to be shifted from start of destination image
   * @param img_n_channels - the number of channels of images
   */
  void applyHomography(
    const image_stitcher::Image& src_img, image_stitcher::Image& dst_img,
    const Eigen::Matrix3f& homography, int shift_by_rows = 0, int shift_by_cols = 0, int img_n_channels = 3
  );
}

#endif //PANORAMA_VIEW_IMAGE_TRANSFORMATION1_H
