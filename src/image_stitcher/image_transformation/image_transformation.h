#ifndef PANORAMA_VIEW_IMAGE_TRANSFORMATION1_H
#define PANORAMA_VIEW_IMAGE_TRANSFORMATION1_H

#include <Eigen/Dense>
#include "../image_stitcher.h"
#include <array>
#include <limits>

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
   */
  void applyHomography(
    const image_stitcher::Image& src_img, image_stitcher::Image& dst_img,
    const Eigen::Matrix3f& homography, long shift_by_rows = 0, long shift_by_cols = 0
  );

  /**
   *
   * @param homography
   * @param pts
   * @return
   */
  Eigen::MatrixXf applyHomography(
      const Eigen::Matrix3f& homography, const Eigen::MatrixXf& pts
  );
}

#endif //PANORAMA_VIEW_IMAGE_TRANSFORMATION1_H
