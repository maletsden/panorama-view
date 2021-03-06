#include <iostream>
#include "image_transformation.h"

Eigen::MatrixXf image_stitcher::image_transformation::applyHomography(
    const Eigen::Matrix3f& homography, const Eigen::MatrixXf& pts
)
{
  auto res_pts = homography * pts;

  // divide all vectors by 'z' coordinate
  return res_pts.array().rowwise() / res_pts.row(res_pts.rows() - 1).array();
}

/**
 * Get limits for image transformation
 *
 * @param src_img - source image
 * @param dst_img - destination image
 * @param homography - homography transformation matrix
 * @param shift_by_rows - number of rows that need to be shifted from start of destination image
 * @param shift_by_cols - number of cols that need to be shifted from start of destination image
 *
 * @return [min_row, max_row, min_col, max_col]
 */
std::array<long, 4> getTransformationLimits(
    const image_stitcher::Image& img, image_stitcher::Image& dst_img, const Eigen::Matrix3f& homography,
    long shift_by_rows, long shift_by_cols
)
{
  Eigen::MatrixXf pts{3, 4};
  pts << 0,            0, img.rows - 1, img.rows - 1,
         0, img.cols - 1, img.cols - 1,            0,
         1,            1,            1,            1;

  auto transformed_pts = image_stitcher::image_transformation::applyHomography(homography, pts);

  std::array<long, 4> transform_limits{
    std::numeric_limits<long>::max(), // min_row
    std::numeric_limits<long>::min(), // max_row
    std::numeric_limits<long>::max(), // min_col
    std::numeric_limits<long>::min()  // max_col
  };

  for (int i = 0; i < 4; ++i) {
    long x = static_cast<long>(std::roundf(transformed_pts(0, i)));
    long y = static_cast<long>(std::roundf(transformed_pts(1, i)));
    transform_limits[0] = std::min(transform_limits[0], x);
    transform_limits[1] = std::max(transform_limits[1], x);
    transform_limits[2] = std::min(transform_limits[2], y);
    transform_limits[3] = std::max(transform_limits[3], y);
  }

  // shift and limit to destination image sizes
  transform_limits[0] = std::min(
      std::max(transform_limits[0] + shift_by_rows, long(0)), static_cast<long>(dst_img.rows)
  );
  transform_limits[1] = std::min(
      std::max(transform_limits[1] + shift_by_rows, long(0)), static_cast<long>(dst_img.rows)
  );
  transform_limits[2] = std::min(
      std::max(transform_limits[2] + shift_by_cols, long(0)), static_cast<long>(dst_img.cols)
  );
  transform_limits[3] = std::min(
      std::max(transform_limits[3] + shift_by_cols, long(0)), static_cast<long>(dst_img.cols)
  );

  return transform_limits;
}

void image_stitcher::image_transformation::applyHomography(
    const image_stitcher::Image& src_img, image_stitcher::Image& dst_img,
    const Eigen::Matrix3f& homography, long shift_by_rows, long shift_by_cols
)
{
  auto homo_inv = homography.inverse();

  // limit transformation points for better performance
  auto transform_limits = getTransformationLimits(src_img, dst_img, homography, shift_by_rows, shift_by_cols);

  for (long row_i = transform_limits[0]; row_i < transform_limits[1]; ++row_i)
  {
    auto org_x = static_cast<float>(row_i - shift_by_rows);
    auto shift_x = homo_inv(0, 0) * org_x + homo_inv(0, 2);
    auto shift_y = homo_inv(1, 0) * org_x + homo_inv(1, 2);
    auto shift_z = homo_inv(2, 0) * org_x + homo_inv(2, 2);

    for (long col_i = transform_limits[2]; col_i < transform_limits[3]; ++col_i)
    {
      auto org_y = static_cast<float>(col_i - shift_by_cols);

      float z = shift_z + homo_inv(2, 1) * org_y;
      long x = static_cast<long>(std::roundf((shift_x + homo_inv(0, 1) * org_y) / z));
      long y = static_cast<long>(std::roundf((shift_y + homo_inv(1, 1) * org_y) / z));

      if ((x < 0 || x >= static_cast<long>(src_img.rows)) || (y < 0 || y >= static_cast<long>(src_img.cols)))
      {
        continue;
      }

      dst_img(row_i, col_i, 0) = src_img(x, y, 0);
      dst_img(row_i, col_i, 1) = src_img(x, y, 1);
      dst_img(row_i, col_i, 2) = src_img(x, y, 2);
    }
  }
}
