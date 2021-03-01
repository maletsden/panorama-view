#include <iostream>
#include "image_transformation.h"

void copy_pixel(const std::uint8_t *pixel, std::uint8_t *out_pixel) {
  *out_pixel++ = *pixel++;
  *out_pixel++ = *pixel++;
  *out_pixel = *pixel;
}

bool isEmpty(const std::uint8_t *pixel) {
  return *pixel++ == 0 && *pixel++ == 0 && *pixel == 0;
}

void image_stitcher::image_transformation::applyHomography(
    const image_stitcher::Image& src_img, image_stitcher::Image& dst_img,
    const Eigen::Matrix3f& homography, int shift_by_rows, int shift_by_cols, int img_n_channels
) {
  Eigen::MatrixXf transformMat{2, 3};
  transformMat << homography(0, 0), homography(0, 1), homography(0, 2) + static_cast<float>(shift_by_rows),
                  homography(1, 0), homography(1, 1), homography(1, 2) + static_cast<float>(shift_by_cols);

  for (long row_i = 0; row_i < src_img.rows(); ++row_i) {
    auto shift_x = static_cast<float>(row_i) * transformMat(0, 0) + transformMat(0, 2);
    auto shift_y = static_cast<float>(row_i) * transformMat(1, 0) + transformMat(1, 2);
    for (long col_i = 0; col_i < src_img.cols() / img_n_channels; ++col_i) {

      auto new_row_i = static_cast<long>(shift_x + static_cast<float>(col_i) * transformMat(0, 1));
      auto new_col_i = static_cast<long>(shift_y + static_cast<float>(col_i) * transformMat(1, 1));

      if (new_row_i < 0 || new_row_i >= dst_img.rows() || new_col_i < 0 || new_col_i >= (dst_img.cols() / img_n_channels)) {
        continue;
      }

      auto out_pixel = dst_img.data() + new_row_i * dst_img.cols() + new_col_i * img_n_channels;

      if (!isEmpty(out_pixel)) {
        continue;
      }

      auto pixel = src_img.data() + row_i * src_img.cols() + col_i * img_n_channels;
      copy_pixel(pixel, out_pixel);

      // illuminate stitches due to float to long conversion
      if (new_row_i - 1 > 0) {
        copy_pixel(pixel, out_pixel - dst_img.cols());
      }

      if (new_col_i - 1 > 0) {
        copy_pixel(pixel, out_pixel - img_n_channels);
      }

      if (new_row_i - 1 > 0 && new_col_i - 1 > 0) {
        copy_pixel(pixel, out_pixel - dst_img.cols() - img_n_channels);
      }
    }
  }

//  std::vector<uint8_t*> neighbours_pixels;
//  neighbours_pixels.reserve(4);
//  for (long row_i = 0; row_i < dst_img.rows(); ++row_i) {
//    for (long col_i = 0; col_i < dst_img.cols() / img_n_channels; ++col_i) {
//      neighbours_pixels.clear();
//      auto out_pixel = dst_img.data() + row_i * dst_img.cols() + col_i * img_n_channels;
//
//      if (!isEmpty(out_pixel)) {
//        continue;
//      }
//
//      if (col_i - 1 > 0) {
//        neighbours_pixels.push_back(dst_img.data() + row_i * dst_img.cols() + (col_i - 1) * img_n_channels);
//      }
//
//      if (row_i - 1 > 0) {
//        neighbours_pixels.push_back(dst_img.data() + (row_i - 1) * dst_img.cols() + col_i * img_n_channels);
//      }
//
//      if (row_i + 1 < dst_img.rows()) {
//        neighbours_pixels.push_back(dst_img.data() + (row_i + 1) * dst_img.cols() + col_i * img_n_channels);
//      }
//
//      if (col_i + 1 < (dst_img.cols() / img_n_channels)) {
//        neighbours_pixels.push_back(dst_img.data() + row_i * dst_img.cols() + (col_i + 1) * img_n_channels);
//      }
//
//      uint16_t r = 0, g = 0, b = 0, count = 0;
//      for (uint8_t *n_pixel: neighbours_pixels) {
//        if (isEmpty(n_pixel)) {
//          continue;
//        }
//        r += *n_pixel++;
//        g += *n_pixel++;
//        b += *n_pixel;
//        count++;
//      }
//      if (!count) {
//        continue;
//      }
//      r /= count;
//      g /= count;
//      b /= count;
//      uint8_t convolution_pixel[3] = {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b)};
//
//      copy_pixel(convolution_pixel, out_pixel);
//    }
//  }
}
