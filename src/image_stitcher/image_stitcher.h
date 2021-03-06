#ifndef PANORAMA_VIEW_IMAGE_STITCHER_H
#define PANORAMA_VIEW_IMAGE_STITCHER_H

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>

namespace image_stitcher {
  /**
   * Keypoints match
   */
  class Match {
  public:
    /**
     * Source point
     */
    std::size_t pt1;

    /**
     * Destination point
     */
    std::size_t pt2;

    /**
     * Distance between points
     */
    double distance;

    /**
     * Construct Match
     *
     * @param pt1_ - source point
     * @param pt2_ - destination point
     * @param d - distance between points
     */
    Match(std::size_t pt1_, std::size_t pt2_, double d): pt1(pt1_), pt2(pt2_), distance(d) {}
  };

  /**
   * Keypoints matches
   */
  typedef std::vector<Match> Matches;

  /**
   * Standard image_stitcher container for image data
   */
  typedef Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> ImageData;

  /**
   * Standard image_stitcher image
   */
  class Image {
  public:
    ImageData data;
    std::size_t rows;
    std::size_t cols;
    std::size_t channels;

    Image(std::size_t rows_, std::size_t cols_, std::size_t channels_):
      data(ImageData::Zero(rows_, cols_ * channels_)), rows(rows_), cols(cols_), channels(channels_) {}

    Image(uint8_t *data_, std::size_t rows_, std::size_t cols_, std::size_t channels_):
        data(Eigen::Map<ImageData>{
          data_, static_cast<Eigen::Index>(rows_), static_cast<Eigen::Index>(cols_ * channels_)
        }), rows(rows_), cols(cols_), channels(channels_) {}

    Image(const ImageData& img, std::size_t channels_):
        data(img), rows(static_cast<std::size_t>(img.rows())),
        cols(static_cast<std::size_t>(img.cols()) / channels_), channels(channels_) {}

    uint8_t& operator()(std::size_t x, std::size_t y, std::size_t z) const
    {
      if (x >= rows || y >= cols || z >= channels)
      {
        throw std::runtime_error("Image: Index error");
      }
      return const_cast<uint8_t &>(data.data()[x * cols * channels + y * channels + z]);
    }
  };
}

#include "knn_finder/knn_finder.h"
#include "homography_calculator/homography_calculator.h"
#include "image_transformation/image_transformation.h"

#endif //PANORAMA_VIEW_IMAGE_STITCHER_H
