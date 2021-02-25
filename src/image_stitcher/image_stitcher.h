#ifndef PANORAMA_VIEW_IMAGE_STITCHER_H
#define PANORAMA_VIEW_IMAGE_STITCHER_H

#include <cstddef>
#include <vector>

namespace image_stitcher {
  class Match {
  public:
    std::size_t pt1;
    std::size_t pt2;
    double distance;

    Match(std::size_t pt1_, std::size_t pt2_, double d): pt1(pt1_), pt2(pt2_), distance(d) {}
  };

  typedef std::vector<Match> Matches;
  typedef Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> Image;
}

#include "knn_finder/knn_finder.h"
#include "homography_calculator/homography_calculator.h"
#include "image_transformation/image_transformation.h"

#endif //PANORAMA_VIEW_IMAGE_STITCHER_H
