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
}

#endif //PANORAMA_VIEW_IMAGE_STITCHER_H
