#ifndef PANORAMA_VIEW_JPEG_IMG_H
#define PANORAMA_VIEW_JPEG_IMG_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <bits/unique_ptr.h>
#include <jpeglib.h>

class JPEGImage {
public:
  JDIMENSION m_image_height;
  JDIMENSION m_image_width;
  JDIMENSION m_image_channels;
  J_COLOR_SPACE m_in_color_space;
  std::unique_ptr<JSAMPLE[]> m_image_data;

  JPEGImage(JDIMENSION img_h, JDIMENSION img_w, int img_c = 3, J_COLOR_SPACE in_color_space = JCS_RGB):
    m_image_height(img_h), m_image_width(img_w), m_image_channels(img_c), m_in_color_space(in_color_space),
    m_image_data(std::unique_ptr<JSAMPLE[]>(new JSAMPLE[img_h * img_w * img_c])) {}



  JSAMPLE* getPixel(std::size_t row_idx, std::size_t col_idx);
  JSAMPLE* getPixelChannel(std::size_t row_idx, std::size_t col_idx, std::size_t channel_idx);
  JPEGImage RGBToGRAYSCALE();
};

#endif //PANORAMA_VIEW_JPEG_IMG_H
