#include "jpeg_img.h"

JSAMPLE *JPEGImage::getPixel(std::size_t row_idx, std::size_t col_idx) {
  return m_image_data.get() + row_idx * (m_image_width * m_image_channels) + col_idx * m_image_channels;
}

JSAMPLE *JPEGImage::getPixelChannel(std::size_t row_idx, std::size_t col_idx, std::size_t channel_idx) {
  return getPixel(row_idx, col_idx) + channel_idx;
}

JPEGImage JPEGImage::RGBToGRAYSCALE() {
  JPEGImage img{m_image_height, m_image_width, 1, JCS_GRAYSCALE};
  auto org_data = m_image_data.get();
  auto data = img.m_image_data.get();

  for (JDIMENSION i = 0; i < m_image_height * m_image_width; ++i) {
    auto pixel = org_data + i * m_image_channels;
    data[i] = (pixel[0] + pixel[1] + pixel[2]) / 3;
  }
  return img;
}
