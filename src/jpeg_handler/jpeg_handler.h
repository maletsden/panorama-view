#ifndef PANORAMA_VIEW_JPEG_READER_H
#define PANORAMA_VIEW_JPEG_READER_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "../jpeg_img.h"
#include <jpeglib.h>

namespace jpeg_handler {
  void writeImage(const JPEGImage& img, const std::string &filename, uint8_t quality = 255);
  JPEGImage readImage(const std::string &filename);
}

#endif //PANORAMA_VIEW_JPEG_READER_H
