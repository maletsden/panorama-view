#include <gtest/gtest.h>
#include "test_knn_finder.h"
#include "test_homography_calculator.h"
#include "test_image_transformation.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
