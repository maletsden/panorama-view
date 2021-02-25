#include <iostream>
#include <vector>
#include <chrono>

#include "src/jpeg_img.h"
#include "src/jpeg_handler/jpeg_handler.h"

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include "src/image_stitcher/image_stitcher.h"

using namespace cv;
using namespace std;

void filterGoodMatches(const image_stitcher::Matches& matches, image_stitcher::Matches& good_matches) {
  auto match_w_min_distance = std::min_element(
      matches.begin(), matches.end(),
      [](const image_stitcher::Match& m1, const image_stitcher::Match& m2) {
        return m1.distance < m2.distance;
      }
  );

  std::copy_if(
    matches.begin(), matches.end(), std::back_inserter(good_matches),
    [&match_w_min_distance](const image_stitcher::Match& match) {
      return match.distance < match_w_min_distance->distance * 3;
    }
  );
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> getKeypointsPair(
  const image_stitcher::Matches& matches, const std::vector<cv::KeyPoint>& src_keypoints,
  const std::vector<cv::KeyPoint>& dst_keypoints
) {
  auto key_pair = std::make_pair(
    Eigen::MatrixXf{3, matches.size()},
    Eigen::MatrixXf{3, matches.size()}
  );

  for (std::size_t i = 0; i < matches.size(); ++i) {
    auto pt1 = src_keypoints[matches[i].pt1].pt;
    auto pt2 = dst_keypoints[matches[i].pt2].pt;

    std::get<0>(key_pair).col(i) << pt1.y, pt1.x, 1;
    std::get<1>(key_pair).col(i) << pt2.y, pt2.x, 1;
  }

  return key_pair;
}

int main() {
//  JPEGImage img{200, 200};
//
//  // dummy image test
//  for (JDIMENSION i = 0; i < img.m_image_height; ++i) {
//    auto pixel = img.getPixel(i, i);
//    *pixel++ = 255;
//    *pixel++ = 255;
//    *pixel = 255;
//  }
//
//  jpeg_handler::writeImage(img,"test.jpg");
//  auto mountain1 = jpeg_handler::readImage("test-images/mountain_1.jpg");
//  jpeg_handler::writeImage(mountain1, "mountain1_.jpg");
//
//  auto mountain1_grey = mountain1.RGBToGRAYSCALE();
//
//  jpeg_handler::writeImage(mountain1_grey, "mountain1_grey_.jpg");

  // TODO: replace opencv SIFT with another library or write the new one
  std::vector<cv::Mat> images{
      cv::imread("test-images/mountain_1.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_2.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_3.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_4.jpg", cv::IMREAD_COLOR),
  };

  std::vector<Eigen::Map<image_stitcher::Image>> images_eigen;
  images_eigen.reserve(images.size());
  for (auto& image: images) {
    images_eigen.emplace_back(image.data, image.rows, image.cols * image.channels());
  }

  std::vector<cv::Mat> gray_images{images.size()};

  auto detector = cv::SIFT::create();
  std::vector<std::vector<cv::KeyPoint>> keypoints_cv{images.size()};
  std::vector<cv::Mat> descriptors_cv{images.size()};
  std::vector<Eigen::MatrixXf> descriptors{images.size()};


  for (std::size_t i = 0; i < images.size(); ++i) {
    // convert image to grayscale
    cv::cvtColor(images[i], gray_images[i], cv::COLOR_BGR2GRAY);
    // calculate keypoints and their descriptors
    detector->detectAndCompute(gray_images[i], cv::noArray(), keypoints_cv[i], descriptors_cv[i]);
    // convert to eigen matrix
    cv::cv2eigen(descriptors_cv[i], descriptors[i]);
  }

  const std::size_t ref_img_idx = images.size() >> 1u;

  std::vector<image_stitcher::Matches> left_matches, right_matches;
  left_matches.reserve(ref_img_idx);
  right_matches.reserve(images.size() - ref_img_idx - 1);

  // calculates matches by running brut force k-Nearest-Neighbours algorithm
  for (std::size_t i = 0; i < ref_img_idx; ++i) {
    left_matches.emplace_back(
      image_stitcher::knn_finder::bruteForce(descriptors[i], descriptors[i + 1], 3)
    );
  }

  for (std::size_t i = ref_img_idx + 1; i < images.size(); ++i) {
    right_matches.emplace_back(
      image_stitcher::knn_finder::bruteForce(descriptors[i], descriptors[i - 1], 3)
    );
  }

//  return 0;

  // choose only "good" matches (ones that matches pretty well)
  std::vector<image_stitcher::Matches> left_good_matches{left_matches.size()},
                                       right_good_matches{right_matches.size()};
  for (std::size_t i = 0; i < left_matches.size(); ++i) {
    filterGoodMatches(left_matches[i], left_good_matches[i]);
  }
  for (std::size_t i = 0; i < right_matches.size(); ++i) {
    filterGoodMatches(right_matches[i], right_good_matches[i]);
  }

  std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> left_keypoints, right_keypoints;
  for (std::size_t i = 0; i < left_good_matches.size(); ++i) {
    left_keypoints.emplace_back(
      getKeypointsPair(left_good_matches[i], keypoints_cv[i], keypoints_cv[i + 1])
    );
  }
  for (std::size_t i = 0; i < right_good_matches.size(); ++i) {
    right_keypoints.emplace_back(
        getKeypointsPair(right_good_matches[i], keypoints_cv[ref_img_idx + i + 1], keypoints_cv[ref_img_idx + i])
    );
  }

  std::vector<Eigen::Matrix3f> left_homography, right_homography;
  left_homography.reserve(left_keypoints.size());
  right_homography.reserve(right_keypoints.size());
  for (const auto& keypoints_pair: left_keypoints) {
    left_homography.emplace_back(
      image_stitcher::homography_calculator::RANSAC(
        keypoints_pair, image_stitcher::homography_calculator::MSE, 500
      )
    );
  }
  for (const auto& keypoints_pair: right_keypoints) {
    right_homography.emplace_back(
      image_stitcher::homography_calculator::RANSAC(
        keypoints_pair, image_stitcher::homography_calculator::MSE, 500
      )
    );
  }

  const auto& ref_img_eigen = images_eigen[ref_img_idx];
  image_stitcher::Image panorama{
    image_stitcher::Image::Zero(ref_img_eigen.rows(), ref_img_eigen.cols() * images.size())
  };
  panorama.block(
    0, ref_img_eigen.cols() * ref_img_idx, ref_img_eigen.rows(), ref_img_eigen.cols()
  ) = ref_img_eigen;

  std::vector<Eigen::Matrix3f> direct_left_homography, direct_right_homography;
  direct_left_homography.resize(left_homography.size());
  direct_right_homography.reserve(right_homography.size());

  for (int i = static_cast<int>(ref_img_idx) - 1; i >= 0; --i) {
    if (i == static_cast<int>(ref_img_idx) - 1) {
      direct_left_homography[i] = left_homography[i];
    } else {
      direct_left_homography[i] = left_homography[i] * direct_left_homography[i + 1];
    }

    image_stitcher::image_transformation::applyHomography(
        images_eigen[i], panorama, direct_left_homography[i],
        0, static_cast<int>(ref_img_eigen.cols() * ref_img_idx / 3)
    );

    cv::Mat panorama_cv;
    cv::eigen2cv(panorama, panorama_cv);
    panorama_cv = panorama_cv.reshape(3, 0);

    imshow("panorama_cv", panorama_cv);
    cv::waitKey(0);
  }

  for (std::size_t i = ref_img_idx + 1; i < images.size(); ++i) {
    auto& homo = right_homography[i - ref_img_idx - 1];

    if (i == ref_img_idx + 1) {
      direct_right_homography.emplace_back(homo);
    } else {
      direct_right_homography.emplace_back(homo * direct_right_homography.back());
    }

    image_stitcher::image_transformation::applyHomography(
        images_eigen[i], panorama, direct_right_homography.back(),
        0, static_cast<int>(ref_img_eigen.cols() * ref_img_idx / 3)
    );

    cv::Mat panorama_cv;
    cv::eigen2cv(panorama, panorama_cv);
    panorama_cv = panorama_cv.reshape(3, 0);

    imshow("panorama_cv", panorama_cv);
    cv::waitKey(0);
  }


  // TODO: crop image
  // TODO: illuminate all black dots (by average of neighbours)
}