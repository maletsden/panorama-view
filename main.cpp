#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>

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

#define DEBUG 0

image_stitcher::Matches filterGoodMatches(const std::vector<image_stitcher::Matches>& matches) {
  image_stitcher::Matches good_matches;

  // Lowe's ratio test (second best match distance, must be at least twice larger that first one)
  // in other words, select only matches that are distinguishable
  for (auto &pt_matches: matches) {
    if ((pt_matches[0].distance / pt_matches[1].distance) <= 0.5) {
      good_matches.emplace_back(pt_matches[0]);
    }
  }

  return good_matches;
}

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
  // to ensure that compiler will not change the order fences are added
  std::atomic_thread_fence(std::memory_order_seq_cst);
  auto res_time = std::chrono::high_resolution_clock::now();
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return res_time;
}

template<class D>
inline long long to_ms(const D& d) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
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

  auto start = get_current_time_fenced();

  // TODO: replace opencv SIFT with another library or write the new one
  std::vector<cv::Mat> images{
      cv::imread("test-images/mountain_1.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_2.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_3.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_4.jpg", cv::IMREAD_COLOR),
  };

  //  const std::size_t KRefImgIdx = images.size() >> 1u;
  const std::size_t KRefImgIdx = 1;
  const std::size_t KLeftImgNum = KRefImgIdx;
  const std::size_t KRightImgNum = images.size() - KRefImgIdx - 1;
  const std::size_t KRANSACIterNum = 200;
  const auto KTransformCost = image_stitcher::homography_calculator::MSE;
  const std::size_t KKNNNeighNum = 2;

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

  // choose only "good" matches (ones that matches pretty well)
  std::vector<std::vector<image_stitcher::Match>> left_good_matches, right_good_matches;
  left_good_matches.reserve(KLeftImgNum);
  right_good_matches.reserve(KRightImgNum);

  std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> left_keypoint_pairs, right_keypoint_pairs;
  left_keypoint_pairs.reserve(KLeftImgNum);
  right_keypoint_pairs.reserve(KRightImgNum);

  std::vector<Eigen::Matrix3f> left_homography, right_homography;
  left_homography.reserve(KLeftImgNum);
  right_homography.reserve(KRightImgNum);

  // calculates matches by running brut force or KDTree based algorithm k-Nearest-Neighbours algorithm
  for (std::size_t i = 0; i < KRefImgIdx; ++i) {
//    auto matches = image_stitcher::knn_finder::bruteForce(descriptors[i], descriptors[i + 1], kNN_k);
    auto matches = image_stitcher::knn_finder::randomKDTreeForest(descriptors[i], descriptors[i + 1], KKNNNeighNum);

    left_good_matches.emplace_back(filterGoodMatches(matches));

#if DEBUG
    std::vector<cv::DMatch> matches_cv;
    matches_cv.reserve(left_good_matches.back().size());
    for (auto &match: left_good_matches.back()) {
      matches_cv.emplace_back(static_cast<int>(match.pt1), static_cast<int>(match.pt2), match.distance);
    }

    cv::Mat img_w_matches;
    cv::drawMatches(images[i], keypoints_cv[i], images[i + 1], keypoints_cv[i + 1], matches_cv, img_w_matches);
    imshow("img_w_matches", img_w_matches);
    cv::waitKey(0);
#endif

    left_keypoint_pairs.emplace_back(
      getKeypointsPair(left_good_matches.back(), keypoints_cv[i], keypoints_cv[i + 1])
    );

    left_homography.emplace_back(
      image_stitcher::homography_calculator::RANSAC(
        left_keypoint_pairs.back(), KTransformCost, KRANSACIterNum
      )
    );
  }

  for (std::size_t i = KRefImgIdx + 1; i < images.size(); ++i) {
//    auto matches = image_stitcher::knn_finder::bruteForce(descriptors[i], descriptors[i - 1], kNN_k);
    auto matches = image_stitcher::knn_finder::randomKDTreeForest(descriptors[i], descriptors[i - 1], KKNNNeighNum);

    right_good_matches.emplace_back(filterGoodMatches(matches));

#if DEBUG
    std::vector<cv::DMatch> matches_cv;
    matches_cv.reserve(right_good_matches.back().size());
    for (auto &match: right_good_matches.back()) {
      matches_cv.emplace_back(static_cast<int>(match.pt1), static_cast<int>(match.pt2), match.distance);
    }

    cv::Mat img_w_matches;
    cv::drawMatches(images[i], keypoints_cv[i], images[i - 1], keypoints_cv[i - 1], matches_cv, img_w_matches);
    imshow("img_w_matches", img_w_matches);
    cv::waitKey(0);
#endif
    right_keypoint_pairs.emplace_back(
      getKeypointsPair(right_good_matches.back(), keypoints_cv[i], keypoints_cv[i - 1])
    );

    right_homography.emplace_back(
      image_stitcher::homography_calculator::RANSAC(
        right_keypoint_pairs.back(), image_stitcher::homography_calculator::Huber, KRANSACIterNum
      )
    );
  }

  const auto& ref_img_eigen = images_eigen[KRefImgIdx];
  image_stitcher::Image panorama{
    image_stitcher::Image::Zero(ref_img_eigen.rows(), ref_img_eigen.cols() * images.size())
  };
  panorama.block(
    0, ref_img_eigen.cols() * KRefImgIdx, ref_img_eigen.rows(), ref_img_eigen.cols()
  ) = ref_img_eigen;

  std::vector<Eigen::Matrix3f> direct_left_homography, direct_right_homography;
  direct_left_homography.resize(KLeftImgNum);
  direct_right_homography.reserve(KRightImgNum);

  for (int i = static_cast<int>(KRefImgIdx) - 1; i >= 0; --i) {
    if (i == static_cast<int>(KRefImgIdx) - 1) {
      direct_left_homography[i] = left_homography[i];
    } else {
      direct_left_homography[i] = left_homography[i] * direct_left_homography[i + 1];
    }

    image_stitcher::image_transformation::applyHomography(
      images_eigen[i], panorama, direct_left_homography[i],
      0, static_cast<int>(ref_img_eigen.cols() * KRefImgIdx / 3)
    );

#if DEBUG
    cv::Mat panorama_cv;
    cv::eigen2cv(panorama, panorama_cv);
    panorama_cv = panorama_cv.reshape(3, 0);

    imshow("panorama_cv", panorama_cv);
    cv::waitKey(0);
#endif
  }

  for (std::size_t i = KRefImgIdx + 1; i < images.size(); ++i) {
    auto& homo = right_homography[i - KRefImgIdx - 1];

    if (i == KRefImgIdx + 1) {
      direct_right_homography.emplace_back(homo);
    } else {
      direct_right_homography.emplace_back(homo * direct_right_homography.back());
    }

    image_stitcher::image_transformation::applyHomography(
        images_eigen[i], panorama, direct_right_homography.back(),
        0, static_cast<int>(ref_img_eigen.cols() * KRefImgIdx / 3)
    );

#if DEBUG
    cv::Mat panorama_cv;
    cv::eigen2cv(panorama, panorama_cv);
    panorama_cv = panorama_cv.reshape(3, 0);

    imshow("panorama_cv", panorama_cv);
    cv::waitKey(0);
#endif
  }

  cv::Mat panorama_cv;
  cv::eigen2cv(panorama, panorama_cv);
  panorama_cv = panorama_cv.reshape(3, 0);

  cv::imwrite("panorama_view.jpg", panorama_cv);

  auto finish = get_current_time_fenced();

  std::cout << "Panorama successfully created!" << std::endl;
  std::cout << "Total consumed time: " << to_ms(finish - start) << std::endl;



  // TODO: crop image
  // TODO: illuminate all black dots (by average of neighbours)
}