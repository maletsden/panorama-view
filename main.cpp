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

  auto start = get_current_time_fenced();

  // TODO: replace opencv SIFT with another library or write the new one
  std::vector<cv::Mat> images_cv{
      cv::imread("test-images/mountain_1.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_2.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_3.jpg", cv::IMREAD_COLOR),
      cv::imread("test-images/mountain_4.jpg", cv::IMREAD_COLOR),
  };

  //  const std::size_t KRefImgIdx = images.size() >> 1u;
  const std::size_t KRefImgIdx = 1;
  const std::size_t KLeftImgNum = KRefImgIdx;
  const std::size_t KRightImgNum = images_cv.size() - KRefImgIdx - 1;
  const std::size_t KRANSACIterNum = 200;
  const auto KTransformCost = image_stitcher::homography_calculator::MSE;
  const std::size_t KKNNNeighNum = 2;

  std::vector<image_stitcher::Image> images;
  images.reserve(images_cv.size());
  for (auto& image: images_cv) {
    images.emplace_back(image.data, image.rows, image.cols, image.channels());
  }

  std::vector<cv::Mat> gray_images{images_cv.size()};

  auto detector = cv::SIFT::create();
  std::vector<std::vector<cv::KeyPoint>> keypoints_cv{images_cv.size()};
  std::vector<cv::Mat> descriptors_cv{images_cv.size()};
  std::vector<Eigen::MatrixXf> descriptors{images_cv.size()};


  for (std::size_t i = 0; i < images_cv.size(); ++i) {
    // convert image to grayscale
    cv::cvtColor(images_cv[i], gray_images[i], cv::COLOR_BGR2GRAY);
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
    cv::drawMatches(images_cv[i], keypoints_cv[i], images_cv[i + 1], keypoints_cv[i + 1], matches_cv, img_w_matches);
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
    cv::drawMatches(images_cv[i], keypoints_cv[i], images_cv[i - 1], keypoints_cv[i - 1], matches_cv, img_w_matches);
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

  const auto& ref_img_eigen = images[KRefImgIdx];

  image_stitcher::Image panorama{
    ref_img_eigen.rows, ref_img_eigen.cols * images.size(), ref_img_eigen.channels
  };

  // set reference image
  panorama.data.block(
    0, ref_img_eigen.cols * ref_img_eigen.channels * KRefImgIdx,
    ref_img_eigen.rows, ref_img_eigen.cols * ref_img_eigen.channels
  ) = ref_img_eigen.data;

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
      images[i], panorama, direct_left_homography[i],
      0, static_cast<int>(ref_img_eigen.cols * KRefImgIdx)
    );

#if DEBUG
    cv::Mat panorama_cv;
    cv::eigen2cv(panorama.data, panorama_cv);
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
        images[i], panorama, direct_right_homography.back(),
        0, static_cast<int>(ref_img_eigen.cols * KRefImgIdx)
    );

#if DEBUG
    cv::Mat panorama_cv;
    cv::eigen2cv(panorama.data, panorama_cv);
    panorama_cv = panorama_cv.reshape(3, 0);

    imshow("panorama_cv", panorama_cv);
    cv::waitKey(0);
#endif
  }

  // crop black parts of panorama
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> black_mask{panorama.rows, panorama.cols};
  for (std::size_t row_i = 0; row_i < panorama.rows; ++row_i)
  {
    for (std::size_t col_i = 0; col_i < panorama.cols; ++col_i)
    {
      black_mask(row_i, col_i) = (
        panorama(row_i, col_i, 0) | panorama(row_i, col_i, 1) | panorama(row_i, col_i, 2)
      ) == 0 ? 1 : 0;
    }
  }

  std::size_t left_col = 0;
  for (std::size_t col_i = 0; col_i < panorama.cols; ++col_i) {
    auto col_start = black_mask.data() + col_i * black_mask.rows();
    if (std::count(col_start, col_start + black_mask.rows(), 1) < black_mask.rows() * 0.05) {
      break;
    }
    left_col++;
  }

  std::size_t right_col = panorama.cols - 1;
  for (std::size_t col_i = panorama.cols - 1; col_i > 0; --col_i) {
    auto col_start = black_mask.data() + col_i * black_mask.rows();
    if (std::count(col_start, col_start + black_mask.rows(), 1) < black_mask.rows() * 0.05) {
      break;
    }
    right_col--;
  }

  auto cropped_width = right_col - left_col;

  auto black_pixel = std::count(
    black_mask.data() + left_col * black_mask.rows(),
    black_mask.data() + (std::min(right_col + 1, panorama.cols)) * black_mask.rows(),
    1
  );
  auto panorama_proportion = 1 - black_pixel / (panorama.rows * cropped_width);

  std::size_t new_rows_num = panorama.rows * std::sqrt(panorama_proportion);
  std::size_t new_cols_num = cropped_width * std::sqrt(panorama_proportion);

  image_stitcher::ImageData cropped_panorama{new_rows_num, new_cols_num * panorama.channels};
  cropped_panorama = panorama.data.block(
      (panorama.rows - new_rows_num) / 2, ((cropped_width - new_cols_num) / 2 + left_col) * panorama.channels,
      new_rows_num, new_cols_num * panorama.channels
  );

  // Save image
  cv::Mat cropped_panorama_cv;
  cv::eigen2cv(cropped_panorama, cropped_panorama_cv);
  cropped_panorama_cv = cropped_panorama_cv.reshape(3, 0);

  cv::imwrite("results/mountain.jpg", cropped_panorama_cv);

  auto finish = get_current_time_fenced();

  std::cout << "Panorama successfully created!" << std::endl;
  std::cout << "Total consumed time: " << to_ms(finish - start) << std::endl;



  // TODO: crop image
}