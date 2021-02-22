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


#include "src/image_stitcher/image_stitcher.h"
#include "src/image_stitcher/bf_matcher/bf_matcher.h"
#include "src/image_stitcher/homography_calculator/homography_calculator.h"

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
  const cv::Mat mountain1 = cv::imread("test-images/mountain_1.jpg", cv::IMREAD_COLOR);
  const cv::Mat mountain2 = cv::imread("test-images/mountain_2.jpg", cv::IMREAD_COLOR);

  auto detector = cv::SIFT::create();
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat desc1_cv, desc2_cv;
  detector->detectAndCompute(mountain1, cv::noArray(), keypoints1, desc1_cv);
  detector->detectAndCompute(mountain2, cv::noArray(), keypoints2, desc2_cv);

  // Add results to image and save.
  cv::Mat output;
  cv::drawKeypoints(mountain1, keypoints1, output);
  cv::imwrite("sift_result.jpg", output);


  auto matcher = cv::BFMatcher::create();
  std::vector<std::vector<cv::DMatch> > cv_matches;
  matcher->knnMatch( desc1_cv, desc2_cv, cv_matches, 1 );

  //-- Draw matches
  cv::Mat img_matches;
  drawMatches( mountain1, keypoints1, mountain2, keypoints2, cv_matches, img_matches );

  imshow("img_matches",img_matches);
  cv::waitKey(0);

  // Convert to eigen matrix
  Eigen::MatrixXf desc1, desc2;
  cv::cv2eigen(desc1_cv, desc1);
  cv::cv2eigen(desc2_cv, desc2);

  // calculates matches by running brut force k-Nearest-Neighbours algorithm
  auto matches = image_stitcher::bf_matcher::kNN(desc1, desc2, 1);

  // choose only "good" matches (ones that are matched pretty well)
  auto match_w_min_distance = std::min_element(
    matches.begin(), matches.end(), [](const image_stitcher::Match& m1, const image_stitcher::Match& m2) {
      return m1.distance < m2.distance;
    }
  );

  image_stitcher::Matches good_matches;
  for (auto& match: matches) {
    if (match.distance < match_w_min_distance->distance * 3) {
      good_matches.emplace_back(match);
    }
  }

  // construct 3x(good_matches.size()) matrices with all keypoints
  Eigen::MatrixXf points1{3, good_matches.size()}, points2{3, good_matches.size()};

  for (std::size_t i = 0; i < good_matches.size(); ++i) {
    auto pt1 = keypoints1[good_matches[i].pt1].pt;
    auto pt2 = keypoints2[good_matches[i].pt2].pt;

    points1.col(i) << pt1.x, pt1.y, 1;
    points2.col(i) << pt2.x, pt2.y, 1;
  }

  auto homography = image_stitcher::homography_calculator::RANSAC(points1, points2);

  std::cout << homography << std::endl;

}