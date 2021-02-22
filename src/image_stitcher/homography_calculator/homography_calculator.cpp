#include <random>
#include "homography_calculator.h"

double image_stitcher::homography_calculator::MSE(const Eigen::MatrixXf& pts1, const Eigen::MatrixXf& pts2) {
  return static_cast<double>((pts1 - pts2).array().square().sum());
}

Eigen::Matrix3f image_stitcher::homography_calculator::calcHomography(
    const Eigen::MatrixXf& src_pts, const Eigen::MatrixXf& dst_pts
) {
  typedef Eigen::Matrix<float, 8, 8> HomographyMatrix;
  HomographyMatrix PH;
  Eigen::VectorXf b{8};
  for (unsigned int i = 0, j = 0; i < 4; i++) {

    const double srcX = src_pts(0, i);
    const double srcY = src_pts(1, i);
    const double dstX = dst_pts(0, i);
    const double dstY = dst_pts(1, i);

    b(j) = dstX;
    PH.row(j++) << srcX, srcY,  1.,   0.,   0.,  0., -srcX*dstX, -srcY*dstX;
    b(j) = dstY;
    PH.row(j++) <<   0.,   0.,  0., srcX, srcY,  1., -srcX*dstY, -srcY*dstY;
  }

  // solve PH * x = b
  Eigen::VectorXf x{8};
  Eigen::HouseholderQR<HomographyMatrix> qr(PH);
  x = qr.solve(b);

  Eigen::Matrix3f result;
  result << x(0), x(1), x(2),
            x(3), x(4), x(5),
            x(6), x(7), 1;

  return result;
}

Eigen::Matrix3f image_stitcher::homography_calculator::RANSAC(
    const Eigen::MatrixXf& src_pts, const Eigen::MatrixXf& dst_pts,
    transformCost cost, std::size_t num_iter
) {
  if (src_pts.rows() != 3 || dst_pts.rows() != 3) {
    throw std::runtime_error(
        "image_stitcher::homography_calculator::RANSAC: source and destination points must have 3 x m dimensions"
    );
  }

  if (src_pts.cols() != dst_pts.cols()) {
    throw std::runtime_error(
      "image_stitcher::homography_calculator::RANSAC: source and destination points have different dimensions"
    );
  }

  Eigen::Matrix3f best_homography;
  double min_cost = std::numeric_limits<double>::max();

  std::vector<std::size_t> all_points_indexes;
  all_points_indexes.reserve(src_pts.cols());
  for (int i = 0; i < src_pts.cols(); ++i) {
    all_points_indexes.push_back(i);
  }

  const int sample_size = 4;
  std::vector<std::size_t> sample_points_indexes;
  sample_points_indexes.reserve(sample_size);

  Eigen::MatrixXf sample_src_pts{3, 4}, sample_dst_pts{3, 4};

  for (std::size_t iter = 0; iter < num_iter; ++iter) {

    // sample points
    std::sample(all_points_indexes.begin(), all_points_indexes.end(), std::back_inserter(sample_points_indexes),
                sample_size, std::mt19937{std::random_device{}()});
    for (int i = 0; i < sample_size; ++i) {
      sample_src_pts.col(i) = src_pts.col(sample_points_indexes[i]);
      sample_dst_pts.col(i) = dst_pts.col(sample_points_indexes[i]);
    }
    sample_points_indexes.clear();

    auto homography = calcHomography(sample_src_pts, sample_dst_pts);

    // update best homography if needed
    auto sample_cost = cost(homography * src_pts, dst_pts);
    if (sample_cost < min_cost) {
      min_cost = sample_cost;
      best_homography = homography;
    }
  }

  return best_homography;
}