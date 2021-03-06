#ifndef PANORAMA_VIEW_KD_TREE_H
#define PANORAMA_VIEW_KD_TREE_H
#include <Eigen/Dense>
#include <random>
#include <bits/unique_ptr.h>

/**
 * Randomized K-Dimensional Tree
 * (on each level use randomly chosen dimension from top `kd_tree_dims_num` variance dimensions)
 */
class RandomKDTree {
public:
  /**
   * K-Dimensional Tree Node
   */
  class KDTreeNode {
  public:

    /**
     * Comparison value
     */
    float value = -1;

    /**
     * Pointer to left tree of (smaller that `value`) points
     */
    std::unique_ptr<KDTreeNode> left;

    /**
     * Pointer to right tree of (larger that `value`) points
     */
    std::unique_ptr<KDTreeNode> right;

    /**
     * Flag that tells whether this node is a leaf in the tree
     */
    bool m_is_leaf = false;

    /**
     * Points that are assigned to this node
     */
    std::vector<std::size_t> m_points_indexes;

    /**
     * Construct KDTreeNode
     *
     * @param pts_idx - points that are assigned to this node
     */
    explicit KDTreeNode(std::vector<std::size_t>&& pts_idx): m_points_indexes(std::move(pts_idx)) {}

    /**
     * Default destructor
     */
    ~KDTreeNode() = default;
  };

  /**
   * Points based on which this tree is constructed
   */
  const Eigen::MatrixXf& points;

  /**
   * Root node of RandomKDTree
   */
  std::unique_ptr<KDTreeNode> tree;

  /**
   * The amount of nearest neighbours that will be calculated
   */
  std::size_t k = 2;

  /**
   * The vector of dimensions that are used on each tree level
   */
  std::vector<std::size_t> dimensions_idx;

  /**
   * Constructs RandomKDTree
   *
   * @param points_ - points based on which this tree is constructed
   * @param k_ - the amount of nearest neighbours that will be calculated
   * @param kd_tree_dims_num - the number of first top variance dimensions
   *                           that will be used for constructing this tree
   * @param kd_tree_leaves_percent - the amount of points that will be left on leaves of this tree
   *                                 (in percentage of total points number)
   */
  RandomKDTree(
    const Eigen::MatrixXf& points_, std::size_t k_,
    std::size_t kd_tree_dims_num = 5, float kd_tree_leaves_percent = 0.02
  );

  /**
   * Recursively construct the tree using depth first traversal
   *
   * @param subtree - the pointer to the subtree that will be constructed
   * @param dimension_i - the index of dimension that will be used in this node construction
   */
  void construct(KDTreeNode* subtree, std::size_t dimension_i = 0);

  /**
   * Travers the RandomKDTree to find `k` nearest neighbours for given point
   *
   * @param point - point for KNN calculation
   *
   * @return - indexes of KNN
   */
  [[nodiscard]] std::vector<std::size_t> findKNN(const Eigen::VectorXf& point) const;
};

#endif //PANORAMA_VIEW_KD_TREE_H
