#pragma once

#include "Hmesh.h"
#include <Eigen/Eigen>
#include "../utils/QuadraticObjectiveD.h"
#include <memory>
#include <vector>
namespace geom {

class ARAP {
public:
  ARAP(const geom::Hmesh &mesh,
               const Eigen::MatrixXi &edges = Eigen::MatrixXi(), double shrink_factor = 0.8);

  ARAP(const ARAP &other);

  ARAP &set_objective();
  ARAP &set_fixed_indices(const std::vector<int> &fixed_indices);
  ARAP &set_fixed_indices(const Eigen::VectorXi &fixed_indices);
  ARAP &precompute();
  Eigen::MatrixXd solve(const Eigen::MatrixXd &X, bool zero_rhs = false);
  ARAP &set_fixed_positions(const Eigen::MatrixXd &fixed_positions);

  ARAP &set_fixed(const std::vector<int> &indices,
                          const Eigen::MatrixXd &positions);

  ARAP &add_soft_constraints(double weight,
                                     const std::vector<int> &indices,
                                     const Eigen::MatrixXd &positions);
  void set_soft_constraints_weight(double weight);

  void set_edges_weight(double weight);

  struct VertexOneRing {
    VertexOneRing() {
      v0.reserve(16);
      v1.reserve(16);
      edges_vec.reserve(16);
      cot_weights.reserve(16);
    }
    // The spokes and rim edge indices.
    std::vector<int> v0, v1;
    // Corresponding cotangent weights.
    std::vector<double> cot_weights;
    std::vector<Eigen::RowVector3d> edges_vec;
    // The original edges (V(v1, :) - V(v0, :)).
    auto original_edges() const {
      return Eigen::Matrix<double, -1, 3, Eigen::RowMajor>::Map(
          (double *)edges_vec.data(), v0.size(), 3);
    };
    // [start_row, start_row + num_edges()) contain the linear equations
    // corresponding to this one ring.
    int start_row;
    int num_edges() { return v0.size(); }
    Eigen::Block<Eigen::Matrix<double, -1, -1>> edges(Eigen::MatrixXd &rhs) {
      return rhs.middleRows(start_row, num_edges());
    }
  };

public:
  void build_one_rings();

  std::vector<VertexOneRing> one_rings;
  geom::Hmesh mesh;
  double shrink_factor;

  Eigen::MatrixXi edges;
  std::unique_ptr<utils::QuadraticObjectiveD> objective;
};
} // namespace geom