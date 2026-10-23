#include "arap.h"
#include "../utils/conversions.h"
#include <chrono>
#include <igl/svd3x3.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ARAP_TERM_INDEX 0
#define EDGES_ENERGY_INDEX 1
#define SOFT_ENERGY_TERM_INDEX 2

namespace geom {
ARAP::ARAP(const geom::Hmesh &mesh, const Eigen::MatrixXi &edges,
           double shrink_factor)
    : mesh(mesh), edges(edges), shrink_factor(shrink_factor) {}

ARAP::ARAP(const ARAP &other) : one_rings(other.one_rings), mesh(other.mesh) {
  objective = std::make_unique<utils::QuadraticObjectiveD>(*other.objective);
}

ARAP &ARAP::set_objective() {
  build_one_rings();
  objective = std::make_unique<utils::QuadraticObjectiveD>(mesh.nv(), 3);
  // Add the ARAP energy.
  objective->add_weighted_equations(
      one_rings.size(), [&](int v, auto &x, const auto &add_eq) {
        auto &ring = one_rings[v];
        for (int ind = 0; ind < ring.v0.size(); ind++) {
          int i = ring.v0[ind], j = ring.v1[ind];
          double w_ij = ring.cot_weights[ind];
          // w_ij |(xj - xi) - rhs|^2.
          add_eq(w_ij, x(j) - x(i));
        }
      });

  objective->add_weighted_equations(
      edges.rows(), [&](int i, auto &x, auto add_eq) {
        add_eq(100, x(edges(i, 1)) - x(edges(i, 0)));
      });
  return *this;
}

ARAP &ARAP::set_fixed_indices(const std::vector<int> &fixed_indices) {
  objective->set_known_indices(
      Eigen::VectorXi::Map(fixed_indices.data(), fixed_indices.size()));
  return *this;
}

ARAP &ARAP::set_fixed_indices(const Eigen::VectorXi &fixed_indices) {
  objective->set_known_indices(fixed_indices);
  return *this;
}

ARAP &ARAP::set_fixed(const std::vector<int> &indices,
                      const Eigen::MatrixXd &positions) {
  std::set<int> used;
  std::vector<int> slice;
  std::vector<int> new_indices;
  slice.reserve(indices.size());
  new_indices.reserve(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    if (!used.insert(indices[i]).second)
      continue;
    slice.push_back(i);
    new_indices.push_back(indices[i]);
  }
  objective->set_knowns(convert::eigen_vec_map(new_indices),
                        positions(slice, Eigen::placeholders::all));
  return *this;
}

ARAP &ARAP::add_soft_constraints(double weight, const std::vector<int> &indices,
                                 const Eigen::MatrixXd &positions) {
  objective->add_weighted_equations_with_rhs(
      indices.size(), [&](int i, auto &x, auto add_eq) {
        add_eq(weight, x(indices[i]), positions.row(i));
      });
  return *this;
}

void ARAP::set_soft_constraints_weight(double weight) {
  objective->get_quadratic_term(SOFT_ENERGY_TERM_INDEX).set_weight(weight);
}

void ARAP::set_edges_weight(double weight) {
  objective->get_quadratic_term(EDGES_ENERGY_INDEX).set_weight(weight);
}

ARAP &ARAP::precompute() {
  objective->prefactor();
  return *this;
}

static Eigen::Matrix3d closest_rotation(const Eigen::Matrix3d &mat) {
  Eigen::Matrix3d U, V;
  Eigen::Vector3d S;
  igl::svd3x3(mat, U, S, V);
  // IGL's svd3x3 guarantees that U, V are rotation matrices.
  return U * V.transpose();
}

Eigen::MatrixXd ARAP::solve(const Eigen::MatrixXd &X, bool zero_rhs) {
  Eigen::MatrixXd rhs(objective->get_quadratic_term(0).A.rows(), 3);
  if (zero_rhs) {
    rhs.setZero();
    return objective->solve(rhs);
  }
// Calculate the RHS.
#pragma omp parallel for
  for (auto &one_ring : one_rings) {
    auto edges = X(one_ring.v1, Eigen::placeholders::all) -
                 X(one_ring.v0, Eigen::placeholders::all);
    auto old_edges = one_ring.original_edges();
    // Find best fitting rotation from original to current edges.
    Eigen::Matrix3d R = closest_rotation(old_edges.transpose() * edges);
    // Add the rotated original edges to the RHS.
    one_ring.edges(rhs) = old_edges * R;
  }
  objective->get_quadratic_term(ARAP_TERM_INDEX).set_b(rhs);

  Eigen::MatrixXd edge_rhs(edges.rows(), 3);
#pragma omp parallel for
  for (int i = 0; i < edges.rows(); i++) {
    double orig_edge_length =
        (mesh.V.row(edges(i, 1)) - mesh.V.row(edges(i, 0))).norm();

    Eigen::RowVector3d cur_edge = X.row(edges(i, 1)) - X.row(edges(i, 0));
    double cur_edge_length = cur_edge.norm();
    edge_rhs.row(i) =
        cur_edge * shrink_factor * orig_edge_length / (cur_edge_length + 1e-8);
  }
  objective->get_quadratic_term(EDGES_ENERGY_INDEX).set_b(edge_rhs);

  return objective->solve();
}

ARAP &ARAP::set_fixed_positions(const Eigen::MatrixXd &fixed_positions) {
  objective->set_known_values(fixed_positions);
  return *this;
}

void ARAP::build_one_rings() {

  one_rings.resize(mesh.nv());
  auto add_edge = [&](int dest, int v0, int v1, double cot_weight) {
    one_rings[dest].v0.push_back(v0);
    one_rings[dest].v1.push_back(v1);
    one_rings[dest].edges_vec.push_back(mesh.V.row(v1) - mesh.V.row(v0));
    one_rings[dest].cot_weights.push_back(cot_weight);
  };

  for (auto &e : mesh.edges) {
    double cot_weight = e.cot_theta();
    double twin_cot_weight = (e.twin() ? e.twin()->cot_theta() : 0);
    int v0 = e.vi, v1 = e.next()->vi;
    // Add the edge to the neighborhoods of the triangle vertices.
    add_edge(e.vi, v0, v1, cot_weight + twin_cot_weight);
    if (!e.twin())
      add_edge(e.next()->vi, v1, v0, cot_weight);
    // Rim edge.
    add_edge(e.prev()->vi, v0, v1, cot_weight);
  }
  int current_row = 0;
  for (int v = 0; v < one_rings.size(); v++) {
    one_rings[v].start_row = current_row;
    current_row += one_rings[v].num_edges();
  }
}

} // namespace geom