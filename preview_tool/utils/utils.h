#pragma once
#include <Eigen/Eigen>

namespace utils {
bool is_inside_polygon(const Eigen::MatrixXd &poly,
                       const Eigen::Vector2d &point);
bool segment_intersects_poly(const Eigen::MatrixXd &poly,
                             const Eigen::Vector2d &p1,
                             const Eigen::Vector2d &p2);
bool segments_intersect(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                        const Eigen::Vector2d &q1, const Eigen::Vector2d &q2);

template <typename T>
Eigen::MatrixX<T> slice_row(const Eigen::MatrixX<T> &verts, int row) {
  Eigen::MatrixX<T> res(verts.rows() - 1, verts.cols());
  for (int i = 0; i < row; i++) {
    res.row(i) = verts.row(i);
  }
  for (int i = row + 1; i < verts.rows(); i++) {
    res.row(i - 1) = verts.row(i);
  }
  return res;
}


template <typename T>
Eigen::SparseMatrix<T> spcat(
    const std::vector<std::vector<Eigen::SparseMatrix<T>>>& mats) {
  int n_rows = 0, n_cols = 0;
  for (auto& mat : mats) {
    n_rows += mat[0].rows();
  }
  for (auto& mat : mats[0]) {
    n_cols += mat.cols();
  }

  std::vector<Eigen::Triplet<T>> triplets;
  int row_offset = 0, col_offset = 0;
  for (auto& mat : mats) {
    col_offset = 0;
    for (auto& submat : mat) {
      for (int k = 0; k < submat.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(submat, k); it;
             ++it) {
          triplets.push_back(Eigen::Triplet<T>(
              it.row() + row_offset, it.col() + col_offset, it.value()));
        }
      }
      col_offset += submat.cols();
    }
    row_offset += mat[0].rows();
  }
  Eigen::SparseMatrix<T> res(n_rows, n_cols);
  res.setFromTriplets(triplets.begin(), triplets.end());
  return res;
}

} // namespace utils