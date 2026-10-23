#pragma once
#include <Eigen/Eigen>
#include <functional>
#include <memory>
#include <vector>

#include "LinearExpressionBase.h"

namespace utils {

class QuadraticObjectiveD {
  // add_equation(weight, linear_expression).
  using VarType = double;
  using AddRhsFunc = std::function<void(const VarType &)>;
  using SparseMatrix = Eigen::SparseMatrix<double>;
  using MatType = Eigen::MatrixX<double>;

  enum SolverType { CHOL, LU, QR };

  // Hard constraints of the form Cx = d.
  struct HardConstraint {
    Eigen::SparseMatrix<double> C;
    Eigen::MatrixXd d;
  };
  // Quadratic term of the form weight * (Ax - b)^T * W * (Ax - b).
  struct QuadraticTerm {
    QuadraticTerm(QuadraticObjectiveD *objective, const SparseMatrix &A,
                  const Eigen::VectorXd &weights = Eigen::VectorXd(),
                  double weight = 1, const MatType &b = MatType());

    SparseMatrix A;
    double weight = 1;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W =
        Eigen::VectorXd::Ones(A.rows()).asDiagonal();
    MatType b_mat;

    QuadraticObjectiveD *objective;

    // Cached values.
    // Cached matrix of weight * unknown(A).transpose() * W.
    SparseMatrix weight_unknownA_W;
    // Cached matrix of weight * unknown(A).transpose() * W * known(A).
    SparseMatrix weight_unknownA_W_knownA;

    // Cached rhs.
    MatType rhs_b;
    MatType rhs_known;

    QuadraticTerm *set_b(const MatType &b_mat) {
      this->b_mat = b_mat;
      rhs_b = weight_unknownA_W * b_mat;
      return this;
    }
    QuadraticTerm *set_b(std::vector<Eigen::VectorXd> &rhs) {
      this->b_mat.resize(rhs.size(), rhs[0].size());
      for (int i = 0; i < rhs.size(); i++) {
        this->b_mat.row(i) = rhs[i];
      }
      rhs_b = weight_unknownA_W * b_mat;
      return this;
    }
    QuadraticTerm *update_b(int num_elements,
                            const std::function<void(int, const AddRhsFunc &)>
                                &handle_element_func);
    QuadraticTerm *set_b(int num_elements,
                         const std::function<MatType(int)> &b_func);
    QuadraticTerm *set_element_weight(const Eigen::VectorXd &diagonal_weights) {
      W = diagonal_weights.asDiagonal();
      objective->update_cache(this);
      return this;
    }
    QuadraticTerm *set_weight(double w) {
      weight = w;
      objective->update_cache(this);
      return this;
    }
  };

public:
  QuadraticObjectiveD(int n, int dim = -1);
  QuadraticObjectiveD(const std::vector<int> &shape, int dim = -1);

  explicit QuadraticObjectiveD(const QuadraticObjectiveD &other);

  /*
   * Add a quadratic energy of the form:
   *             weight * (Ax - b)^T * W * (Ax - b)
   */
  QuadraticTerm *
  add_quadratic_energy(const SparseMatrix &A,
                       const Eigen::VectorXd &W = Eigen::VectorXd(),
                       double weight = 1);

  QuadraticTerm *add_weighted_equations(int num_elements,
                                        const auto &handle_element_func) {
    auto factory = [&](int i, int j = 0) {
      return LinearExpressionVariableD(i + variable_shape[0] * j);
    };
    std::vector<Eigen::Triplet<VarType>> triplets;
    std::vector<double> weights;
    int cur_row = 0;
    auto add_equation = [&](double w, const auto &t) {
      t.append_to(cur_row++, triplets);
      weights.push_back(w);
    };
    for (int i = 0; i < num_elements; i++) {
      handle_element_func(i, factory, add_equation);
    }
    Eigen::SparseMatrix<VarType> A(cur_row, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::Map<Eigen::VectorXd> w(weights.data(), weights.size());
    return add_quadratic_energy(A, w);
  }

  static Eigen::VectorXd convert_to_vecxd(double x) {
    return Eigen::VectorXd::Constant(1, x);
  }
  static const Eigen::VectorXd &convert_to_vecxd(const Eigen::VectorXd &x) {
    return x;
  }

  QuadraticTerm *
  add_weighted_equations_with_rhs(int num_elements,
                                  const auto &handle_element_func) {
    auto factory = [&](int i, int j = 0) {
      return LinearExpressionVariableD(i + variable_shape[0] * j);
    };
    std::vector<Eigen::Triplet<VarType>> triplets;
    std::vector<double> weights;
    std::vector<Eigen::VectorXd> rhs;
    int cur_row = 0;
    auto add_equation = [&](double w, const auto &t, const auto &b) {
      t.append_to(cur_row++, triplets);
      weights.push_back(w);
      rhs.push_back(convert_to_vecxd(b));
    };
    for (int i = 0; i < num_elements; i++) {
      handle_element_func(i, factory, add_equation);
    }
    Eigen::SparseMatrix<VarType> A(cur_row, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::Map<Eigen::VectorXd> w(weights.data(), weights.size());
    auto res = add_quadratic_energy(A, w);
    res->set_b(rhs);
    return res;
  }

  // Add hard constraints of the form Cx = d
  QuadraticObjectiveD &add_hard_constraints(const SparseMatrix &C,
                                            const MatType &d);

  QuadraticObjectiveD &add_hard_constraints(int num_elements,
                                            const auto &handle_element_func) {
    auto factory = [&](int i, int j = 0) {
      return LinearExpressionVariableD(i + variable_shape[0] * j);
    };
    std::vector<Eigen::Triplet<VarType>> triplets;
    int cur_row = 0;
    auto add_equation = [&](const auto &t) {
      t.append_to(cur_row++, triplets);
    };
    for (int i = 0; i < num_elements; i++) {
      handle_element_func(i, factory, add_equation);
    }
    Eigen::SparseMatrix<VarType> A(cur_row, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    add_hard_constraints(A, Eigen::MatrixXd::Zero(cur_row, n_cols()));
    return *this;
  }

  QuadraticObjectiveD &update_known_values(const MatType &d, int offset = 0);

  // Set the known indiecs.
  QuadraticObjectiveD &set_known_indices(const Eigen::VectorXi &indices);
  // Set known values.
  QuadraticObjectiveD &set_known_values(const MatType &d);
  // Set known indices and values.
  QuadraticObjectiveD &set_knowns(const Eigen::VectorXi &indices,
                                  const MatType &d);

  /**
   * Prefactorized the system.
   */
  QuadraticObjectiveD &prefactor();

  MatType solve();
  MatType solve(const MatType &b);
  int rank() const;

  QuadraticTerm &get_quadratic_term(int index) {
    return *_quadratic_terms[index];
  }

  inline int n_free() const { return n - _known_indices.size(); }
  inline int n_cols() const {
    if (dim > 0)
      return dim;
    if (_quadratic_terms.size() > 0)
      return _quadratic_terms[0]->b_mat.cols();
    return 1;
  }

  double get_sqrd_error(const MatType &x) const;
  SparseMatrix build_lhs_matrix() const;

private:
  void update_known_values_rhs();
  void update_cache(QuadraticTerm *term);

  void update_cache();
  SparseMatrix known(const SparseMatrix &mat);

  SparseMatrix unknown(const SparseMatrix &mat) const;

private:
  int n;
  std::vector<int> variable_shape;
  int dim;
  // Equality constraints.
  std::vector<int> _unknown_indices;
  std::vector<int> _known_indices;
  MatType _knowns_vals;

  std::vector<std::unique_ptr<QuadraticTerm>> _quadratic_terms;
  std::vector<HardConstraint> _hard_constraints;

  bool requires_precompute = true;
  SolverType _solver_type = SolverType::CHOL;
  Eigen::SimplicialLDLT<SparseMatrix> _psd_solver;
  Eigen::SparseLU<SparseMatrix> _lu_solver;
};

} // namespace utils
