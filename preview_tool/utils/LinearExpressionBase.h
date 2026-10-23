#pragma once

#include <Eigen/Eigen>
#include <complex>
#include <iostream>
#include <string>

namespace utils {

template <typename Derived, typename T> class LinearExpressionBase {
public:
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    static_cast<const Derived *>(this)->append_to(row, vec);
  }

  auto get_constant_part() const {
    return static_cast<const Derived *>(this)->get_constant_part();
  }

  template<typename G>
  decltype(auto) standardize(const G& x) const {
    // If it's an eigen expression.
    if constexpr (std::is_base_of_v<Eigen::EigenBase<G>, G>) {
      return x.array();
    } else {
      return x;
    }
  }

  std::string print() const {
    return static_cast<const Derived *>(this)->print();
  }

  friend std::ostream &operator<<(std::ostream &s,
                                  LinearExpressionBase const &expr) {
    return s << "LinearExpression(" << expr.print() << ")";
  }
};

template <typename T, typename G>
class LinearExpressionConstant
    : public LinearExpressionBase<LinearExpressionConstant<T, G>, T> {
public:
  G _value;

  explicit LinearExpressionConstant(const G &value) : _value(value) {}
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {}
  void append_to(int row, const T &coef,
                 std::vector<Eigen::Triplet<T>> &vec) const {}

  auto get_constant_part(const T &coef = 1.0) const { return coef * _value; }

  std::string print(const T &coef = T(1.0)) const {
    std::ostringstream out(std::ios::fixed);
    out.precision(2);
    out << coef * _value;
    return out.str();
  }
};

template <typename T>
class LinearExpressionVariable
    : public LinearExpressionBase<LinearExpressionVariable<T>, T> {
public:
  int _index;

  LinearExpressionVariable(int index) : _index(index) {}
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    vec.emplace_back(row, _index, 1);
  }
  void append_to(int row, const T &coef,
                 std::vector<Eigen::Triplet<T>> &vec) const {
    vec.emplace_back(row, _index, coef);
  }
  auto get_constant_part(const T &coef = 1.0) const { return T(0.0); }
  std::string to_str(const T &val) const {
    std::ostringstream out(std::ios::fixed);
    out.precision(2);
    out << val;
    return out.str();
  }

  std::string print(const T &coef = T(1.0)) const {
    return (coef != 1.0 ? (to_str(coef) + " * ") : std::string()) + "x_" +
           std::to_string(_index);
  }
};

template <typename Derived1, typename Derived2, typename T>
class LinearExpressionAdd
    : public LinearExpressionBase<LinearExpressionAdd<Derived1, Derived2, T>,
                                  T> {
public:
  const Derived1 _derived1;
  const Derived2 _derived2;

  LinearExpressionAdd(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2) {}
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.append_to(row, vec);
    _derived2.append_to(row, vec);
  }
  void append_to(int row, const T &coef,
                 std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.append_to(row, coef, vec);
    _derived2.append_to(row, coef, vec);
  }
  auto get_constant_part(const T &coef = 1.0) const {
    return this->standardize(_derived1.get_constant_part(coef)) +
           this->standardize(_derived2.get_constant_part(coef));
  }
  std::string print(const T &coef = T(1.0)) const {
    return _derived1.print(coef) + " + " + _derived2.print(coef);
  }
};

template <typename Derived1, typename Derived2, typename T>
class LinearExpressionSub
    : public LinearExpressionBase<LinearExpressionSub<Derived1, Derived2, T>,
                                  T> {
public:
  const Derived1 _derived1;
  const Derived2 _derived2;

  LinearExpressionSub(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2) {}
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.append_to(row, vec);
    _derived2.append_to(row, -1, vec);
  }
  void append_to(int row, const T &coef,
                 std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.append_to(row, coef, vec);
    _derived2.append_to(row, -coef, vec);
  }
  auto get_constant_part(const T &coef = 1.0) const {
    return this->standardize(_derived1.get_constant_part(coef)) -
           this->standardize(_derived2.get_constant_part(coef));
  }
  std::string print(const T &coef = T(1.0)) const {
    return _derived1.print(coef) + " + " + _derived2.print(-coef);
  }
};

template <typename Derived, typename T>
class LinearExpressionScalarMul
    : public LinearExpressionBase<LinearExpressionScalarMul<Derived, T>, T> {
public:
  const Derived _derived;
  T _scalar;
  LinearExpressionScalarMul(T scalar, const Derived &derived)
      : _scalar(scalar), _derived(derived) {}
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived.append_to(row, _scalar, vec);
  }
  void append_to(int row, const T &coef,
                 std::vector<Eigen::Triplet<T>> &vec) const {
    _derived.append_to(row, coef * _scalar, vec);
  }
  auto get_constant_part(const T &coef = 1.0) const {
    return _derived.get_constant_part(coef * _scalar);
  }
  std::string print(const T &coef = T(1.0)) const {
    return _derived.print(coef * _scalar);
  }
};

template <typename Derived1, typename Derived2, typename T>
LinearExpressionAdd<Derived1, Derived2, T>
operator+(LinearExpressionBase<Derived1, T> const &derived1,
          LinearExpressionBase<Derived2, T> const &derived2) {
  return LinearExpressionAdd<Derived1, Derived2, T>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <typename Derived1, typename Derived2, typename T>
LinearExpressionSub<Derived1, Derived2, T>
operator-(LinearExpressionBase<Derived1, T> const &derived1,
          LinearExpressionBase<Derived2, T> const &derived2) {
  return LinearExpressionSub<Derived1, Derived2, T>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <typename Derived, typename T, typename G>
LinearExpressionScalarMul<Derived, T>
operator*(const G &scalar, LinearExpressionBase<Derived, T> const &derived) {
  return LinearExpressionScalarMul<Derived, T>(
      scalar, *static_cast<const Derived *>(&derived));
}

template <typename Derived, typename T, typename G>
LinearExpressionScalarMul<Derived, T>
operator*(LinearExpressionBase<Derived, T> const &derived, const G &scalar) {
  return LinearExpressionScalarMul<Derived, T>(
      scalar, *static_cast<const Derived *>(&derived));
}

template <typename Derived, typename T, typename G,
          std::enable_if_t<!std::is_base_of_v<LinearExpressionBase<G, T>, G>,
                           int> = 0>
auto operator+(const G &scalar,
               LinearExpressionBase<Derived, T> const &derived) {
  return LinearExpressionConstant<T, G>(scalar) +
         *static_cast<const Derived *>(&derived);
}
template <typename Derived, typename T, typename G,
          std::enable_if_t<!std::is_base_of_v<LinearExpressionBase<G, T>, G>,
                           int> = 0>
auto operator+(LinearExpressionBase<Derived, T> const &derived,
               const G &scalar) {
  return *static_cast<const Derived *>(&derived) +
         LinearExpressionConstant<T, G>(scalar);
}

template <typename Derived, typename T, typename G,
          std::enable_if_t<!std::is_base_of_v<LinearExpressionBase<G, T>, G>,
                           int> = 0>
auto operator-(const G &scalar,
               LinearExpressionBase<Derived, T> const &derived) {
  return LinearExpressionConstant<T, G>(scalar) -
         *static_cast<const Derived *>(&derived);
}
template <typename Derived, typename T, typename G,
          std::enable_if_t<!std::is_base_of_v<LinearExpressionBase<G, T>, G>,
                           int> = 0>
auto operator-(LinearExpressionBase<Derived, T> const &derived,
               const G &scalar) {
  return *static_cast<const Derived *>(&derived) -
         LinearExpressionConstant<T, G>(scalar);
}

using LinearExpressionVariableD = LinearExpressionVariable<double>;
using LinearExpressionVariableC =
    LinearExpressionVariable<std::complex<double>>;

} // namespace utils
