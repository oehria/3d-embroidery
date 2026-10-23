#include "utils.h"

namespace utils {
bool is_inside_polygon(const Eigen::MatrixXd &poly,
                       const Eigen::Vector2d &point) {
  // Check if the point is inside the polygon defined by poly.
  // poly is an Nx2 matrix where each row is a vertex of the polygon.
  int n = poly.rows();
  bool inside = false;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    if (((poly(i, 1) > point.y()) != (poly(j, 1) > point.y())) &&
        (point.x() < (poly(j, 0) - poly(i, 0)) * (point.y() - poly(i, 1)) /
                             (poly(j, 1) - poly(i, 1)) +
                         poly(i, 0))) {
      inside = !inside;
    }
  }
  return inside;
}

bool on_segment(const Eigen::Vector2d &p, const Eigen::Vector2d &q,
                const Eigen::Vector2d &r) {
  // Check if point q lies on segment pr
  return q.x() <= std::max(p.x(), r.x()) && q.x() >= std::min(p.x(), r.x()) &&
         q.y() <= std::max(p.y(), r.y()) && q.y() >= std::min(p.y(), r.y());
}

bool segments_intersect(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                        const Eigen::Vector2d &q1, const Eigen::Vector2d &q2) {
  auto orientation = [](const Eigen::Vector2d &p, const Eigen::Vector2d &q,
                        const Eigen::Vector2d &r) -> int {
    double val =
        (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y());
    if (std::abs(val) < 1e-10)
      return 0;               // colinear (with floating point tolerance)
    return (val > 0) ? 1 : 2; // 1 = clockwise, 2 = counterclockwise
  };

  int o1 = orientation(p1, p2, q1);
  int o2 = orientation(p1, p2, q2);
  int o3 = orientation(q1, q2, p1);
  int o4 = orientation(q1, q2, p2);

  // General case
  if (o1 != o2 && o3 != o4)
    return true;

  // Special cases - check if any endpoint lies on the other segment
  if (o1 == 0 && on_segment(p1, q1, p2))
    return true;
  if (o2 == 0 && on_segment(p1, q2, p2))
    return true;
  if (o3 == 0 && on_segment(q1, p1, q2))
    return true;
  if (o4 == 0 && on_segment(q1, p2, q2))
    return true;

  return false;
}

bool segment_intersects_poly(const Eigen::MatrixXd &poly,
                             const Eigen::Vector2d &p1,
                             const Eigen::Vector2d &p2) {
  Eigen::Vector2d safe_p1 = p1 + 1e-4 * (p2 - p1).normalized();
  Eigen::Vector2d safe_p2 = p2 + 1e-4 * (p1 - p2).normalized();
  // Check if the segment p1p2 intersects with any edge of the polygon.
  int n = poly.rows();
  for (int i = 0; i < n; i++) {
    Eigen::Vector2d q1 = poly.row(i);
    Eigen::Vector2d q2 = poly.row((i + 1) % n);
    if (segments_intersect(safe_p1, safe_p2, q1, q2)) {
      return true; // Found an intersection
    }
  }
  return false;
}

} // namespace utils