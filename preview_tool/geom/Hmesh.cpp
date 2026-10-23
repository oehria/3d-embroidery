#include "Hmesh.h"

#include <assert.h>

#include <queue>
#include <set>
#include <vector>

#include "../utils/conversions.h"
#include "../utils/utils.h"

namespace geom {

Hmesh::Edge *Hmesh::Face::edge() const { return &mesh->edge(ei); }

Eigen::Vector2d Hmesh::Face::to_local_coords(const Eigen::Vector3d &vec) const {
  return frame.transpose() * (vec - edge()->origin()->coords());
}
Eigen::Vector2d
Hmesh::Face::bary_to_local_coords(const Eigen::Vector3d &bary) const {
  return to_local_coords(bary(0) * mesh->V.row(mesh->F[index][0]) +
                         bary(1) * mesh->V.row(mesh->F[index][1]) +
                         bary(2) * mesh->V.row(mesh->F[index][2]));
}
Eigen::Vector3d Hmesh::Face::to_bary_coords(const Eigen::Vector3d &vec) const {
  Eigen::Matrix<double, 3, 2> A;
  A << edge()->prev()->vec(), -edge()->next()->vec();
  Eigen::Vector3d res;
  res.head(2) = (A.transpose() * A).inverse() * A.transpose() *
                (vec - edge()->prev()->origin()->coords());
  res(2) = 1 - res(0) - res(1);
  return res;
}
Eigen::Vector3d
Hmesh::Face::to_global_coords(const Eigen::Vector2d &vec) const {
  return frame * vec + edge()->origin()->coords();
}

Eigen::VectorXd Hmesh::Face::center() const {
  Edge *orig_e = edge();
  Edge *e = orig_e;
  Eigen::VectorXd center = Eigen::VectorXd::Zero(mesh->V.cols());
  int count = 0;
  do {
    center += e->origin()->coords();
    count++;
    e = e->next();
  } while (e != orig_e);
  return center / count;
}

double Hmesh::Face::area() const {
  auto &face = mesh->F[index];
  if (mesh->V.cols() == 3) {
    Eigen::Vector3d vec_area = Eigen::Vector3d::Zero();
    for (int i = 0; i < nf(); i++) {
      vec_area +=
          ((Eigen::Vector3d)mesh->V.row(face[i]))
              .cross((Eigen::Vector3d)mesh->V.row(face[(i + 1) % nf()]));
    }
    return vec_area.norm() / 2;
  }
  double area = 0;
  for (int i = 1; i < nf() - 1; i++) {
    Eigen::Vector2d v0 = mesh->V.row(face[0]);
    Eigen::Vector2d v1 = mesh->V.row(face[i]);
    Eigen::Vector2d v2 = mesh->V.row(face[i + 1]);
    Eigen::Vector2d e0 = v1 - v0, e1 = v2 - v0;
    area += 0.5 * std::abs(e0.x() * e1.y() - e0.y() * e1.x());
  }
  return area;
}

Eigen::Vector3d Hmesh::Face::normal() const {
  auto &face = mesh->F[index];
  Eigen::Vector3d vec_area = Eigen::Vector3d::Zero();
  for (int i = 0; i < nf(); i++) {
    vec_area += ((Eigen::Vector3d)mesh->V.row(face[i]))
                    .cross((Eigen::Vector3d)mesh->V.row(face[(i + 1) % nf()]));
  }
  return vec_area.normalized();
}

Hmesh::Edge *Hmesh::Vertex::edge() const { return &mesh->edge(ei); }

Eigen::VectorXd Hmesh::Vertex::coords() const { return mesh->V.row(index); }
Eigen::VectorXd Hmesh::Vertex::normal() const {
  Eigen::VectorXd normal = Eigen::VectorXd::Zero(mesh->V.cols());
  double total_area = 0;
  Edge *e = edge();
  do {
    normal += e->face()->normal() * e->face()->area();
    total_area += e->face()->area();
    e = e->prev()->twin();
  } while (e != edge() && e != nullptr);
  return (normal / total_area).normalized();
}

bool Hmesh::Face::is_boundary() {
  Edge *orig_e = edge();
  Edge *e = orig_e;
  do {
    if (!e->twin())
      return true;
    e = e->next();
  } while (e != orig_e);
  return false;
}

static std::vector<std::vector<int>> convert_faces(const Eigen::MatrixXi &F,
                                                   const Eigen::VectorXi &D) {
  std::vector<std::vector<int>> faces(F.rows());
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < D(i); j++) {
      faces[i].push_back(F(i, j));
    }
  }
  return faces;
}

Hmesh::Hmesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
             const Eigen::VectorXi &D)
    : Hmesh(V, convert_faces(F, D)) {}

Hmesh::Hmesh(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &F)
    : V(V), F(F) {
  int nv = V.rows(), nf = F.size();
  verts.reserve(nv);
  faces.reserve(nf);
  int num_edges = 0;
  for (int i = 0; i < F.size(); i++) {
    num_edges += F[i].size();
  }
  edges.reserve(num_edges);
  verts_edges.resize(nv);

  // Push vertices.
  for (int i = 0; i < nv; i++) {
    verts.push_back(Vertex{.index = i, .ei = -1, .mesh = this});
  }

  // Faces and edges.
  int cur_edge_index = 0;
  for (int i = 0; i < nf; i++) {
    // Push face.
    faces.push_back(Face{.index = i, .ei = cur_edge_index, .mesh = this});
    // Push face edges.
    int verts_per_face = F[i].size();
    for (int j = 0; j < verts_per_face; j++) {
      int vi = F[i][j], vip1 = F[i][(j + 1) % verts_per_face];
      int ei = cur_edge_index + j;
      // Edge ei is from vertex vi to vip1.
      edges.push_back(
          Edge{.index = ei,
               .vi = vi,
               .fi = i,
               .fvi = j,
               .ti = -1,
               .ni = cur_edge_index + (j + 1) % verts_per_face,
               .pi = cur_edge_index + (j - 1 + verts_per_face) % verts_per_face,
               .mesh = this});
      // Update vertex edge.
      if (vertex(vi).ei == -1) {
        verts[vi].ei = ei;
      }
      // Update verts_edges.
      verts_edges[vi][vip1] = ei;
      auto it = verts_edges[vip1].find(vi);
      if (it != verts_edges[vip1].end()) {
        edges[ei].ti = it->second;
        edges[it->second].ti = ei;
      }
    }
    cur_edge_index += verts_per_face;
  }
  // Calculate frames.
  if (V.cols() == 3) {
    for (auto &face : faces) {
      face.frame.col(0) = face.edge()->vec().normalized();
      face.frame.col(1) = face.normal().cross(face.frame.col(0)).normalized();
    }
    for (auto &edge : edges) {
      if (edge.twin()) {
        edge.frame_rot =
            edge.twin()->face()->frame.transpose() * edge.face()->frame;
      }
    }
  }
  // Make the boundary edge the first in each vertex.
  for (int i = 0; i < nv; i++) {
    if (verts[i].ei == -1)
      continue;

    Edge *e = verts[i].edge();
    if (!e || !e->twin()) {
      continue;
    }
    do {
      e = e->twin()->next();
    } while (e != verts[i].edge() && e->twin());
    verts[i].ei = e->index;
  }
}

Hmesh &Hmesh::operator=(const Hmesh &other) {
  V = other.V;
  F = other.F;
  verts = other.verts;
  faces = other.faces;
  edges = other.edges;
  verts_edges = other.verts_edges;
  for (auto &v : verts) {
    v.mesh = this;
  }
  for (auto &f : faces) {
    f.mesh = this;
  }
  for (auto &e : edges) {
    e.mesh = this;
  }
  return *this;
}

inline std::vector<std::vector<int>> to_vector(const Eigen::MatrixXi &F) {
  std::vector<std::vector<int>> F_vec(F.rows());
  for (int i = 0; i < F.rows(); i++) {
    F_vec[i].resize(F.cols());
    for (int j = 0; j < F.cols(); j++) {
      F_vec[i][j] = F(i, j);
    }
  }
  return std::move(F_vec);
}

Hmesh::Hmesh(const Hmesh &other)
    : V(other.V), F(other.F), verts(other.verts),
      faces(other.faces), edges(other.edges), verts_edges(other.verts_edges) {
  for (auto &v : verts) {
    v.mesh = this;
  }
  for (auto &f : faces) {
    f.mesh = this;
  }
  for (auto &e : edges) {
    e.mesh = this;
  }
}

Hmesh::Hmesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
    : Hmesh(V, to_vector(F)) {}


Hmesh::VertexStarIterator::VertexStarIterator(Edge *e)
    : VertexStarIterator(e, true) {}
Hmesh::VertexStarIterator::VertexStarIterator(Edge *e, bool is_begin)
    : e(e), is_begin(is_begin) {}
const Hmesh::VertexStarIterator &Hmesh::VertexStarIterator::operator++() {
  e = e->prev()->twin();
  is_begin = false;
  return *this;
}
bool Hmesh::VertexStarIterator::operator==(
    const VertexStarIterator &other) const {
  return !e || !e->face() || (e == other.e && is_begin == other.is_begin);
}
bool Hmesh::VertexStarIterator::operator!=(
    const VertexStarIterator &other) const {
  return !(operator==(other));
}

Hmesh::Vertex *Hmesh::Edge::origin() const { return &mesh->vertex(vi); }
Hmesh::Face *Hmesh::Edge::face() const { return &mesh->face(fi); }
Hmesh::Edge *Hmesh::Edge::prev() const { return &mesh->edge(pi); }
Hmesh::Edge *Hmesh::Edge::next() const { return &mesh->edge(ni); }
Hmesh::Edge *Hmesh::Edge::twin() const {
  if (ti == -1)
    return nullptr;
  return &mesh->edge(ti);
}

double Hmesh::Edge::cos_theta() {
  return -next()->vec().normalized().dot(prev()->vec().normalized());
}
double Hmesh::Edge::sin_theta() {
  return static_cast<Eigen::Vector3d>(next()->vec().normalized())
      .cross(static_cast<Eigen::Vector3d>(prev()->vec().normalized()))
      .norm();
}
double Hmesh::Edge::cot_theta() {
  // return cos_theta() / sin_theta();
  Eigen::Vector3d prev_vec = prev()->vec();
  Eigen::Vector3d next_vec = next()->vec();
  return -prev_vec.dot(next_vec) / prev_vec.cross(next_vec).norm();
}
double Hmesh::Edge::theta() { return acos(cos_theta()); }
Hmesh::Vertex *Hmesh::Edge::target() const { return next()->origin(); }
Eigen::VectorXd Hmesh::Edge::vec() const {
  return next()->origin()->coords() - origin()->coords();
}
Eigen::Vector2d Hmesh::Edge::local_vec() const {
  return face()->frame.transpose() * vec();
}
double Hmesh::Edge::length() { return vec().norm(); }

std::vector<std::set<int>> Hmesh::vertex_neighbors() {
  std::vector<std::set<int>> neighbors(V.rows());
  for (int i = 0; i < F.size(); i++) {
    for (int j = 0; j < F[i].size(); j++) {
      int v0 = F[i][j], v1 = F[i][(j + 1) % F[i].size()];
      neighbors[v0].insert(v1);
      neighbors[v1].insert(v0);
    }
  }
  return neighbors;
}

Eigen::Vector3d Hmesh::Edge::normal() const {
  if (!twin())
    return face()->normal();
  return (face()->normal() * face()->area() +
          twin()->face()->normal() * twin()->face()->area())
      .normalized();
}

Hmesh::Point Hmesh::Point::operator+(const Eigen::Vector2d &vec) const {
  return add(vec).first;
}

std::pair<geom::Hmesh::Point, bool>
geom::Hmesh::Point::add(const Eigen::Vector2d &vec) const {
  Hmesh::Point res(*this);
  Eigen::Vector2d dir = vec.normalized();
  double len = vec.norm();
  while (len > 1e-6) {
    // Find closest edge.
    double min_t = std::numeric_limits<double>::max();
    Edge *min_edge = nullptr;
    auto &f = mesh->face(res.face);
    auto origin = f.edge()->origin()->coords();
    Edge *e = f.edge();
    for (int i = 0; i < 3; i++) {
      Eigen::Vector2d p0 =
          f.frame.transpose() * (e->origin()->coords() - origin);
      Eigen::Vector2d p1 =
          f.frame.transpose() * (e->next()->origin()->coords() - origin);
      Eigen::Vector2d d = (p1 - p0).normalized();
      // res.local_coords + t * dir = p0 + s * d
      // [dir, -d] \ [p0 - res.local_doords]
      double det = dir.y() * d.x() - dir.x() * d.y();
      // If det > 0, we're walking 'into' this triangle (coming from this edge).
      if (-det > 1e-6) {
        double t = Eigen::RowVector2d(-d.y() / det, d.x() / det) *
                   (p0 - res.local_coords);
        if (t < min_t) {
          min_t = t;
          min_edge = e;
        }
      }
      e = e->next();
    }

    // If we don't need to cross another triangle.
    if (min_t >= len) {
      // Update coords and return.
      res.local_coords += len * dir;
      return {res, true};
    }
    // Otherwise, update the direction, length, and new local_coords.
    if (min_edge->twin() == nullptr) {
      res.local_coords += min_t * dir;
      return {res, false};
    }
    res.face = min_edge->twin()->fi;
    // Optimize this...
    res.local_coords = mesh->face(res.face).to_local_coords(
        f.to_global_coords(res.local_coords + min_t * dir));
    dir = (min_edge->frame_rot * dir).normalized();
    len -= min_t;
  }
  return {res, true};
}

std::pair<geom::Hmesh::Point, Eigen::Vector2d>
geom::Hmesh::Point::transport(const Eigen::Vector2d &vec,
                               const Eigen::Vector2d &transport_vec) const {
  Hmesh::Point res(*this);
  Eigen::Vector2d dir = vec.normalized();
  Eigen::Vector2d transported = transport_vec.normalized();
  double len = vec.norm();
  while (len > 1e-6) {
    // Find closest edge.
    double min_t = std::numeric_limits<double>::max();
    Edge *min_edge = nullptr;
    auto &f = mesh->face(res.face);
    auto origin = f.edge()->origin()->coords();
    Edge *e = f.edge();
    for (int i = 0; i < 3; i++) {
      Eigen::Vector2d p0 =
          f.frame.transpose() * (e->origin()->coords() - origin);
      Eigen::Vector2d p1 =
          f.frame.transpose() * (e->next()->origin()->coords() - origin);
      Eigen::Vector2d d = (p1 - p0).normalized();
      // res.local_coords + t * dir = p0 + s * d
      // [dir, -d] \ [p0 - res.local_doords]
      double det = dir.y() * d.x() - dir.x() * d.y();
      // If det > 0, we're walking 'into' this triangle (coming from this edge).
      if (-det > 1e-6) {
        double t = Eigen::RowVector2d(-d.y() / det, d.x() / det) *
                   (p0 - res.local_coords);
        if (t < min_t) {
          min_t = t;
          min_edge = e;
        }
      }
      e = e->next();
    }

    // If we don't need to cross another triangle.
    if (min_t >= len) {
      // Update coords and return.
      res.local_coords += len * dir;
      return {res, transported};
    }
    // Otherwise, update the direction, length, and new local_coords.
    if (min_edge->twin() == nullptr) {
      res.local_coords += min_t * dir;
      return {res, transported};
    }
    res.face = min_edge->twin()->fi;
    // Optimize this...
    res.local_coords = mesh->face(res.face).to_local_coords(
        f.to_global_coords(res.local_coords + min_t * dir));
    dir = (min_edge->frame_rot * dir).normalized();
    transported = (min_edge->frame_rot * transported).normalized();
    len -= min_t;
  }
  return {res, transported};
}

} // namespace utils