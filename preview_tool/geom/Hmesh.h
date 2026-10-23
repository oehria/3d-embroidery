#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

namespace geom {

class Hmesh {
public:
  // Mesh elements.
  struct Edge;
  struct Face;
  struct Vertex;

  Face &face(int i) { return faces[i]; }
  const Face &face(int i) const { return faces[i]; }
  Vertex &vertex(int i) { return verts[i]; }
  const Vertex &vertex(int i) const { return verts[i]; }
  Edge &edge(int i) { return edges[i]; }
  const Edge &edge(int i) const { return edges[i]; }

  struct Point {
    int face;
    Eigen::Vector2d local_coords;
    Point operator+(const Eigen::Vector2d &vec) const;
    // Returns the point and true if we didn't escape the surface.
    std::pair<Point, bool> add(const Eigen::Vector2d &vec) const;

    // Transport vec along the geodesic at direction dir (with length |dir|).
    std::pair<Point, Eigen::Vector2d>
    transport(const Eigen::Vector2d &dir, const Eigen::Vector2d &vec) const;
    Eigen::Vector3d global_coords() const {
      return mesh->face(face).to_global_coords(local_coords);
    }
    Hmesh *mesh;
  };

  struct Face {
    int index;   // index of the face.
    int ei;      // index of one of the edges.
    Hmesh *mesh; // pointer to the mesh.
    int color = -1;

    Edge *edge() const;

    double area() const;
    Eigen::VectorXd center() const;
    Eigen::Vector3d normal() const;
    bool is_boundary();

    Eigen::Matrix<double, 3, 2> frame;
    Eigen::Vector2d bary_to_local_coords(const Eigen::Vector3d &bary) const;
    Eigen::Vector2d to_local_coords(const Eigen::Vector3d &vec) const;
    Eigen::Vector3d to_bary_coords(const Eigen::Vector3d &vec) const;
    Point project_point(const Eigen::Vector3d &vec) const {
      return Point{
          .face = index, .local_coords = to_local_coords(vec), .mesh = mesh};
    }
    Eigen::Vector3d to_global_coords(const Eigen::Vector2d &vec) const;

    // Following the notation from "Discrete differential operators on polygonal
    // meshes".
    int nf() const { return mesh->F[index].size(); }
  };
  struct Vertex {
    int index;   // index of the vertex.
    int ei;      // index of one of the edges.
    Hmesh *mesh; // pointer to the mesh.

    Eigen::VectorXd coords() const;
    Eigen::VectorXd normal() const;
    Edge *edge() const;
  };

  struct Edge {
    int index;
    int vi;  // index of the origin vertex.
    int fi;  // index of the face.
    int fvi; // index of the face vertex (vi = F[fi][fvi]).
    int ti;  // index of the twin edge.
    int ni;  // index of the next edge.
    int pi;  // index of the previous edge.

    int shift_x = 0;
    int shift_y = 0;
    inline bool is_periodic() const { return shift_x || shift_y; }
    Hmesh *mesh;

    // To move from face() frame to twin()->face() frame multiply by frame_rot.
    Eigen::Matrix2d frame_rot;

    Vertex *origin() const;
    Vertex *target() const;
    Face *face() const;
    Edge *prev() const;
    Edge *next() const;
    Edge *twin() const;
    Eigen::Vector3d normal() const;

    double cos_theta();
    double sin_theta();
    double cot_theta();
    double theta();
    Eigen::VectorXd vec() const;
    Eigen::VectorXd mid() const {
      return (origin()->coords() + target()->coords()) / 2.0;
    }
    Eigen::Vector2d local_vec() const;
    double length();
    inline bool is_boundary() const { return !twin(); }
    std::shared_ptr<void> data;
    template <typename T> T &get_data() {
      return *static_cast<T *>(data.get());
    }
  };

  struct VertexStarIterator {
    VertexStarIterator(Edge *e);
    VertexStarIterator(Edge *e, bool is_begin);
    const VertexStarIterator &operator++();
    bool operator==(const VertexStarIterator &other) const;
    bool operator!=(const VertexStarIterator &other) const;
    Edge *operator*() const { return e; }
    Edge *e;
    bool is_begin;
  };

  struct NodeEdges {
    NodeEdges(const Vertex &v) : e(v.edge()) {}
    VertexStarIterator begin() { return VertexStarIterator(e, true); }
    VertexStarIterator end() { return VertexStarIterator(e, false); }

    Edge *e;
  };

public:
  Eigen::MatrixXd V;
  std::vector<std::vector<int>> F;


  std::vector<Vertex> verts;
  std::vector<Face> faces;
  std::vector<Edge> edges;
  int total_edges;

  std::vector<std::unordered_map<int, int>> verts_edges;
  Hmesh() = default;
  Hmesh(const Hmesh &other);
  Hmesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
  Hmesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
        const Eigen::VectorXi &D);
  Hmesh(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &F);
  Hmesh &operator=(const Hmesh &other);

  std::vector<std::set<int>> vertex_neighbors();


  inline int nv() const { return verts.size(); }
  inline int nf() const { return faces.size(); }
};

} // namespace utils