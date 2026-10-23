#include "imgui.h"
#include "polyscope/types.h"
#include "utils/conversions.h"
#include <fstream>
#include <geom/Hmesh.h>
#include <geom/arap.h>
#include <igl/readOBJ.h>
#include <iostream>
#include <memory>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <sstream>
#include <thread>

std::set<int> constrained_verts;

Eigen::MatrixXd orig_V;
geom::Hmesh flat_mesh;
Eigen::MatrixXd edge_V;
Eigen::MatrixXi edge_E;
std::vector<std::pair<int, int>> seam_verts;

double shrink_factor = 0.8;

Eigen::MatrixXd X;
float soft_constraint_weight = 1e2, edges_weight = 0;
bool arap_updated = false;
bool arap_running = false;

std::string seam_verts_file;
std::unique_ptr<geom::ARAP> arap;

Eigen::MatrixXi filter_bad_edges() {
  std::vector<Eigen::Vector2i> good_edges;
  for (int i = 0; i < edge_E.rows(); i++) {
    double orig_edge_length =
        (flat_mesh.V.row(edge_E(i, 1)) - flat_mesh.V.row(edge_E(i, 0))).norm();
    if (orig_edge_length < 1e-6) {
      // Bad edge, skip it.
      continue;
    }
    constrained_verts.insert(edge_E(i, 0));
    constrained_verts.insert(edge_E(i, 1));
    good_edges.push_back(edge_E.row(i));
  }
  return convert::to_eig_mat(good_edges);
}

std::vector<std::pair<int, int>>
read_seam_verts_file(const std::string &filename) {
  std::vector<std::pair<int, int>> res;
  std::ifstream infile(filename);
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    int a, b;
    iss >> a >> b;
    res.push_back({a, b});
  }
  return res;
}

void init_arap() {
  arap = std::make_unique<geom::ARAP>(flat_mesh, edge_E, shrink_factor);
  std::set<int> unique_indices;
  for (int i = 0; i < edge_E.rows(); i++) {
    unique_indices.insert(edge_E(i, 0));
    unique_indices.insert(edge_E(i, 1));
  }
  std::vector<int> indices(constrained_verts.begin(), constrained_verts.end());
  Eigen::MatrixXd constraints = edge_V(indices, Eigen::placeholders::all);
  arap->set_objective();
  arap->add_soft_constraints(soft_constraint_weight, indices, constraints);

  if (!seam_verts.empty()) {

    arap->objective->add_hard_constraints(
        seam_verts.size(), [&](int i, auto &x, auto &add_eq) {
          int v0 = seam_verts[i].first, v1 = seam_verts[i].second;
          add_eq(x(v0) - x(v1));
        });
  }
  X = flat_mesh.V;
  arap_updated = true;
}

Eigen::VectorXd calculate_shrinking_rate() {
  Eigen::VectorXd shrink_rate(edge_E.rows());
  for (int i = 0; i < edge_E.rows(); i++) {
    double orig_edge_length =
        (flat_mesh.V.row(edge_E(i, 1)) - flat_mesh.V.row(edge_E(i, 0))).norm();
    if (orig_edge_length < 1e-6) {
      shrink_rate(i) = shrink_factor;
      continue;
    }
    double cur_edge_length = (X.row(edge_E(i, 1)) - X.row(edge_E(i, 0))).norm();
    // cur_edge_length should be 0.8 * orig_edge_length, so the error is the
    // difference.

    shrink_rate(i) = cur_edge_length / orig_edge_length;
  }
  return shrink_rate;
}

void report_shrink_error() {
  auto shrink_rate = calculate_shrinking_rate();
  std::cout << "Average shrink error: " << shrink_rate.mean() << std::endl;
  std::cout << "Max shrink error: " << shrink_rate.maxCoeff() << std::endl;
  std::cout << "Min shrink error: " << shrink_rate.minCoeff() << std::endl;
}

void init_polyscope() {
  // Initialize Polyscope
  polyscope::init();
  polyscope::options::ssaaFactor = 4;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
  // Set up a callback to handle keypress events
  polyscope::state::userCallback = []() {
    if (ImGui::CollapsingHeader("Interactive ARAP Controls")) {
      if (ImGui::Button("Init arap")) {
        init_arap();
      }
      if (ImGui::Button("Reset arap")) {
        if (!arap) {
          init_arap();
        }
        X = flat_mesh.V;
        arap_updated = true;
      }
      if (ImGui::Button("Run one arap iteration")) {
        if (!arap) {
          init_arap();
        }
        X = arap->solve(X);
        arap_updated = true;
      }
      if (!arap_running) {
        if (ImGui::Button("Run arap")) {
          if (!arap) {
            init_arap();
          }
          arap_running = true;
          std::thread([&]() {
            while (arap_running) {
              X = arap->solve(X);
              arap_updated = true;
            }
          }).detach();
        }
      } else {
        if (ImGui::Button("Stop arap")) {
          arap_running = false;
        }
      }

      if (ImGui::SliderFloat("Soft constraints weight", &soft_constraint_weight,
                             0, 1e2)) {
        if (!arap) {
          init_arap();
        }
        arap->set_soft_constraints_weight(soft_constraint_weight);
      }
      if (ImGui::SliderFloat("Edges weight", &edges_weight, 0, 1e2)) {
        if (!arap) {
          init_arap();
        }
        arap->set_edges_weight(edges_weight);
      }
    }

    ImGui::Separator();
    static bool one_click_preview_running = false;
    if (!one_click_preview_running) {
      if (ImGui::Button("One click preview")) {
        if (!arap) {
          init_arap();
        }
        one_click_preview_running = true;
        soft_constraint_weight = 100;
        edges_weight = 0;
        arap->set_soft_constraints_weight(soft_constraint_weight);
        arap->set_edges_weight(edges_weight);
        std::thread([&]() {
          polyscope::info(
              "Running 20 iterations of ARAP with soft constraints weight = " +
              std::to_string(soft_constraint_weight) +
              " and edges weight = " + std::to_string(edges_weight));
          for (int i = 0; i < 20 && one_click_preview_running; i++) {
            X = arap->solve(X);
            arap_updated = true;
          }
          soft_constraint_weight = 1e-3;
          edges_weight = 300;
          arap->set_soft_constraints_weight(soft_constraint_weight);
          arap->set_edges_weight(edges_weight);
          polyscope::info(
              "Running 20 iterations of ARAP with soft constraints weight = " +
              std::to_string(soft_constraint_weight) +
              " and edges weight = " + std::to_string(edges_weight));
          for (int i = 0; i < 20 && one_click_preview_running; i++) {
            X = arap->solve(X);
            arap_updated = true;
          }
          polyscope::info("One click preview finished.");
          one_click_preview_running = false;
        }).detach();
      }
    } else {
      if (ImGui::Button("Stop one click preview")) {
        one_click_preview_running = false;
      }
    }
    ImGui::Separator();

    if (ImGui::Button("Report shrink error")) {
      report_shrink_error();
    }
    if (ImGui::Button("Export obj")) {
      std::ofstream outfile("output.obj");
      for (int i = 0; i < X.rows(); i++) {
        outfile << "v " << X(i, 0) << " " << X(i, 1) << " " << X(i, 2)
                << std::endl;
      }
      // Flat mesh V(:, 0:2) are the original UV coordinates, which we can use
      // as texture coordinates.
      for (int i = 0; i < flat_mesh.V.rows(); i++) {
        outfile << "vt " << orig_V(i, 0) << " " << orig_V(i, 1) << std::endl;
      }

      for (int i = 0; i < flat_mesh.F.size(); i++) {
        // OBJ format is 1-indexed.
        outfile << "f " << flat_mesh.F[i][0] + 1 << "/" << flat_mesh.F[i][0] + 1
                << " " << flat_mesh.F[i][1] + 1 << "/" << flat_mesh.F[i][1] + 1
                << " " << flat_mesh.F[i][2] + 1 << "/" << flat_mesh.F[i][2] + 1
                << std::endl;
      }
      outfile.close();
    }

    if (arap_updated) {
      arap_updated = false;
      polyscope::getSurfaceMesh("mesh")->updateVertexPositions(X);
    }
  };
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
read_edge_mesh(const std::string &filename) {
  std::vector<Eigen::Vector3d> verts;
  std::vector<Eigen::Vector2i> edges;
  std::ifstream infile(filename);

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") {
      double x, y, z;
      iss >> x >> y >> z;
      verts.push_back(Eigen::Vector3d(x, y, z));
    } else if (type == "l") {
      int a, b;
      iss >> a >> b;
      edges.push_back(Eigen::Vector2i(a - 1, b - 1));
    }
  }

  return {convert::to_eig_mat(verts), convert::to_eig_mat(edges)};
}

int main(int argc, char *argv[]) {
  init_polyscope();
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readOBJ(argv[1], V, F);
  orig_V = V;
  flat_mesh = geom::Hmesh(V, F);
  flat_mesh.V = (flat_mesh.V *
                 Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).matrix())
                    .eval();
  flat_mesh.V = (flat_mesh.V *
                 Eigen::AngleAxisd(M_PI / 3, Eigen::Vector3d::UnitY()).matrix())
                    .eval();
  polyscope::registerSurfaceMesh("mesh", flat_mesh.V, F)
      ->setShadeStyle(polyscope::MeshShadeStyle::Smooth);
  std::cout << "Reading edge mesh from " << argv[2] << std::endl;
  std::tie(edge_V, edge_E) = read_edge_mesh(argv[2]);
  std::cout << "Number of edges: " << edge_E.rows() << std::endl;
  edge_E = filter_bad_edges();

  polyscope::show();
  return 0;
}