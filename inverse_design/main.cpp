#include <iostream>

#include <Eigen/Core>
#include <directional/readOBJ.h>
#include <directional/polyvector_to_raw.h>
#include <directional/polyvector_iteration_functions.h>
#include <directional/PolyVectorData.h>
#include <directional/polyvector_field.h>
#include <directional/write_raw_field.h>
#include <directional/directional_viewer.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/CartesianField.h>
#include <directional/setup_integration.h>
#include <directional/integrate.h>
#include <directional/writeOBJ.h>
#include <igl/local_basis.h>
#include <igl/cat.h>
#include <igl/grad.h>
#include <igl/arap.h>
#include <igl/lscm.h>
#include <directional/combing.h>
#include "polyscope/curve_network.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <filesystem>
#include <igl/barycenter.h>

using namespace directional;
using namespace Eigen;
using namespace std;

directional::TriMesh mesh;
directional::PCFaceTangentBundle ftb;
directional::CartesianField field, alignment_field, smoothed_field, curl_field, pv_smooth, P, S, S_new, finaloutput, init_jacs;
int N = 4;

directional::DirectionalViewer viewer;
const string res_dir = "../res/"; // output folder, relative to the build folder the program is run from
bool show_smooth = true;
bool show_curl = true;
bool show_main = true;
bool show_cons = true;
bool show_intjac = true;
float c_smooth = 1;

double stitch_dist=0.2;
double c_cons=0.3;
double jump_thresh=0.5;
MatrixXd UV;

double curr_energy, prev_energy;

std::vector<std::vector<Eigen::Vector3d>> streamlines;
std::vector<std::vector<int>> streamline_faces;

double shrink_factor = 0.8;
double s_ratio = 0.0;
double jac_smoothness=10;

double newest_smooth_jac=0;

bool cutinit = false;

directional::TriMesh meshCut;

int max_iters=200;

vector<tuple<int, double>> cat_energy; // holds collection of energy terms (category denoting if it's smoothness, jacobian etc etc) to check convergence

std::vector<double> energy_diff;

std::vector<std::vector<Eigen::Vector3d>> uv_streamlines;
std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> uv_boundary;

std::vector<std::vector<bool>> final_mask;

void save_field_rendering(directional::CartesianField &field, string file_loc){
    directional::CartesianField combedfield;
    directional::principal_matching(field);
    directional::combing(field, combedfield);
    directional::write_raw_field(file_loc, combedfield);
}

// utils
static void computeSurfaceGradientMatrix(const MatrixXd &V, const MatrixXi &F, SparseMatrix<double> &D1, SparseMatrix<double> &D2, vector<MatrixXd> &local_basis)
{
    MatrixXd F1, F2, F3;
    SparseMatrix<double> DD, Dx, Dy, Dz;

    igl::local_basis(V, F, F1, F2, F3);
    local_basis.clear();
    local_basis.push_back(F1);
    local_basis.push_back(F2);
    local_basis.push_back(F3);
    igl::grad(V, F, DD);

    Dx = DD.topLeftCorner(F.rows(), V.rows());
    Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
    Dz = DD.bottomRightCorner(F.rows(), V.rows());

    D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
    D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;
}

static inline void SSVD2x2(const Eigen::Matrix2d &J, Eigen::Matrix2d &U, Eigen::Matrix2d &S, Eigen::Matrix2d &V)
{
    double e = (J(0) + J(3)) * 0.5;
    double f = (J(0) - J(3)) * 0.5;
    double g = (J(1) + J(2)) * 0.5;
    double h = (J(1) - J(2)) * 0.5;
    double q = sqrt((e * e) + (h * h));
    double r = sqrt((f * f) + (g * g));
    double a1 = atan2(g, f);
    double a2 = atan2(h, e);
    double rho = (a2 - a1) * 0.5;
    double phi = (a2 + a1) * 0.5;

    S(0) = q + r;
    S(1) = 0;
    S(2) = 0;
    S(3) = q - r;

    double c = cos(phi);
    double s = sin(phi);
    U(0) = c;
    U(1) = s;
    U(2) = -s;
    U(3) = c;

    c = cos(rho);
    s = sin(rho);
    V(0) = c;
    V(1) = -s;
    V(2) = s;
    V(3) = c;
}

// optimization energy terms
directional::CartesianField smooth_orientation(const directional::CartesianField &field, const directional::PolyVectorData &pvData)
{
    directional::CartesianField smoothie_field;
    smoothie_field = field;
    directional::principal_matching(smoothie_field);
    int m = mesh.F.rows();
    double smoothWeight = c_smooth; // we wanna smooth
    double rosyWeight = 0;         // we don't have a rosy field (orthogonality of stitches not required)

    Eigen::VectorXi constFaces(N * m); // just adding all faces as const, we can still set the alignment weight to 0, so ultimately this means nothing
    for (int i = 0; i < m; i++)
    {
        constFaces(N * i) = i;
        constFaces(N * i + 1) = i;
        constFaces(N * i + 2) = i;
        constFaces(N * i + 3) = i;
    }

    Eigen::MatrixXd constVectors(N * m, 3);
    MatrixXd ext_align(m, N * 3);
    Eigen::VectorXd alignWeights(N * m);
    VectorXd is_anisotropic(m);
    is_anisotropic.setZero();
    for (int i = 0; i < m; i++)
    {
        RowVector3d vec1, vec2;
        vec1 << field.extField(i, 0), field.extField(i, 1), field.extField(i, 2);
        vec2 << field.extField(i, 3), field.extField(i, 4), field.extField(i, 5);
        // don't care about magnitude in this step
        constVectors.row(N * i) = vec1.normalized();
        constVectors.row(N * i + 1) = vec2.normalized();
        constVectors.row(N * i + 2) = -vec1.normalized(); // can do this due to sign symmetry being used
        constVectors.row(N * i + 3) = -vec2.normalized();

        //correct alignment weights according to stiffness matrix
        alignWeights(N * i) = 0.5*(((vec1.squaredNorm()-vec2.squaredNorm())*(vec1.squaredNorm()-vec2.squaredNorm()))/(vec1.squaredNorm()+vec2.squaredNorm()));//at this stage sv of j and s are same thus we can directly do this
        alignWeights(N * i+1) = 0.5*(((vec1.squaredNorm()-vec2.squaredNorm())*(vec1.squaredNorm()-vec2.squaredNorm()))/(vec1.squaredNorm()+vec2.squaredNorm()));
        alignWeights(N * i+2) = 0.5*(((vec1.squaredNorm()-vec2.squaredNorm())*(vec1.squaredNorm()-vec2.squaredNorm()))/(vec1.squaredNorm()+vec2.squaredNorm()));
        alignWeights(N * i+3) = 0.5*(((vec1.squaredNorm()-vec2.squaredNorm())*(vec1.squaredNorm()-vec2.squaredNorm()))/(vec1.squaredNorm()+vec2.squaredNorm()));
        if (abs(vec1.norm() - vec2.norm()) > 0.5 * (1 / shrink_factor - 1))
        {
            is_anisotropic(i) = 1;
        }
        ext_align.row(i) << constVectors.row(N * i), constVectors.row(N * i + 1), constVectors.row(N * i + 2), constVectors.row(N * i + 3);
    }

    alignment_field.init(ftb, directional::fieldTypeEnum::POLYVECTOR_FIELD, N);
    alignment_field.set_extrinsic_field(ext_align);
    directional::CartesianField rawField;
    directional::PolyVectorData currentdata;
    currentdata.N = N;
    currentdata.tb = &ftb;
    currentdata.verbose = false;
    currentdata.constSpaces = constFaces;
    currentdata.constVectors = constVectors;
    currentdata.signSymmetry = true;

    currentdata.wSmooth = smoothWeight;
    currentdata.wRoSy = rosyWeight;
    currentdata.wAlignment = alignWeights;

    smoothie_field.init(ftb, directional::fieldTypeEnum::POLYVECTOR_FIELD, N);

    directional::polyvector_field(currentdata, smoothie_field);

    VectorXcd fullDofs = currentdata.reducMat * currentdata.reducedDofs + currentdata.reducRhs;
    double smoothness_energy = std::abs((fullDofs.adjoint() * currentdata.smoothMat.adjoint() * currentdata.WSmooth * currentdata.smoothMat * fullDofs).coeff(0, 0) * (currentdata.wSmooth / currentdata.totalSmoothWeight));

    cat_energy.push_back(make_tuple(1, smoothness_energy));

    pv_smooth = smoothie_field;

    directional::polyvector_to_raw(smoothie_field, rawField, N % 2 == 0);
    smoothed_field = rawField;

    alignWeights.setOnes();
    curl_field = directional::curl_projection(smoothie_field, currentdata);
    directional::polyvector_to_raw(curl_field, rawField, N % 2 == 0);
    curl_field = rawField;

    return smoothed_field;
}

void recompute_magnitude(const directional::CartesianField &orientation, const directional::CartesianField &original, directional::CartesianField &magnitude)
{
    magnitude.init(ftb, directional::fieldTypeEnum::RAW_FIELD, 4);
    int m = orientation.extField.rows();
    MatrixXd face_norm;
    igl::per_face_normals(mesh.V, mesh.F, face_norm);
    MatrixXd ext_field(m, 4 * 3);
    RowVector3d vec1, vec2;

    for (int f = 0; f < m; f++){
        Vector3d dir1 = orientation.extField.block(f, 0, 1, 3).normalized(); // find best match for 1st, everything else same constant offset as we have ccw field

        double score = -1000;
        int offset = 0;
        for (int v = 0; v < 4; v++){
            RowVector3d candidate_dir = original.extField.block(f, 3 * v, 1, 3).normalized();
            double curr_score = dir1.dot(candidate_dir);
            if (curr_score > score){
                offset = v;
                score = curr_score;
            }
        }
        for (int v = 0; v < 4; v++)
        {
            int match_idx = (v + offset) % 4;
            // take closest match length
            ext_field.block(f, 3 * v, 1, 3) = original.extField.block(f, 3 * match_idx, 1, 3).norm() * orientation.extField.block(f, 3 * v, 1, 3).normalized();

        }
        // check ccw
        vec1 << ext_field(f, 0), ext_field(f, 1), ext_field(f, 2);
        vec2 << ext_field(f, 3), ext_field(f, 4), ext_field(f, 5);
        RowVector3d cross = vec1.cross(vec2);
        RowVector3d norm = face_norm.row(f);
        double swap = cross.dot(norm);
        if (swap < 0){
            cerr << "recompute_magnitude: field is not CCW at face " << f << endl;
            RowVector3d temp = vec1;
            vec1 = vec2;
            vec2 = temp;
            ext_field.row(f) << vec1, vec2, -vec1, -vec2;
            return;

        }
    }
    magnitude.set_extrinsic_field(ext_field);
}

double compute_stitching_weight(
    int i, int j, 
    const Eigen::Vector3d& v_i, 
    const Eigen::MatrixXd& centers, 
    const Eigen::MatrixXd& normals, 
    double r, 
    double k = 4.0) 
{
    using namespace Eigen;

    Vector3d pos_i = centers.row(i);
    Vector3d pos_j = centers.row(j);
    Vector3d n_i = normals.row(i);
    Vector3d n_j = normals.row(j);

    Vector3d rel_pos = pos_j - pos_i;
    double dist = rel_pos.norm();
    
    if (dist > r || dist < 1e-10) return 0.0;

    // 1. Distance Falloff (Linear)
    double w_dist = 1.0 - (dist / r);

    // 2. Curvature Penalty
    // If faces are pointing in opposite directions (sharp fold), reduce weight
    double w_curv = std::max(0.0, n_i.dot(n_j));

    // 3. Directional Alignment (The Anisotropic "Beam")
    // Project the relative position vector onto the tangent plane of face i
    Vector3d rel_dir = rel_pos / dist;
    Vector3d rel_tangent = (rel_dir - (rel_dir.dot(n_i)) * n_i).normalized();
    
    // How much does the position of face J align with the direction of vector V?
    double alignment = std::abs(rel_tangent.dot(v_i));
    double w_dir = std::pow(alignment, k);

    return w_dist * w_dir;
}

void new_stitching_line_consistency(directional::CartesianField &S, directional::CartesianField &S_cons,double r=jump_thresh/2,double lambda_fab=1,double lambda_c=c_cons){
    // 1. Initialize output and compute face centroids
   S_cons = S;
    MatrixXd centers, normals;
    igl::barycenter(mesh.V, mesh.F, centers);
    igl::per_face_normals(mesh.V, mesh.F, normals); // Required for tangent projection
    
    int num_faces = mesh.F.rows();
    int N = S.N;

    // Pre-calculate neighbor indices for each face within radius r (Set M)
    // For performance on large meshes, consider using igl::octree or nanoflann here.
    std::vector<std::vector<int>> M(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        for (int j = 0; j < num_faces; ++j) {
            if (i == j) continue;
            if ((centers.row(i) - centers.row(j)).norm() <= r) {
                M[i].push_back(j);
            }
        }
    }

    //visualize radius for debugging purposes
    VectorXd facedata(mesh.F.rows());
    facedata.setZero();
    double cons_energy=0;
    double cons_energy_after=0;

    // 2. Iterate over each face
   for (int i = 0; i < num_faces; ++i) {
        for (int d = 0; d < 2; ++d) {
            Vector3d v = S.extField.block(i, d * 3, 1, 3).transpose();
            double r_orig = v.norm();
            if (r_orig < 1e-12) continue;
            Vector3d v_dir = v / r_orig;
            
            double neighbor_mag_sum = 0;
            double total_w_vw = 0;
            std::vector<std::pair<double, double>> neighbor_weight_mag; // (weight, best_w_mag)

            for (int neighbor_idx : M[i]) {
                // CALCULATE THE NEW WEIGHT HERE
                // k=4 is a good starting point for a narrow "line"
                double weight = compute_stitching_weight(i, neighbor_idx, v_dir, centers, normals, r, 4.0);
                if(i==120 && d==0){
                    facedata(neighbor_idx)=weight;
                }
                
                if (weight < 1e-6) continue;

                double max_dot = -std::numeric_limits<double>::infinity();
                double best_w_mag = 0;
                
                for (int k_vec = 0; k_vec < N; ++k_vec) {
                    Vector3d w = S.extField.block(neighbor_idx, k_vec * 3, 1, 3).transpose();
                    double w_mag = w.norm();
                    if (w_mag < 1e-12) continue;
                    
                    // Simple 3D dot is "rough" parallel transport
                    double dot = v_dir.dot(w / w_mag);
                    if (dot > max_dot) {
                        max_dot = dot;
                        best_w_mag = w_mag;
                    }
                }
                
                // Use the anisotropic weight
                neighbor_mag_sum += weight * best_w_mag;
                cons_energy+=weight*(best_w_mag-r_orig)*(best_w_mag-r_orig);
                total_w_vw += weight;
                neighbor_weight_mag.emplace_back(weight, best_w_mag);
            }

            // 5. Optimization: Calculate weighted average magnitude
            double r_new = r_orig;
            if (total_w_vw > 0) {
                r_new = (lambda_fab * r_orig + lambda_c * neighbor_mag_sum) /
                        (lambda_fab + lambda_c * total_w_vw);
            }

            for (const auto &wm : neighbor_weight_mag) {
                cons_energy_after += wm.first * (wm.second - r_new) * (wm.second - r_new);
            }

            // Apply the new magnitude to S_cons (maintaining sign symmetry)
            S_cons.extField.block(i, d * 3, 1, 3) = v_dir.transpose() * r_new;
            S_cons.extField.block(i, (d + 2) * 3, 1, 3) = -v_dir.transpose() * r_new;
        }
    }
    cat_energy.push_back(make_tuple(7,cons_energy));
    cat_energy.push_back(make_tuple(8,cons_energy_after));

    viewer.set_surface_face_data(facedata, "linefaces");
}

void parameterize(directional::CartesianField &in, directional::CartesianField &out)
{
    directional::IntegrationData intData(N);

    directional::principal_matching(in);

    directional::CartesianField combedField;

    intData.verbose = false;
    intData.integralSeamless = false;
    intData.roundSeams = false;
    intData.lengthRatio = 1 / (mesh.V.colwise().maxCoeff() - mesh.V.colwise().minCoeff()).norm();

    directional::TriMesh newmeshcut;
    directional::setup_integration(in, intData, newmeshcut, combedField); // comb
    out = combedField;
    directional::write_raw_field(res_dir + "crossfield.raw", in);

    std::vector<int> seamsList;

    for (int i = 0; i < mesh.F.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (intData.face2cut(i, j))
            {
                seamsList.push_back(mesh.FE(i, j));
            }
        }
    }

    Eigen::VectorXi seams = Eigen::VectorXi::Map(seamsList.data(), seamsList.size());

    Eigen::MatrixXd NFunction, NCornerFunction;
    directional::integrate(combedField, intData, newmeshcut, NFunction, NCornerFunction);

    meshCut = newmeshcut;

    UV = NFunction.block(0, 0, NFunction.rows(), 2);

    viewer.highlight_edges(seams, "Seams");
}

void parameterization_field(const directional::CartesianField &S, const MatrixXd &V, const MatrixXi &F, const MatrixXd &UV, directional::CartesianField &P)
{ // given stitching direction and current uv locally project to best fitting jacobian and then encode these as polyvectors to be integrated
    double energy = 0;
    directional::TriMesh mesh_flat;
    mesh_flat = meshCut;
    mesh_flat.V.col(0) = UV.col(0);
    mesh_flat.V.col(1) = UV.col(1);
    mesh_flat.V.col(2).setZero();
    viewer.set_surface_mesh(mesh_flat, "pattern", 1);

    // set up jacobians
    SparseMatrix<double> Dx, Dy;
    vector<MatrixXd> local_basis;
    computeSurfaceGradientMatrix(V, F, Dx, Dy, local_basis);
    SparseMatrix<double> J;
    vector<vector<SparseMatrix<double>>> mat;
    VectorXd Ju, Jv;
    mat.resize(2);
    mat[0].push_back(Dx);
    mat[1].push_back(Dy);
    igl::cat(mat, J);

    int m = F.rows();

    // compute Jacobian
    Ju = J * UV.col(0);
    Jv = J * UV.col(1);

    double stretch_energy=0;
    double abs_stretch_energy=0;

    double energy_fab=0;

    CartesianField rawS = S;

    std::vector<Matrix2d> local_proj;
    std::vector<double> error_viz;
    // Compute local projected jacobians
    for (int f = 0; f < m; f++)
    {
        Eigen::Matrix2d U, S, Vf, Juv;
        Juv << Ju(f), Ju(f + m), Jv(f), Jv(f + m);
        SSVD2x2(Juv, U, S, Vf);
        // compute V based on direction of stitches
        // extract direction of stitches (if existing)
        Vector3d global_stitching_dir = rawS.extField.block(f, 0, 1, 3).transpose();
        Vector3d orth_vec = rawS.extField.block(f, 3, 1, 3).transpose();
        // convert to local coordinate frame of 3d triangle
        Vector2d local_stitching_dir;
        local_stitching_dir << local_basis[0].row(f) * global_stitching_dir, local_basis[1].row(f) * global_stitching_dir;
        // convert to V to axis align
        local_stitching_dir = local_stitching_dir.normalized().eval();
        Eigen::Matrix2d Vt;
        Vt << local_stitching_dir.x(), local_stitching_dir.y(), -local_stitching_dir.y(), local_stitching_dir.x();
        // compute S based on stitching magnitudes --> project to closest
        if (global_stitching_dir.norm() > 1 + 0.5 * (1 / shrink_factor - 1))
        {
            S(0, 0) = 1 / shrink_factor;
        }
        else
        {
            S(0, 0) = 1;
        }
        stretch_energy+=m*mesh.faceAreas(f)*(global_stitching_dir.norm()-S(0,0))*(global_stitching_dir.norm()-S(0,0));
        abs_stretch_energy+=std::abs(global_stitching_dir.norm()-S(0,0));

        if (orth_vec.norm() > 1 + 0.5 * (1 / shrink_factor - 1))
        {
            S(1, 1) = 1 / shrink_factor;
        }
        else
        {
            S(1, 1) = 1;
        }
        stretch_energy+=m*mesh.faceAreas(f)*(orth_vec.norm()-S(1,1))*(orth_vec.norm()-S(1,1));
        abs_stretch_energy+=std::abs(orth_vec.norm()-S(1,1));
        // pure fabrication-adherence energy: fit against the unprojected stitch magnitudes
        Eigen::Matrix2d S_actual;
        S_actual << global_stitching_dir.norm(), 0, 0, orth_vec.norm();
        Matrix2d U_fit_actual, S_u_actual, U_u_actual, V_u_actual;
        SSVD2x2((Juv * Vt.transpose() * S_actual), U_u_actual, S_u_actual, V_u_actual);
        U_fit_actual = U_u_actual * V_u_actual.transpose();
        energy_fab += mesh.faceAreas(f) * (Juv - U_fit_actual * S_actual * Vt).squaredNorm();

        // fit U to get best fitting final alignment
        Matrix2d U_fit, S_u, U_u, V_u;

        SSVD2x2((Juv * Vt.transpose() * S), U_u, S_u, V_u);
        U_fit = U_u * V_u.transpose();
        local_proj.push_back(U_fit * S * Vt);
        energy += mesh.faceAreas(f) * (Juv - U_fit * S * Vt).squaredNorm();

        error_viz.push_back((Juv - U_fit * S * Vt).squaredNorm());
    }
    prev_energy=get<1>(cat_energy[cat_energy.size()-1]);
    cat_energy.push_back(make_tuple(5, energy));
    cat_energy.push_back(make_tuple(4, stretch_energy));
    cat_energy.push_back(make_tuple(0, abs_stretch_energy));

    cat_energy.push_back(make_tuple(9, energy_fab));
    curr_energy=energy;
    viewer.psSurfaceMeshList[0]->addFaceScalarQuantity("face_feasibility_error", error_viz);

    // convert to parameterization polyvectorfield out
    P.init(ftb, directional::fieldTypeEnum::RAW_FIELD, 4); // p istelf raw
    // build extrinsic field
    MatrixXd ext_field;
    ext_field.resize(m, 4 * 3);

    MatrixXd face_norm;
    igl::per_face_normals(mesh.V, mesh.F, face_norm); 
    Vector3d vec1, vec2;
    for (int f = 0; f < m; f++)
    {
        MatrixXd Jac = MatrixXd(2, 3);
        Jac.row(0) = local_proj[f](0, 0) * local_basis[0].row(f) + local_proj[f](0, 1) * local_basis[1].row(f);
        Jac.row(1) = local_proj[f](1, 0) * local_basis[0].row(f) + local_proj[f](1, 1) * local_basis[1].row(f);
        ext_field.row(f) << Jac.row(0), Jac.row(1), -Jac.row(0), -Jac.row(1);
        // check ccw
        vec1 << ext_field(f, 0), ext_field(f, 1), ext_field(f, 2);
        vec2 << ext_field(f, 3), ext_field(f, 4), ext_field(f, 5);
        RowVector3d cross = vec1.cross(vec2);
        RowVector3d norm = face_norm.row(f);
        double swap = cross.dot(norm);
        if (swap < 0)
        {
            RowVector3d temp = vec1;
            vec1 = vec2;
            vec2 = temp;
            ext_field.row(f) << vec1, vec2, -vec1, -vec2;
        }
    }
    P.set_extrinsic_field(ext_field);
}

void u_fitting(
    const directional::CartesianField &S, 
    const Eigen::MatrixXd &V, 
    const Eigen::MatrixXi &F, 
    const directional::CartesianField &J_field, 
    directional::CartesianField &P)
{
    int m = F.rows();
    
    Eigen::SparseMatrix<double> Dx, Dy;
    std::vector<Eigen::MatrixXd> local_basis;
    computeSurfaceGradientMatrix(V, F, Dx, Dy, local_basis);

    Eigen::MatrixXd face_normals;
    igl::per_face_normals(V, F, face_normals);

    std::vector<Eigen::Matrix2d> local_proj;
    local_proj.reserve(m);

    for (int f = 0; f < m; f++)
    {
        // 1. Build Juv directly from J_field's internal 2D representation
        // intField row layout: [u.x, u.y, v.x, v.y]
        Eigen::Matrix2d Juv;
        Juv << J_field.intField(f, 0), J_field.intField(f, 1),
               J_field.intField(f, 2), J_field.intField(f, 3);

        // 2. Extract local stitching direction from S.intField
        Eigen::Vector2d local_stitching_dir(S.intField(f, 0), S.intField(f, 1));
        local_stitching_dir.normalize();

        // 3. Build Vt (Rotation to align with stitching direction)
        Eigen::Matrix2d Vt;
        Vt << local_stitching_dir.x(), local_stitching_dir.y(), 
             -local_stitching_dir.y(), local_stitching_dir.x();

        // 4. Determine target scaling S_target
        Eigen::Matrix2d S_target = Eigen::Matrix2d::Identity();
        double threshold = 1.0 + 0.3 * (1.0 / shrink_factor - 1.0);

        if (S.extField.block(f, 0, 1, 3).norm() > threshold)
            S_target(0, 0) = 1.0 / shrink_factor;
        
        if (S.extField.block(f, 3, 1, 3).norm() > threshold)
            S_target(1, 1) = 1.0 / shrink_factor;

        // 5. Fit Rotation U via Procrustes: min || Juv - U * S_target * Vt ||
        Eigen::Matrix2d U_svd, S_svd, V_svd;
        SSVD2x2((Juv * Vt.transpose() * S_target), U_svd, S_svd, V_svd);
        Eigen::Matrix2d U_fit = U_svd * V_svd.transpose();

        local_proj.push_back(U_fit * S_target * Vt);
    }

    // 6. Finalize output field P
    P.init(ftb, directional::fieldTypeEnum::RAW_FIELD, 4);
    Eigen::MatrixXd ext_field(m, 12);

    for (int f = 0; f < m; f++)
    {
        // Map 2D fitted Jacobian back to 3D extrinsic vectors
        Eigen::Vector3d u_vec = local_proj[f](0, 0) * local_basis[0].row(f).transpose() + 
                                local_proj[f](0, 1) * local_basis[1].row(f).transpose();
        Eigen::Vector3d v_vec = local_proj[f](1, 0) * local_basis[0].row(f).transpose() + 
                                local_proj[f](1, 1) * local_basis[1].row(f).transpose();

        // CCW orientation check
        if (u_vec.cross(v_vec).dot(face_normals.row(f)) < 0)
        {
            std::swap(u_vec, v_vec);
        }

        ext_field.row(f) << u_vec.transpose(), v_vec.transpose(), -u_vec.transpose(), -v_vec.transpose();
    }

    P.set_extrinsic_field(ext_field);
}

void print_energy()
{
    for(int i=0; i<10; i++){
        cout<<"category "<<i<<endl;
        for (auto t : cat_energy)
        {
            if(get<0>(t)==i){
                cout<<get<1>(t)<<endl;
            }
        }
    }
}

// initialization
void local_global_init(const MatrixXd &V, const MatrixXi &F, MatrixXd &UV, double ratio, int num_iter, directional::CartesianField &out_field)
{
    // set up jacobians
    SparseMatrix<double> Dx, Dy;
    vector<MatrixXd> local_basis;
    computeSurfaceGradientMatrix(V, F, Dx, Dy, local_basis);
    SparseMatrix<double> J;
    vector<vector<SparseMatrix<double>>> mat;
    VectorXd Ju, Jv;
    mat.resize(2);
    mat[0].push_back(Dx);
    mat[1].push_back(Dy);
    igl::cat(mat, J);
    // set up solver matrix
    mat.clear();
    mat.resize(4);
    SparseMatrix<double> dzero;
    dzero.resize(Dx.rows(), Dx.cols());
    dzero.setZero();
    mat[0].push_back(Dx);
    mat[0].push_back(dzero);
    mat[1].push_back(Dy);
    mat[1].push_back(dzero);
    mat[2].push_back(dzero);
    mat[2].push_back(Dx);
    mat[3].push_back(dzero);
    mat[3].push_back(Dy);
    SparseMatrix<double> A;
    igl::cat(mat, A);
    SparseMatrix<double> rhs_mult = A.transpose();
    A = (A.transpose() * A).eval();
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success)
        cerr << "local_global_init: factorization failed: " << solver.lastErrorMessage() << endl;
    // set stretching singular values
    VectorXd rhs_jacs(F.rows() * 4);
    int m = F.rows();
    // start minimization iterations
    MatrixXd ext_field;
    ext_field.resize(F.rows(), 3 * 4);
    ext_field.setZero();
    MatrixXd ext_jac;
    ext_jac.resize(F.rows(), 3 * 4);
    ext_jac.setZero();

    for (int iter = 0; iter < num_iter; iter++)
    {
        // update Jacobian
        Ju = J * UV.col(0);
        Jv = J * UV.col(1);
        // collect current stretching stats
        MatrixXd stretching_factors(m, 2);
        for (int f = 0; f < m; f++)
        {
            Eigen::Matrix2d U, S, Vf, Juv;
            Juv << Ju(f), Ju(f + m), Jv(f), Jv(f + m);
            SSVD2x2(Juv, U, S, Vf);
            stretching_factors.row(f) << S(0, 0), S(1, 1);
        }
        // sort, pair so index is remembered, and remap such that about the upper ratio will be assigned to the higher and the rest to no stitch value
        const int nTotal = 2 * m;
        // Flatten matrix into a vector
        double val_high = 1 / shrink_factor;
        double val_low = 1;
        std::vector<std::pair<double, int>> values;
        values.reserve(nTotal);
        for (int i = 0; i < m; ++i)
        {
            values.emplace_back(stretching_factors(i, 0), 2 * i);
            values.emplace_back(stretching_factors(i, 1), 2 * i + 1);
        }
        // Sort descending by value
        std::sort(values.begin(), values.end(),
                  [](auto &a, auto &b)
                  { return a.first > b.first; });
        // Determine cutoff index
        int nHigh = static_cast<int>(ratio * nTotal);
        // Prepare result vector
        std::vector<double> mapped(nTotal, val_low);
        for (int k = 0; k < nHigh; ++k)
        {
            mapped[values[k].second] = val_high;
        }
        // Reconstruct matrix
        MatrixXd out(m, 2);
        for (int i = 0; i < m; ++i)
        {
            out(i, 0) = mapped[2 * i];
            out(i, 1) = mapped[2 * i + 1];
        }
        // Compute updated local projected jacobians
        for (int f = 0; f < m; f++)
        {
            Eigen::Matrix2d U, S, Vf, Juv, local_proj;
            Juv << Ju(f), Ju(f + m), Jv(f), Jv(f + m);
            SSVD2x2(Juv, U, S, Vf);
            S << out(f, 0), 0, 0, out(f, 1); // project to computed
            local_proj = U * S * Vf.transpose();
            if (local_proj.determinant() < 0)
            {
                local_proj.row(1) *= -1;
            }
            rhs_jacs(f) = local_proj(0, 0);
            rhs_jacs(f + m) = local_proj(0, 1);
            rhs_jacs(f + 2 * m) = local_proj(1, 0);
            rhs_jacs(f + 3 * m) = local_proj(1, 1);
            // update vector field
            Eigen::Vector2d max_stretch_dir_uv = Vf.col(0); // direction of biggest change in parameterization
            Eigen::RowVector3d global_dir = max_stretch_dir_uv(0) * local_basis[0].row(f) +
                                            max_stretch_dir_uv(1) * local_basis[1].row(f);
            ext_field.block(f, 0, 1, 3) = global_dir.normalized() * stretching_factors(f,0);
            ext_field.block(f, 6, 1, 3) = -global_dir.normalized() * stretching_factors(f,0);
            max_stretch_dir_uv = Vf.col(1); // other direction
            global_dir = max_stretch_dir_uv(0) * local_basis[0].row(f) +
                         max_stretch_dir_uv(1) * local_basis[1].row(f);

            ext_field.block(f, 3, 1, 3) = global_dir.normalized() * stretching_factors(f,1);
            ext_field.block(f, 9, 1, 3) = -global_dir.normalized() * stretching_factors(f,1);
            // update jac field
            Eigen::Vector2d jacdir = Juv.row(0).transpose(); // direction of biggest change in parameterization
            global_dir = jacdir(0) * local_basis[0].row(f) +
                            jacdir(1) * local_basis[1].row(f);
            ext_jac.block(f, 0, 1, 3) = global_dir;
            ext_jac.block(f, 6, 1, 3) = -global_dir;
            jacdir = Juv.row(1).transpose(); // other direction
            global_dir = jacdir(0) * local_basis[0].row(f) +
                         jacdir(1) * local_basis[1].row(f);
            ext_jac.block(f, 3, 1, 3) = global_dir;
            ext_jac.block(f, 9, 1, 3) = -global_dir;
        }
        VectorXd b = rhs_mult * rhs_jacs;
        VectorXd sol = solver.solve(b);
        UV.col(0) = sol.block(0, 0, UV.rows(), 1);
        UV.col(1) = sol.block(UV.rows(), 0, UV.rows(), 1);
    }

    // transform into ccw N=4 field so subsequent processing works
    Vector3d vec1, vec2;
    MatrixXd face_norm;
    igl::per_face_normals(V, F, face_norm);
    for (int i = 0; i < m; i++)
    {
        vec1 << ext_field(i, 0), ext_field(i, 1), ext_field(i, 2);
        vec2 << ext_field(i, 3), ext_field(i, 4), ext_field(i, 5);
        RowVector3d cross = vec1.cross(vec2);
        RowVector3d norm = face_norm.row(i);
        double swap = cross.dot(norm);
        if (swap < 0)
        {
            RowVector3d temp = vec1;
            vec1 = vec2;
            vec2 = temp;
            ext_field.row(i) << vec1, vec2, -vec1, -vec2;
        }
    }

    if (cutinit)
    {
        directional::PCFaceTangentBundle ftbcut;
        ftbcut.init(meshCut);
        out_field.init(ftbcut, directional::fieldTypeEnum::POLYVECTOR_FIELD, 4);
    }
    else
    {
        out_field.init(ftb, directional::fieldTypeEnum::POLYVECTOR_FIELD, 4);
    }

    out_field.set_extrinsic_field(ext_field);
    init_jacs.init(ftb, directional::fieldTypeEnum::POLYVECTOR_FIELD, 4);
    init_jacs.set_extrinsic_field(ext_jac);
    viewer.set_cartesian_field(out_field);

    // visualize
    directional::TriMesh mesh_flat;
    if (cutinit)
    {
        mesh_flat = meshCut;
    }
    else
    {
        mesh_flat = mesh;
    }
    mesh_flat.V.col(0) = UV.col(0);
    mesh_flat.V.col(1) = UV.col(1);
    mesh_flat.V.col(2).setZero();
    viewer.set_surface_mesh(mesh_flat, "pattern", 1);
}

void isometric_param(const MatrixXd &V, const MatrixXi &F, MatrixXd &UV)
{
    // arap parameterization
    igl::ARAPData arap_data;
    arap_data.with_dynamics = true; 
    Eigen::VectorXi b = Eigen::VectorXi::Zero(0);
    Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0, 0);
    arap_data.max_iter = 100;
    // 2 means that we're going to solve in 2d
    igl::arap_precomputation(V, F, 2, b, arap_data);
    igl::lscm(V, F, UV);
    igl::arap_solve(bc, arap_data, UV);

    // visualize
    directional::TriMesh mesh_flat;
    mesh_flat = mesh;
    mesh_flat.V.col(0) = UV.col(0);
    mesh_flat.V.col(1) = UV.col(1);
    mesh_flat.V.col(2).setZero();
    viewer.set_surface_mesh(mesh_flat, "pattern", 1);
}

void S_from_J(const MatrixXd &V, const MatrixXi &F, const MatrixXd &UV, directional::CartesianField &S)
{
    // set up jacobians
    SparseMatrix<double> Dx, Dy;
    vector<MatrixXd> local_basis;
    computeSurfaceGradientMatrix(V, F, Dx, Dy, local_basis);
    SparseMatrix<double> J;
    vector<vector<SparseMatrix<double>>> mat;
    VectorXd Ju, Jv;
    mat.resize(2);
    mat[0].push_back(Dx);
    mat[1].push_back(Dy);
    igl::cat(mat, J);
    int m = F.rows();

    Ju = J * UV.col(0);
    Jv = J * UV.col(1);

    MatrixXd ext_field(m, 3 * 4);
    MatrixXd face_norm;
    igl::per_face_normals(V, F, face_norm);

    double after_stretch_error = 0;
    double shrink_target = 1.0 / shrink_factor;

    for (int f = 0; f < m; f++)
    {
        Eigen::Matrix2d U, S, Vf, Juv;
        Juv << Ju(f), Ju(f + m), Jv(f), Jv(f + m);
        SSVD2x2(Juv, U, S, Vf);
        Matrix2d stretch = S * Vf.transpose();

        for (int i = 0; i < 2; i++)
        {
            double s=std::abs(S(i,i));
            double closest = (std::abs(s - 1.0) < std::abs(s - shrink_target)) ? 1.0 : shrink_target;
            after_stretch_error += mesh.faceAreas(f) * (s - closest) * (s - closest);
        }

        // get global direction
        RowVector3d gs1 = stretch(0, 0) * local_basis[0].row(f) + stretch(0, 1) * local_basis[1].row(f);
        RowVector3d gs2 = stretch(1, 0) * local_basis[0].row(f) + stretch(1, 1) * local_basis[1].row(f);
        // set field and check ccw
        RowVector3d cross = gs1.cross(gs2);
        RowVector3d norm = face_norm.row(f);
        double swap = cross.dot(norm);
        if (swap < 0)
        {
            RowVector3d temp = gs1;
            gs1 = gs2;
            gs2 = temp;
        }

        ext_field.row(f) << gs1, gs2, -1 * gs1, -1 * gs2;
    }
    cat_energy.push_back(make_tuple(2, after_stretch_error));
    S.set_extrinsic_field(ext_field);
}

directional::CartesianField jac_smoothing(const directional::CartesianField &field, double smoothness=jac_smoothness)
{
    int m = mesh.F.rows();
    int N = field.N;
    
    // 1. Setup the data container
    directional::PolyVectorData data;
    data.N = N;
    data.tb = field.tb;
    data.signSymmetry = true;
    
    // 2. Set constant weights
    data.wSmooth = smoothness;
    data.wRoSy = 0.0;
    
    // 3. Define soft alignment to all faces using the input field's magnitude
    // Every face (i) and every vector (j) in the PolyVector is constrained
    data.constSpaces.resize(N * m);
    data.constVectors.resize(N * m, 3);
    data.wAlignment.resize(N * m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < N; j++) {
            int idx = N * i + j;
            data.constSpaces(idx) = i;
            
            // Extract vector j from face i and keep its magnitude
            data.constVectors.row(idx) = field.extField.block(i, j * 3, 1, 3);
            
            // Constant alignment weight for "soft" alignment
            data.wAlignment(idx) = 1.0; 
        }
    }

    // 4. Solve for the smoothed field
    directional::CartesianField smoothed_field;
    directional::polyvector_field(data, smoothed_field);

    VectorXcd fullDofs = data.reducMat * data.reducedDofs + data.reducRhs;
    double smoothness_energy = std::abs((fullDofs.adjoint() * data.smoothMat.adjoint() * data.WSmooth * data.smoothMat * fullDofs).coeff(0, 0) * (data.wSmooth / data.totalSmoothWeight));

    newest_smooth_jac=smoothness_energy;

    // 5. Convert to raw for the return type (matching original logic)
    directional::CartesianField rawField;
    directional::polyvector_to_raw(smoothed_field, rawField, N % 2 == 0);
    return rawField;
}

void optimizeee(directional::CartesianField &starting_S)
{
    // smooth stitching line and align isotropic vectors
    directional::PolyVectorData pvData;
    directional::CartesianField rawField;

    MatrixXd prev_S, curr_S;
    prev_S=starting_S.extField;
    curr_S=starting_S.extField;
    MatrixXd prev_J, curr_J;

    // at least 3 iterations, then stop once S changes by less than 0.05% of its norm (or max_iters is hit)
    int i=0;
    while (i<3||((curr_S-prev_S).norm()>0.0005*curr_S.norm() && i<max_iters))
    {
        i++;
        cout<<"iteration "<<i<<endl;
        save_field_rendering(starting_S, res_dir + "input_field.raw");
        smooth_orientation(starting_S, pvData);

        save_field_rendering(smoothed_field, res_dir + "smoothed_orientation.raw");

        viewer.set_cartesian_field(smoothed_field, "smoothed", 2);
        viewer.set_field_color({107.0 / 255.0, 8.0 / 255.0, 125.0 / 255.0}, 2);

        viewer.toggle_singularities(true); // set to singularities of curl free?
        // reproject magnitude
        recompute_magnitude(smoothed_field, starting_S, S_new); //this does not recompute it, just keep value as it but show it as the field, computationally not needed, more for visualization purposes, this is what next function will align to essentially
        prev_S=curr_S;
        curr_S=S_new.extField;
        viewer.set_cartesian_field(S_new, "reproj mag", 3);
        viewer.set_field_color({8.0 / 255.0, 125.0 / 255.0, 107.0 / 255.0}, 3);
        finaloutput = S_new;

        // consistency
        S = S_new;
        new_stitching_line_consistency(S, S_new); // returned field is smoothedraw
        viewer.set_cartesian_field(S_new, "cons", 4);
        save_field_rendering(S_new, res_dir + "magnitude_optimized.raw");

        viewer.set_field_color({200 / 255.0, 125.0 / 255.0, 107.0 / 255.0}, 4);

        // fit jacobians based on this updated stitching field, now P holds final to integrate field
        prev_J=P.extField;

        parameterization_field(S_new, meshCut.V, meshCut.F, UV, P);
        save_field_rendering(P, res_dir + "jacobian_field.raw");

        directional::principal_matching(P);

        //new: curl reduction loop with rot fitting --> get new singularity?
        for(int j=0; j<10; j++){
            rawField = jac_smoothing(P);
            save_field_rendering(rawField, res_dir + "smoothed_jacobian_field.raw");

            u_fitting(S_new, meshCut.V, meshCut.F, rawField, P);
            save_field_rendering(P, res_dir + "reprojected_smooth_field.raw");
        }
        //save 10th smoothed jacs
        cat_energy.push_back(make_tuple(3, newest_smooth_jac));
        // integrate
        parameterize(P, S);
        curr_J=P.extField;
        energy_diff.push_back((curr_J-prev_J).norm());

        viewer.set_cartesian_field(S, "integration", 5);
        save_field_rendering(S, res_dir + "updated_jacobians_from_integration.raw");

        viewer.set_field_color({0 / 255.0, 0 / 255.0, 0 / 255.0}, 5);
        // reconstruct S through principal directions of stretch
        S_from_J(meshCut.V, meshCut.F, UV, starting_S);
        save_field_rendering(starting_S, res_dir + "reconstructed_s_from_j.raw");

    }
    cout<<"number of iterations "<<i<<endl;
}

void save_field(directional::CartesianField &field, string file_loc, int comp = 0)
{
    directional::CartesianField combedfield;
    directional::principal_matching(field);
    directional::combing(field, combedfield);

    // Open the file for writing
    std::ofstream outFile(file_loc);

    if (!outFile.is_open())
    {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    // field.extField is an (n x 12) Eigen matrix
    // We want the first 3 columns of every row
    for (int i = 0; i < combedfield.extField.rows(); ++i)
    {
        RowVector3d vec0 = combedfield.extField.block(i, 3 * comp, 1, 3);
        outFile << vec0(0) << " "
                << vec0(1) << " "
                << vec0(2) << "\n";
    }

    outFile.close();
    std::cout << "Field successfully saved to " << file_loc << std::endl;
}

void load_streamlines(std::string filename)
{
    std::ifstream inFile(filename);
    if (!inFile.is_open())
        return;
    VectorXd facedata(mesh.F.rows());
    facedata.setZero();

    std::string line;
    while (std::getline(inFile, line))
    {
        if (line.empty())
            continue;

        std::vector<Eigen::Vector3d> currentPoints;
        std::vector<int> currentFaces;
        std::stringstream ss(line);

        double x, y, z;
        int fid;

        // Try to read a point (x,y,z)
        while (ss >> x >> y >> z)
        {
            currentPoints.emplace_back(x, y, z);

            // Now try to read a face ID.
            // If it succeeds, it's a segment. If it fails, we hit the last point.
            if (ss >> fid)
            {
                currentFaces.push_back(fid);
                facedata(fid) = 1;
            }
            else
            {
                // This was the last point on the line (no FID follows it)
                break;
            }
        }

        if (!currentPoints.empty())
        {
            streamlines.push_back(std::move(currentPoints));
            streamline_faces.push_back(std::move(currentFaces));
        }
    }

    inFile.close();
    viewer.set_surface_face_data(facedata, "linefaces");
}

void visualize_flattened_results(
    const std::vector<std::vector<Eigen::Vector3d>> &streamlines,
    const std::vector<std::vector<int>> &streamline_faces,
    const std::vector<std::vector<bool>> &mask,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &UV,
    const Eigen::MatrixXi &F_cut,
    std::vector<std::vector<Eigen::Vector3d>> &uv_streamlines,
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &uv_boundary)
{
    uv_streamlines.clear();

    // Containers for the single combined curve network
    std::vector<Eigen::Vector3d> all_nodes;
    std::vector<std::array<size_t, 2>> all_edges;

    for (size_t i = 0; i < streamlines.size(); ++i)
    {
        const auto &points = streamlines[i];
        const auto &faces = streamline_faces[i];

        if (i >= mask.size())
            continue;

        for (size_t j = 0; j < faces.size(); ++j)
        {
            // --- MASK CHECK ---
            if (j >= mask[i].size() || !mask[i][j])
            {
                continue;
            }

            int fid = faces[j];
            if (fid < 0 || fid >= F.rows())
                continue;

            // Geometry lookup
            Eigen::Vector3d v[3] = {V.row(F(fid, 0)), V.row(F(fid, 1)), V.row(F(fid, 2))};
            Eigen::Vector2d u[3] = {UV.row(F_cut(fid, 0)), UV.row(F_cut(fid, 1)), UV.row(F_cut(fid, 2))};

            // Barycentric mapping helper
            auto get_uv = [&](Eigen::Vector3d p)
            {
                Eigen::Vector3d v0 = v[1] - v[0], v1 = v[2] - v[0], v2 = p - v[0];
                double d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1);
                double d20 = v2.dot(v0), d21 = v2.dot(v1);
                double denom = d00 * d11 - d01 * d01;
                if (std::abs(denom) < 1e-12)
                    return u[0];
                double v_coord = (d11 * d20 - d01 * d21) / denom;
                double w_coord = (d00 * d21 - d01 * d20) / denom;
                return (Eigen::Vector2d)((1.0 - v_coord - w_coord) * u[0] + v_coord * u[1] + w_coord * u[2]);
            };

            Eigen::Vector2d uvS = get_uv(points[j]);
            Eigen::Vector2d uvE = get_uv(points[j + 1]);

            // add segment to global soup
            size_t startIdx = all_nodes.size();
            all_nodes.push_back({uvS.x(), uvS.y(), 0.0});
            all_nodes.push_back({uvE.x(), uvE.y(), 0.0});
            all_edges.push_back({startIdx, startIdx + 1});
        }
    }

    // --- Single Registration ---
    if (!all_nodes.empty())
    {
        // Registering with the same name overwrites the previous one automatically
        auto *psCurve = polyscope::registerCurveNetwork("Flattened Streamlines", all_nodes, all_edges);
        psCurve->setColor({0.2, 0.6, 1.0});
        psCurve->setRadius(0.0005);
    }
    else
    {
        // If the mask filtered EVERYTHING, remove the curve network from the view
        polyscope::removeCurveNetwork("Flattened Streamlines", false);
    }

    // --- 2. Visualize UV Outline (Boundary) ---
    // An edge is a boundary if it belongs to exactly one face in F_cut
    std::map<std::pair<int, int>, int> edgeCount;
    for (int i = 0; i < F_cut.rows(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int v1 = F_cut(i, j);
            int v2 = F_cut(i, (j + 1) % 3);
            // Sort to treat (v1, v2) the same as (v2, v1)
            edgeCount[{std::min(v1, v2), std::max(v1, v2)}]++;
        }
    }

    std::vector<Eigen::Vector3d> boundaryNodes;
    std::vector<std::array<size_t, 2>> boundaryEdges;
    for (auto const &[edge, count] : edgeCount)
    {
        if (count == 1)
        { // It's a boundary edge in UV space
            size_t startIdx = boundaryNodes.size();
            boundaryNodes.push_back({UV(edge.first, 0), UV(edge.first, 1), 0.0});
            boundaryNodes.push_back({UV(edge.second, 0), UV(edge.second, 1), 0.0});
            boundaryEdges.push_back({startIdx, startIdx + 1});
            uv_boundary.push_back({Eigen::Vector2d(UV(edge.first, 0), UV(edge.first, 1)),
                                   Eigen::Vector2d(UV(edge.second, 0), UV(edge.second, 1))});
        }
    }
    auto *psBoundary = polyscope::registerCurveNetwork("uv_outline", boundaryNodes, boundaryEdges);
    psBoundary->setColor({0.1, 0.1, 0.1}); // Black outline
    psBoundary->setRadius(0.001);          // Slightly thicker
}

void save_results_to_svg(
    const std::string &filename,
    const std::vector<std::vector<Eigen::Vector3d>> &streamlines,
    const std::vector<std::vector<int>> &streamline_faces,
    const std::vector<std::vector<bool>> &mask,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &UV,
    const Eigen::MatrixXi &F_cut,
    const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &uv_boundary)
{
    // 1. Setup SVG and TXT file paths
    std::ofstream svg(filename);
    std::string txt_filename = filename.substr(0, filename.find_last_of('.')) + "_barycentric.txt";
    std::ofstream txt(txt_filename);

    if (!svg.is_open() || !txt.is_open())
    {
        std::cerr << "Could not open files for export." << std::endl;
        return;
    }

    // --- 1. Calculate Bounding Box ---
    double minX = 1e10, minY = 1e10, maxX = -1e10, maxY = -1e10;
    for (int i = 0; i < UV.rows(); ++i)
    {
        minX = std::min(minX, UV(i, 0));
        minY = std::min(minY, UV(i, 1));
        maxX = std::max(maxX, UV(i, 0));
        maxY = std::max(maxY, UV(i, 1));
    }

    // Add a small margin (5%) so the lines aren't touching the edge of the "paper"
    double rawWidth = maxX - minX;
    double rawHeight = maxY - minY;
    double marginX = rawWidth * 0.05;
    double marginY = rawHeight * 0.05;

    double viewMinX = minX - marginX;
    double viewMinY = minY - marginY;
    double viewWidth = rawWidth + 2.0 * marginX;
    double viewHeight = rawHeight + 2.0 * marginY;

    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svg << "<svg width=\"" << viewWidth << "cm\" height=\"" << viewHeight << "cm\" "
        << "viewBox=\"" << viewMinX << " " << viewMinY << " " << viewWidth << " " << viewHeight << "\" "
        << "xmlns=\"http://www.w3.org/2000/svg\">\n";

    // Helper: Compute Barycentric and UV in one go
    auto get_bary_and_uv = [&](int fid, const Eigen::Vector3d &p, Eigen::Vector3d &bary) -> Eigen::Vector2d
    {
        Eigen::Vector3d v0 = V.row(F(fid, 0)), v1 = V.row(F(fid, 1)), v2 = V.row(F(fid, 2));
        Eigen::Vector2d u0 = UV.row(F_cut(fid, 0)), u1 = UV.row(F_cut(fid, 1)), u2 = UV.row(F_cut(fid, 2));

        Eigen::Vector3d f0 = v1 - v0, f1 = v2 - v0, f2 = p - v0;
        double d00 = f0.dot(f0), d01 = f0.dot(f1), d11 = f1.dot(f1);
        double d20 = f2.dot(f0), d21 = f2.dot(f1);
        double denom = d00 * d11 - d01 * d01;

        if (std::abs(denom) < 1e-12) {
            bary << 1.0, 0.0, 0.0;
            return u0;
        }

        double v_c = (d11 * d20 - d01 * d21) / denom;
        double w_c = (d00 * d21 - d01 * d20) / denom;
        double u_c = 1.0 - v_c - w_c;

        bary << u_c, v_c, w_c;
        return u_c * u0 + v_c * u1 + w_c * u2;
    };

    // --- 2. Process Streamlines ---
    svg << "  <g id=\"streamlines\" fill=\"none\" stroke=\"#3399FF\" stroke-width=\"0.02\">\n";
    
    txt << "# StreamlineID FaceID Bary_U Bary_V Bary_W\n";

    for (size_t i = 0; i < streamlines.size(); ++i)
    {
        bool in_polyline = false;
        Eigen::Vector2d last_uv;
        const double jump_threshold = 1e-6;
        for (size_t j = 0; j < streamline_faces[i].size(); ++j)
        {
            bool segment_valid = (i < mask.size() && j < mask[i].size() && mask[i][j]);
            if (segment_valid)
            {
                int fid = streamline_faces[i][j];
                Eigen::Vector3d baryS, baryE;
                Eigen::Vector2d uvS = get_bary_and_uv(fid, streamlines[i][j], baryS);
                Eigen::Vector2d uvE = get_bary_and_uv(fid, streamlines[i][j+1], baryE);

                // Write Barycentric coordinates to text file
                // We save the start point of the segment.
                // For the very last segment, we also save the endpoint.
                txt << i << " " << fid << " " << baryS.x() << " " << baryS.y() << " " << baryS.z() << "\n";
                if (j == streamline_faces[i].size() - 1) {
                     txt << i << " " << fid << " " << baryE.x() << " " << baryE.y() << " " << baryE.z() << "\n";
                }

                bool jumped = in_polyline && (uvS - last_uv).norm() > jump_threshold;
                if (!in_polyline || jumped)
                {
                    if (in_polyline) svg << "\" />\n";
                    svg << "    <polyline points=\"" << uvS.x() << "," << uvS.y() << " ";
                    in_polyline = true;
                }
                svg << uvE.x() << "," << uvE.y() << " ";
                last_uv = uvE;
            }
            else if (in_polyline)
            {
                svg << "\" />\n";
                in_polyline = false;
            }
        }
        if (in_polyline) svg << "\" />\n";
    }

    svg << "  </g>\n";
    txt.close();

     // --- 3. Process Boundary (Reordered into Polylines) ---
    svg << "  <g id=\"boundary\" fill=\"none\" stroke=\"#FF4444\" stroke-width=\"0.04\" stroke-linejoin=\"round\">\n";

    // A. Build an Adjacency List
    struct VecCompare
    {
        bool operator()(const Eigen::Vector2d &a, const Eigen::Vector2d &b) const
        {
            if (std::abs(a.x() - b.x()) > 1e-9)
                return a.x() < b.x();
            return a.y() < b.y() - 1e-9;
        }
    };

    // Store neighbors for each point
    std::map<Eigen::Vector2d, std::vector<Eigen::Vector2d>, VecCompare> adj;
    for (const auto &edge : uv_boundary)
    {
        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first);
    }

    // B. Traverse and generate polylines
    while (!adj.empty())
    {
        std::vector<Eigen::Vector2d> polyline;

        // Start at any available vertex
        auto it = adj.begin();
        Eigen::Vector2d start_node = it->first;
        polyline.push_back(start_node);

        Eigen::Vector2d current = start_node;
        bool found_next = true;

        while (found_next)
        {
            found_next = false;
            auto &neighbors = adj[current];

            if (!neighbors.empty())
            {
                Eigen::Vector2d next = neighbors.back();
                neighbors.pop_back(); // Remove this edge direction

                // Remove the back-reference from the neighbor to prevent "backtracking"
                auto &back_neighbors = adj[next];
                for (auto b_it = back_neighbors.begin(); b_it != back_neighbors.end(); ++b_it)
                {
                    if ((*b_it - current).norm() < 1e-9)
                    {
                        back_neighbors.erase(b_it);
                        break;
                    }
                }

                // Clean up map keys with no more edges
                if (neighbors.empty())
                    adj.erase(current);

                polyline.push_back(next);
                current = next;
                found_next = true;
            }
            else
            {
                // If current node is exhausted but the map isn't,
                // we clean up and move to the next "island" or loop
                adj.erase(current);
            }
        }

        // Write the gathered polyline to SVG
        if (polyline.size() > 1)
        {
            svg << "    <polyline points=\"";
            for (const auto &p : polyline)
            {
                svg << p.x() << "," << p.y() << " ";
            }
            svg << "\" />\n";
        }
    }
    svg << "  </g>\n";

    svg << "</svg>\n";
    svg.close();
}

std::vector<std::vector<bool>> get_streamline_mask(
    const directional::CartesianField &field,
    const std::vector<std::vector<Eigen::Vector3d>> &streamlines,
    const std::vector<std::vector<int>> &streamline_faces)
{

    std::vector<std::vector<bool>> mask;
    mask.reserve(streamlines.size());

    for (size_t i = 0; i < streamlines.size(); ++i)
    {
        const auto &points = streamlines[i];

        // Ensure we have a corresponding face list for this streamline
        if (i >= streamline_faces.size())
        {
            mask.push_back(std::vector<bool>(std::max(0, (int)points.size() - 1), false));
            continue;
        }

        const auto &faces = streamline_faces[i];
        std::vector<bool> current_curve_mask;

        // A curve with N points has N-1 segments
        int num_segments = std::max(0, (int)points.size() - 1);
        current_curve_mask.reserve(num_segments);

        for (size_t j = 0; j < num_segments; ++j)
        {
            bool keep = false;

            // 1. Bounds Safety Check
            // Ensure face index exists in the face list and is valid for the field
            if (j < faces.size())
            {
                int face_idx = faces[j];

                if (face_idx >= 0 && face_idx < (int)field.extField.rows())
                {

                    Eigen::Vector3d p_start = points[j];
                    Eigen::Vector3d p_end = points[j + 1];
                    Eigen::Vector3d segment_vec = p_end - p_start;
                    double len = segment_vec.norm();

                    if (len > 1e-10)
                    {
                        Eigen::Vector3d segment_dir = segment_vec / len;
                        double max_dot = -1.0;
                        int best_v = -1;

                        // 2. Find closest field direction
                        for (int v = 0; v < 4; ++v)
                        {
                            Eigen::Vector3d field_vec = field.extField.block(face_idx, 3 * v, 1, 3).transpose();
                            double f_norm = field_vec.norm();

                            if (f_norm > 1e-10)
                            {
                                double current_dot = segment_dir.dot(field_vec / f_norm);
                                // Note: if your field is unoriented, use std::abs(current_dot)
                                if (current_dot > max_dot)
                                {
                                    max_dot = current_dot;
                                    best_v = v;
                                }
                            }
                        }

                        // 3. Apply the threshold
                        if (best_v != -1)
                        {
                            double magnitude = field.extField.block(face_idx, 3 * best_v, 1, 3).norm();
                            if (magnitude > 1+0.5*(1/shrink_factor-1))
                            {
                                keep = true;
                            }
                        }
                    }
                }
            }
            current_curve_mask.push_back(keep);
        }
        mask.push_back(current_curve_mask);
    }

    return mask;
}

/**
 * Re-activates sequences of inactive segments if their cumulative length
 * is below a specified threshold.
 */
std::vector<std::vector<bool>> threshold_jump(
    const std::vector<std::vector<Eigen::Vector3d>> &streamlines,
    const std::vector<std::vector<bool>> &curr_mask,
    const double jump_thresh)
{

    // Create a copy of the mask to modify and return
    std::vector<std::vector<bool>> updated_mask = curr_mask;

    for (size_t k = 0; k < streamlines.size(); ++k)
    {
        const auto &curve = streamlines[k];
        auto &mask = updated_mask[k];

        // A curve with N points has N-1 segments
        size_t num_segments = mask.size();

        size_t i = 0;
        while (i < num_segments)
        {
            // We only care about sequences that are currently inactive (false)
            if (!mask[i])
            {
                size_t start_idx = i;
                double total_gap_dist = 0.0;

                // Traverse the sequence of consecutive inactive segments
                while (i < num_segments && !mask[i])
                {
                    // Distance between point i and i+1
                    total_gap_dist += (curve[i + 1] - curve[i]).norm();
                    i++;
                } // after this loop is done we are either at the end of the line or at an active again. if end of line, don't reactivate, if not check with threshold
                size_t end_idx = i; // exclusive

                // If the "jump" is too small, it's not worth the gap; make it active
                if (total_gap_dist < jump_thresh && i != num_segments)
                {
                    for (size_t j = start_idx; j < end_idx; ++j)
                    {
                        mask[j] = true;
                    }
                }
            }
            else
            {
                // Segment is already active, just move to the next one
                i++;
            }
        }
    }

    return updated_mask;
}

/**
 * Deactivates sequences of active segments if their cumulative length
 * is below a specified threshold.
 */
std::vector<std::vector<bool>> min_filter(
    const std::vector<std::vector<Eigen::Vector3d>> &streamlines,
    const std::vector<std::vector<bool>> &curr_mask,
    const double min_len)
{

    // Create a copy of the mask to modify and return
    std::vector<std::vector<bool>> updated_mask = curr_mask;

    for (size_t k = 0; k < streamlines.size(); ++k)
    {
        const auto &curve = streamlines[k];
        auto &mask = updated_mask[k];
        size_t num_segments = mask.size();

        size_t i = 0;
        while (i < num_segments)
        {
            // Identify a sequence of active (true) segments
            if (mask[i])
            {
                size_t start_idx = i;
                double total_active_dist = 0.0;

                // Accumulate distance for the active block
                while (i < num_segments && mask[i])
                {
                    total_active_dist += (curve[i + 1] - curve[i]).norm();
                    i++;
                }
                size_t end_idx = i;

                // If the entire continuous stitch is too short, discard it
                if (total_active_dist < min_len)
                {
                    for (size_t j = start_idx; j < end_idx; ++j)
                    {
                        mask[j] = false;
                    }
                }
            }
            else
            {
                // Skip inactive segments
                i++;
            }
        }
    }

    return updated_mask;
}

struct GridHasher
{
    std::size_t operator()(const Eigen::Vector3i &v) const
    {
        return std::hash<int>()(v.x()) ^ (std::hash<int>()(v.y()) << 1) ^ (std::hash<int>()(v.z()) << 2);
    }
};

// Helper structure to keep the main loop clean
struct SegmentInfo
{
    int curve_idx;
    int seg_idx;
    Eigen::Vector3d midpoint;
    Eigen::Vector3d direction;
};

std::vector<bool> thin_out_stitching_lines(
    std::vector<std::vector<Eigen::Vector3d>> &streamlines,
    std::vector<std::vector<bool>> &curr_mask,
    const double distance)
{
    const size_t num_curves = streamlines.size();
    const double cos_45 = std::cos(45.0 * M_PI / 180.0);
    const double inv_dist = 1.0 / distance;

    std::vector<bool> final_mask(num_curves, false);
    std::vector<int> sorted_indices(num_curves);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    // 1. Pre-calculate lengths and segment data
    std::vector<double> lengths(num_curves, 0.0);
    std::vector<std::vector<Eigen::Vector3d>> segment_dirs(num_curves);
    std::vector<std::vector<uint8_t>> segment_active(num_curves);

    for (size_t i = 0; i < num_curves; ++i)
    {
        if (streamlines[i].size() < 2)
            continue;

        segment_dirs[i].reserve(streamlines[i].size() - 1);
        segment_active[i].reserve(curr_mask[i].size());

        for (size_t j = 0; j < streamlines[i].size() - 1; ++j)
        {
            Eigen::Vector3d edge = streamlines[i][j + 1] - streamlines[i][j];
            double seg_len = edge.norm();
            lengths[i] += seg_len;

            // Store normalized direction; handle zero-length segments
            if (seg_len > 1e-9)
            {
                segment_dirs[i].push_back(edge / seg_len);
            }
            else
            {
                segment_dirs[i].push_back(Eigen::Vector3d::Zero());
            }
            segment_active[i].push_back(curr_mask[i][j] ? 1 : 0);
        }
    }

    // Sort descending: longest curves first
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b)
              { return lengths[a] > lengths[b]; });

    // 2. Build Spatial Hash of all segments (using midpoints)
    std::unordered_map<Eigen::Vector3i, std::vector<SegmentInfo>, GridHasher> grid;
    for (int i = 0; i < (int)num_curves; ++i)
    {
        for (int j = 0; j < (int)segment_dirs[i].size(); ++j)
        {
            Eigen::Vector3d mid = (streamlines[i][j] + streamlines[i][j + 1]) * 0.5;
            Eigen::Vector3i cell = (mid * inv_dist).array().floor().cast<int>();
            grid[cell].push_back({i, j, mid, segment_dirs[i][j]});
        }
    }

    // 3. Greedy Thinning
    for (int idx : sorted_indices)
    {
        // Check if this curve still has any "active" segments
        bool has_active_parts = false;
        for (uint8_t active : segment_active[idx])
        {
            if (active)
            {
                has_active_parts = true;
                break;
            }
        }

        if (!has_active_parts)
            continue;

        // Accept the curve
        final_mask[idx] = true;

        // Use ALL segments of this accepted curve to invalidate nearby parallel segments
        for (size_t j = 0; j < segment_dirs[idx].size(); ++j)
        {
            Eigen::Vector3d cur_mid = (streamlines[idx][j] + streamlines[idx][j + 1]) * 0.5;
            Eigen::Vector3d cur_dir = segment_dirs[idx][j];
            Eigen::Vector3i base_cell = (cur_mid * inv_dist).array().floor().cast<int>();

            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        auto it = grid.find(base_cell + Eigen::Vector3i(dx, dy, dz));
                        if (it == grid.end())
                            continue;

                        for (auto &other : it->second)
                        {
                            if (other.curve_idx == idx)
                                continue;
                            if (segment_active[other.curve_idx][other.seg_idx] == 0)
                                continue;

                            // Distance check
                            if ((other.midpoint - cur_mid).norm() <= distance)
                            {
                                // Angle check: Redundant only if angle < 45 degrees
                                // We use absolute dot product because streamlines representing
                                // the same field might be oriented oppositely.
                                double dot = std::abs(cur_dir.dot(other.direction));
                                if (dot > cos_45)
                                {
                                    segment_active[other.curve_idx][other.seg_idx] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return final_mask;
}

void split_lines(
    const std::vector<std::vector<Eigen::Vector3d>>& streamlines,
    const std::vector<std::vector<bool>>& mask,
    const std::vector<std::vector<int>>& streamline_faces,
    std::vector<std::vector<Eigen::Vector3d>>& streamlines_new,
    std::vector<std::vector<bool>>& mask_new,
    std::vector<std::vector<int>>& streamline_faces_new)
{
    streamlines_new.clear();
    mask_new.clear();
    streamline_faces_new.clear();

    for (size_t i = 0; i < streamlines.size(); ++i)
    {
        const auto& line   = streamlines[i];
        const auto& lmask  = mask[i];
        const auto& lfaces = streamline_faces[i];

        // mask has one entry per segment, so n_segments = line.size() - 1
        // lmask.size() == lfaces.size() == line.size() - 1
        
        std::vector<Eigen::Vector3d> current_pts;
        std::vector<bool>            current_mask;
        std::vector<int>             current_faces;

        for (size_t s = 0; s < lmask.size(); ++s)
        {
            if (lmask[s])
            {
                // Start a new sub-line if we don't have one yet
                if (current_pts.empty())
                {
                    current_pts.push_back(line[s]);     // segment start
                }
                current_pts.push_back(line[s + 1]);     // segment end
                current_mask.push_back(true);
                current_faces.push_back(lfaces[s]);
            }
            else
            {
                // Flush whatever we've accumulated so far
                if (!current_pts.empty())
                {
                    streamlines_new.push_back(std::move(current_pts));
                    mask_new.push_back(std::move(current_mask));
                    streamline_faces_new.push_back(std::move(current_faces));

                    current_pts.clear();
                    current_mask.clear();
                    current_faces.clear();
                }
            }
        }

        // Flush any remaining sub-line at the end of this streamline
        if (!current_pts.empty())
        {
            streamlines_new.push_back(std::move(current_pts));
            mask_new.push_back(std::move(current_mask));
            streamline_faces_new.push_back(std::move(current_faces));
        }
    }
}

void split_lines_self_intersection(
    const std::vector<std::vector<Eigen::Vector3d>>& streamlines,
    const std::vector<std::vector<bool>>& mask,
    const std::vector<std::vector<int>>& streamline_faces,
    double r,
    std::vector<std::vector<Eigen::Vector3d>>& streamlines_new,
    std::vector<std::vector<bool>>& mask_new,
    std::vector<std::vector<int>>& streamline_faces_new)
{
    streamlines_new.clear();
    mask_new.clear();
    streamline_faces_new.clear();

    for (size_t i = 0; i < streamlines.size(); ++i)
    {
        const auto& line   = streamlines[i];
        const auto& lmask  = mask[i];
        const auto& lfaces = streamline_faces[i];

        const size_t n_segs = lmask.size();

        // Precompute segment lengths and cumulative arc-length at each segment midpoint
        std::vector<double> seg_len(n_segs);
        std::vector<double> arc(n_segs); // arc-length at midpoint of segment s
        double cumlen = 0.0;
        for (size_t s = 0; s < n_segs; ++s)
        {
            seg_len[s] = (line[s + 1] - line[s]).norm();
            arc[s] = cumlen + 0.5 * seg_len[s];
            cumlen += seg_len[s];
        }

        // cut[s] = true: flush before segment s and start a new sub-line
        std::vector<bool> cut(n_segs, false);

        for (size_t s = 0; s < n_segs; ++s)
        {
            const Eigen::Vector3d mid_s = 0.5 * (line[s] + line[s + 1]);

            for (size_t t = 0; t < n_segs; ++t)
            {
                if (t == s) continue;

                // Skip segments whose midpoints are within arc-length r of s
                if (std::abs(arc[s] - arc[t]) < r) continue;

                const Eigen::Vector3d mid_t = 0.5 * (line[t] + line[t + 1]);
                if ((mid_s - mid_t).norm() < r)
                {
                    cut[s] = true;
                    break;
                }
            }
        }

        // Walk segments and flush at cut points
        std::vector<Eigen::Vector3d> current_pts;
        std::vector<bool>            current_mask;
        std::vector<int>             current_faces;

        for (size_t s = 0; s < n_segs; ++s)
        {
            if (cut[s] && !current_pts.empty())
            {
                streamlines_new.push_back(std::move(current_pts));
                mask_new.push_back(std::move(current_mask));
                streamline_faces_new.push_back(std::move(current_faces));

                current_pts.clear();
                current_mask.clear();
                current_faces.clear();
            }

            if (current_pts.empty())
                current_pts.push_back(line[s]);

            current_pts.push_back(line[s + 1]);
            current_mask.push_back(lmask[s]);
            current_faces.push_back(lfaces[s]);
        }

        if (!current_pts.empty())
        {
            streamlines_new.push_back(std::move(current_pts));
            mask_new.push_back(std::move(current_mask));
            streamline_faces_new.push_back(std::move(current_faces));
        }
    }
}

void show_rel_seg(
    const std::vector<std::vector<Eigen::Vector3d>>& streamlines,
    const std::vector<std::vector<bool>>& mask)
{
    // Split into contiguous true runs first
    std::vector<std::vector<Eigen::Vector3d>> lines_vis;

    // Inline minimal split so this function is self-contained
    for (size_t i = 0; i < streamlines.size(); ++i)
    {
        const auto& line  = streamlines[i];
        const auto& lmask = mask[i];

        std::vector<Eigen::Vector3d> cur;
        for (size_t s = 0; s < lmask.size(); ++s)
        {
            if (lmask[s])
            {
                if (cur.empty()) cur.push_back(line[s]);
                cur.push_back(line[s+1]);
            }
            else if (!cur.empty())
            {
                lines_vis.push_back(std::move(cur));
                cur.clear();
            }
        }
        if (!cur.empty())
            lines_vis.push_back(std::move(cur));
    }

    // Register each sub-line as a separate curve network with a random color
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.2f, 1.0f);

    for (size_t k = 0; k < lines_vis.size(); ++k)
    {
        const auto& pts = lines_vis[k];
        if (pts.size() < 2) continue;

        // Build flat vertex + edge arrays for polyscope
        std::vector<glm::vec3> verts(pts.size());
        for (size_t v = 0; v < pts.size(); ++v)
            verts[v] = { (float)pts[v].x(), (float)pts[v].y(), (float)pts[v].z() };

        std::vector<std::array<size_t, 2>> edges(pts.size() - 1);
        for (size_t e = 0; e + 1 < pts.size(); ++e)
            edges[e] = { e, e + 1 };

        std::string name = "streamline_" + std::to_string(k);
        auto* cn = polyscope::registerCurveNetwork(name, verts, edges);

        glm::vec3 color = { dist(rng), dist(rng), dist(rng) };
        cn->setColor(color);
        cn->setRadius(0.001f); // adjust to your scene scale
    }

    //polyscope::show();
}

// menu
void myMenu()
{
    ImGui::Text("Inverse Design of Shrinking Thread Embroidery");
    ImGui::InputDouble("ratio", &s_ratio, 0, 0);
    if (ImGui::Button("skeletonization"))
    {
        if (!cutinit)
        {
            isometric_param(mesh.V, mesh.F, UV);
            local_global_init(mesh.V, mesh.F, UV, 1, 60, field);
            local_global_init(mesh.V, mesh.F, UV, s_ratio, 2, field);
        }
        S = field;
        save_field_rendering(field, res_dir + "initial_S.raw");
        P = field;
        cat_energy.push_back(make_tuple(-1, 10000));
    }

    ImGui::InputFloat("Smoothness factor", &c_smooth, 0, 0);
    ImGui::InputDouble("jac smooth", &jac_smoothness, 0, 0);
    ImGui::InputDouble("cons factor", &c_cons, 0, 0);
    ImGui::InputInt("max iterations", &max_iters, 0, 0);
    if (ImGui::Button("loop operations"))
    {
        optimizeee(field);
    }
    
    ImGui::Text("Pattern sampling");

    if (ImGui::Button("do tracing"))
    {
        // save mesh with parameterization for tracing
        directional::writeOBJ(res_dir + "shrinky_out.obj", mesh, UV, meshCut.F);//switched to mesh instead of meshcut
        // save stitching field for tracing
        save_field(finaloutput, res_dir + "shrinky_out0.faceField", 0);
        // do tracing (by executing binary)
        std::system(("../trace-streamlines " + res_dir + "shrinky_out.obj --viz --input-field " + res_dir + "shrinky_out0.faceField --nSym=4  --svg-name " + res_dir + "flatlines0.svg > " + res_dir + "trace.log").c_str()); // tracer output is very verbose, keep it in a log instead of the console
        // the binary always writes its streamlines to ../liness.strm, move them into res and read them into this code
        std::error_code ec;
        std::filesystem::rename("../liness.strm", res_dir + "liness.strm", ec);
        if (ec)
            cerr << "Could not move ../liness.strm into " << res_dir << ": " << ec.message() << endl;
        load_streamlines(res_dir + "liness.strm");
        cout << "streamline tracing completed" << endl;
    }
    ImGui::InputDouble("jump/min thresh", &jump_thresh, 0, 0);
    ImGui::InputDouble("sampling dist", &stitch_dist, 0, 0);
    if (ImGui::Button("postprocess"))
    {
        std::vector<std::vector<bool>> existing_curves = get_streamline_mask(field, streamlines, streamline_faces);
        std::vector<std::vector<bool>> connected_existing_curves = threshold_jump(streamlines, existing_curves, jump_thresh); // do jumping
        std::vector<std::vector<bool>> min_filtered = min_filter(streamlines, connected_existing_curves, jump_thresh);        // discard tiny stitches//bug on multiple calls
        std::vector<std::vector<Eigen::Vector3d>> streamlines_new;
        std::vector<std::vector<bool>> mask_new;
        std::vector<std::vector<int>> streamline_faces_new;
        split_lines(streamlines, min_filtered,streamline_faces,streamlines_new,mask_new,streamline_faces_new);
        split_lines_self_intersection(streamlines_new,mask_new,streamline_faces_new,stitch_dist,streamlines,final_mask,streamline_faces);

        std::vector<bool> thinned_curves = thin_out_stitching_lines(streamlines, final_mask, stitch_dist);

        for (int i = 0; i < thinned_curves.size(); i++)
        {
            if (!thinned_curves[i])
            { // need to deactivate entire curve
                for (int j = 0; j < final_mask[i].size(); j++)
                {
                    final_mask[i][j] = false;
                }
            }
        }
        show_rel_seg(streamlines,final_mask);

        visualize_flattened_results(streamlines, streamline_faces, final_mask, mesh.V, mesh.F, UV, meshCut.F, uv_streamlines, uv_boundary);
        save_results_to_svg(res_dir + "pattern.svg", streamlines, streamline_faces, final_mask, mesh.V, mesh.F, UV, meshCut.F, uv_boundary);
    }
    if (ImGui::Button("save obj with uv"))
    {
        directional::writeOBJ(res_dir + "shrinking_param.obj", meshCut, UV, meshCut.F);
    }
    if (ImGui::Button("print energy"))
    {
        print_energy();
    }

    /*if (ImGui::Checkbox("smoothed field", &show_smooth))
    {
        viewer.toggle_cartesian_field(show_smooth, 2);
    }
    if (ImGui::Checkbox("curl-reduced field", &show_curl))
    {
        viewer.toggle_cartesian_field(show_curl, 3);
    }
    if (ImGui::Checkbox("show main", &show_main))
    {
        viewer.toggle_cartesian_field(show_main, 0);
    }
    if (ImGui::Checkbox("cons", &show_cons))
    {
        viewer.toggle_cartesian_field(show_cons, 4);
    }
    if (ImGui::Checkbox("integration show", &show_intjac))
    {
        viewer.toggle_cartesian_field(show_intjac, 5);
    }
    ImGui::Checkbox("cut init", &cutinit);*/
}

int main(int argc, char *argv[])
{
    // Load mesh
    string data_folder = "";
    string filename = "pringles2.obj"; // default
    if (argc > 1)
    { // read from command line
        filename = argv[1];
    }
    directional::readOBJ(data_folder + filename, mesh);
    ftb.init(mesh);
    meshCut = mesh;
    std::filesystem::create_directories(res_dir);

    viewer.init();
    viewer.set_surface_mesh(mesh, "input", 0);
    viewer.set_callback(myMenu);
    viewer.launch();
}
 