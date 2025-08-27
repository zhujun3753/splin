#ifndef COMMON_LIB_HPP
#define COMMON_LIB_HPP

#include "predefined_types.h"

#include <nav_msgs/Odometry.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>

#include <gtsam/geometry/Pose3.h>
#include <tf/tf.h>

#define SKEW_SYM_MATRIX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define PI_M (3.14159265358)
template <typename T = double>
T cot(const T theta)
{
  return 1.0 / std::tan(theta);
}

template <typename T>
Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3)
{
  T &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
  Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
  if (norm > 0.00001)
  {
    T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
    Eigen::Matrix<T, 3, 3> K;
    K << SKEW_SYM_MATRIX(r_ang);
    /// Roderigous Tranformation
    return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
  }
  return Eye3;
}

template <typename T = double>
inline Eigen::Matrix<T, 3, 3> vec_to_hat(Eigen::Matrix<T, 3, 1> &omega)
{
  Eigen::Matrix<T, 3, 3> Omega;
  Omega << T(0), -omega(2), omega(1),
      omega(2), T(0), -omega(0),
      -omega(1), omega(0), T(0);
  return Omega;
}

template<typename T=double>
Eigen::Matrix<T, 3, 3> hat_d(const Eigen::Vector3d v)
{
    Eigen::Matrix<T, 3, 3> Omega;
    Omega <<  0, -v(2),  v(1)
        ,  v(2),     0, -v(0)
        , -v(1),  v(0),     0;
    return Omega;
}

template <typename T>
Eigen::Matrix<T, 3, 1> SO3_LOG(const Eigen::Matrix<T, 3, 3> &R)
{
  T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

template <typename T = double>
Eigen::Matrix<T, 3, 3> inverse_left_jacobian_of_rotation_matrix(Eigen::Matrix<T, 3, 1> &omega)
{
  // Barfoot, Timothy D, State estimation for robotics. Page 232-237
  Eigen::Matrix<T, 3, 3> res_mat_33;
  T theta = omega.norm();
  if (std::isnan(theta) || theta == 0)
    return Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 1> a = omega / theta;
  Eigen::Matrix<T, 3, 3> hat_a = vec_to_hat(a);
  res_mat_33 = (theta / 2) * (cot(theta / 2)) * Eigen::Matrix<T, 3, 3>::Identity() + (1 - (theta / 2) * (cot(theta / 2))) * a * a.transpose() + (theta / 2) * hat_a;
  // cout << "Omega: " << omega.transpose() << endl;
  // cout << "Res_mat_33:\r\n"  <<res_mat_33 << endl;
  return res_mat_33;
}

Eigen::Quaterniond EulerToEigenQuat(double roll, double pitch, double yaw)
{
  double c1 = cos(roll * 0.5);
  double s1 = sin(roll * 0.5);
  double c2 = cos(pitch * 0.5);
  double s2 = sin(pitch * 0.5);
  double c3 = cos(yaw * 0.5);
  double s3 = sin(yaw * 0.5);
  return Eigen::Quaterniond(c1 * c2 * c3 - s1 * s2 * s3, s1 * c2 * c3 + c1 * s2 * s3, -s1 * c2 * s3 + c1 * s2 * c3, c1 * c2 * s3 + s1 * s2 * c3);
}

Eigen::Matrix3d EulerToRotM(double roll, double pitch, double yaw)
{
  double cx = cos(roll);
  double sx = sin(roll);
  double cy = cos(pitch);
  double sy = sin(pitch);
  double cz = cos(yaw);
  double sz = sin(yaw);
  Eigen::Matrix3d R;
  R << cy * cz, -cy * sz, sy,
      cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx,
      sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy;
  return R;
}

Eigen::Matrix3d QuatToRotM(double w, double x, double y, double z)
{
  Eigen::Matrix3d R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return R;
}

Pose6D OdomMsgToPose6D(const nav_msgs::Odometry::ConstPtr &odom_msg)
{
  auto x = odom_msg->pose.pose.position.x;
  auto y = odom_msg->pose.pose.position.y;
  auto z = odom_msg->pose.pose.position.z;
  auto qx = odom_msg->pose.pose.orientation.x;
  auto qy = odom_msg->pose.pose.orientation.y;
  auto qz = odom_msg->pose.pose.orientation.z;
  auto qw = odom_msg->pose.pose.orientation.w;
  double roll, pitch, yaw;
  tf::Matrix3x3(tf::Quaternion(qx, qy, qz, qw)).getRPY(roll, pitch, yaw);
  return Pose6D{x, y, z, roll, pitch, yaw};
}

gtsam::Pose3 GeoPoseMsgToGTSPose(const geometry_msgs::Pose &pose)
{
  auto x = pose.position.x;
  auto y = pose.position.y;
  auto z = pose.position.z;
  auto qx = pose.orientation.x;
  auto qy = pose.orientation.y;
  auto qz = pose.orientation.z;
  auto qw = pose.orientation.w;
  double roll, pitch, yaw;
  tf::Matrix3x3(tf::Quaternion(qx, qy, qz, qw)).getRPY(roll, pitch, yaw);
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
}

gtsam::Pose3 Pose6DToGTSPose(const Pose6D &p)
{
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z));
}

void Pose6DToEigenRT(const Pose6D &p, gtsam::Matrix3 &R, gtsam::Vector3 &t)
{
  R = gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw).matrix();
  t = gtsam::Point3(p.x, p.y, p.z).vector();
}

Eigen::Matrix4f GeoPoseMsgToEigenM4f(const geometry_msgs::PoseWithCovarianceConstPtr &lc_msg)
{
  auto x = lc_msg->pose.position.x;
  auto y = lc_msg->pose.position.y;
  auto z = lc_msg->pose.position.z;
  auto qx = lc_msg->pose.orientation.x;
  auto qy = lc_msg->pose.orientation.y;
  auto qz = lc_msg->pose.orientation.z;
  auto qw = lc_msg->pose.orientation.w;
  //  Eigen::Quaterniond q_2(qx, qy, qz, qw);
  //  Eigen::Matrix3f R = q_2.toRotationMatrix().cast<float>();
  Eigen::Matrix3f R = QuatToRotM(qw, qx, qy, qz).cast<float>();
  Eigen::Vector3f t(x, y, z);
  Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
  out.block<3, 3>(0, 0) = R;
  out.block<3, 1>(0, 3) = t;
  return out;
}

Eigen::Matrix4f GTSPoseToEigenM4f(const gtsam::Pose3 &pose)
{
  gtsam::Point3 t = pose.translation();
  gtsam::Rot3 R = pose.rotation();
  auto col1 = R.column(1); // Point3
  auto col2 = R.column(2); // Point3
  auto col3 = R.column(3); // Point3

  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  out << col1.x(), col2.x(), col3.x(), t.x(),
      col1.y(), col2.y(), col3.y(), t.y(),
      col1.z(), col2.z(), col3.z(), t.z(),
      0, 0, 0, 1;
  return out.cast<float>();
}

Pose6D GTSPoseToPose6D(const gtsam::Pose3 &pose)
{
  gtsam::Point3 t = pose.translation();
  gtsam::Rot3 R = pose.rotation();
  Pose6D out;
  out.roll = R.roll();
  out.pitch = R.pitch();
  out.yaw = R.yaw();
  out.x = t.x();
  out.y = t.y();
  out.z = t.z();
  return out;
}

void CutVoxel3d(std::unordered_map<VOXEL_LOC, int> &feat_map, const pcl::PointCloud<PointType>::Ptr pl_feat, float voxel_box_size)
{
  uint plsize = pl_feat->size();
  for (uint i = 0; i < plsize; i++)
  {
    // Transform point to world coordinate
    PointType &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_tran(p_c.x, p_c.y, p_c.z);

    // Determine the key of hash table
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_box_size;
      if (loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

    // Find corresponding voxel
    PointType a_pt;
    auto iter = feat_map.find(position);
    if (iter != feat_map.end())
    {
    }
    else // If not finding, build a new voxel
    {
      feat_map[position] = 0;
    }
  }
}

bool CheckIfJustPlane(const pcl::PointCloud<PointType>::Ptr &cloud_in, const float &thr)
{
  pcl::SACSegmentation<PointType> seg;
  pcl::PointIndices inliners;
  pcl::ModelCoefficients coef;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.3);
  seg.setInputCloud(cloud_in);
  seg.segment(inliners, coef);
  float ratio = float(inliners.indices.size()) / float(cloud_in->size());
  //  opt_debug_file <<  "plane_ratio: " << ratio << std::endl;
  if (ratio > thr)
    return true;
  return false;
}

void DownsampleCloud(pcl::PointCloud<PointType> &cloud_in, const float &leafsize)
{
  if (cloud_in.empty())
    return;
  if (leafsize < 0.01)
    return;
  pcl::PointCloud<PointType> cloud_ds;
  pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
  cloud_tmp->points = cloud_in.points;
  cloud_tmp->height = cloud_in.height;
  cloud_tmp->width = cloud_in.width;
  cloud_tmp->header = cloud_in.header;
  pcl::VoxelGrid<PointType> sor;
  sor.setInputCloud(cloud_tmp);
  sor.setLeafSize(leafsize, leafsize, leafsize);
  sor.filter(cloud_ds);
  cloud_in = cloud_ds;
}

struct PlaneSimple
{
  Eigen::Vector3d center, center_0; // 实际中心和其中一个小平面的中心
  Eigen::Matrix3d covariance;
  Eigen::Vector3d normal;   // 最小特征值对应的向量
  Eigen::Vector3d y_normal; // 中间特征值对应的向量
  Eigen::Vector3d x_normal; // 最大特征值对应的向量
  float min_eigen_value = 0;
  float mid_eigen_value = 0;
  float max_eigen_value = 0;
  float d = 0;
  int points_size = 0, points_size_lio = 0;
  bool is_plane = false;
  bool is_merged = false;
  int type = -1;           // -1 非平面，0 普通平面 1 融合平面
  int local_plane_id = -1; // 在全局平面中的id
  float min_norm = 0, max_norm = 0;
  bool ready_to_pca = false;
  float yaw_scale = 0.0f, pitch_scale = 0.0f;
  bool is_filtered = false; // 平面参数已经确定且不再改变
  PlaneSimple *merged_plane = nullptr;
  int global_plane_id = -1, global_plane_loop_id = -1;
  double weight = -1;
  int sphere_id = -1;
  int voxel_ds_size = 0.05;

  PlaneSimple()
  {
    covariance = Eigen::Matrix3d::Zero();
    center = Eigen::Vector3d::Zero();
    normal = Eigen::Vector3d::Zero();
    y_normal = Eigen::Vector3d::Zero();
    x_normal = Eigen::Vector3d::Zero();
  }

  ~PlaneSimple()
  {
    merged_plane = nullptr;
  }

  PlaneSimple(float &yaw_scale_, float &pitch_scale_)
  {
    covariance = Eigen::Matrix3d::Zero();
    center = Eigen::Vector3d::Zero();
    normal = Eigen::Vector3d::Zero();
    y_normal = Eigen::Vector3d::Zero();
    x_normal = Eigen::Vector3d::Zero();
    yaw_scale = yaw_scale_;
    pitch_scale = pitch_scale_;
  }

  double get_weight()
  {
    if (min_eigen_value < 0)
      weight = 0.0;
    if (weight < 0)
      weight = 100 * sqrt(mid_eigen_value * max_eigen_value) * exp(-1 * sqrt(min_eigen_value)) * sqrt(points_size);
    return weight;
  }

  std::vector<Eigen::Vector3d> get_ellipsoid(double scale = 1.0) const
  {
    if (max_eigen_value <= 0 || mid_eigen_value <= 0 || min_eigen_value <= 0)
      return std::vector<Eigen::Vector3d>();
    if (std::isnan(max_eigen_value) || std::isnan(mid_eigen_value) || std::isnan(min_eigen_value))
      return std::vector<Eigen::Vector3d>();
    double a = scale * sqrt(max_eigen_value), b = scale * sqrt(mid_eigen_value), c = scale * sqrt(min_eigen_value);
    Eigen::Matrix3d Rot;
    Rot.col(0) = x_normal;
    Rot.col(1) = y_normal;
    Rot.col(2) = normal;
    // std::cout<<"a,b,c: "<<a<<", "<<b<<", "<<c<<std::endl;
    int sample_n = std::max(int(a * 10), 30);
    std::vector<double> cos_u(sample_n);
    std::vector<double> cos_v(sample_n);
    std::vector<double> sin_u(sample_n);
    std::vector<double> sin_v(sample_n);
    for (int i = 0; i < sample_n; i++)
    {
      double u = double(i) / sample_n * 2 * PI_M;
      double v = double(i) / sample_n * PI_M;
      cos_u[i] = cos(u);
      sin_u[i] = sin(u);
      cos_v[i] = cos(v);
      sin_v[i] = sin(v);
    }
    std::vector<Eigen::Vector3d> xyz(sample_n * sample_n);
    for (int i = 0; i < sample_n; i++)
    {
      for (int j = 0; j < sample_n; j++)
      {
        xyz[j + sample_n * i][0] = a * cos_u[i] * sin_v[j];
        xyz[j + sample_n * i][1] = b * sin_u[i] * sin_v[j];
        xyz[j + sample_n * i][2] = c * cos_v[j];
        xyz[j + sample_n * i] = Rot * xyz[j + sample_n * i] + center;
      }
    }
    return xyz;
  }

  void clear_lio_data()
  {
    points_size_lio = 0;
    is_filtered = false;
  }

  void recover_for_insert()
  {
    if (!ready_to_pca)
      return;
    covariance = (covariance + center * center.transpose()) * points_size;
    center = center * points_size;
    ready_to_pca = false;
  }

  void cal_cov_and_center()
  {
    if (ready_to_pca)
      return;
    center = center / points_size;
    covariance = covariance / points_size - center * center.transpose();
    ready_to_pca = true;
  }

  // 3*\sqrt(\lambda_3) 和最大距离很接近
  float simple_max_dist()
  {
    return 3 * sqrt(min_eigen_value);
  }

  float get_max_eigen_dist()
  {
    return sqrt(max_eigen_value);
  }

  float get_mid_eigen_dist()
  {
    return sqrt(mid_eigen_value);
  }

  float get_min_eigen_dist()
  {
    return sqrt(min_eigen_value);
  }
};

struct GlobalPlaneSimple // 全局平面
{
  Eigen::Matrix<double, 4, 4> C;
  std::vector<int> frame_ids;                               // 每个平面对应的雷达帧在划窗中的id
  std::vector<Eigen::Matrix<double, 4, 4>> C_ijs;           // C_{i,j}
  std::vector<Eigen::Matrix<double, 4, 4>> T_j_C_ijs;       // T_j*C_{i,j}
  std::vector<Eigen::Matrix<double, 4, 4>> T_j_C_ij_T_j_Ts; // T_j*C_{i,j}*T_j^T
  std::vector<PlaneSimple *> plane_ptrs;
  std::vector<Eigen::Vector3d> lg_normals;
  Eigen::Vector3d lambdas, center, center_g0; // center_g0 第一个子平面的中心在全局坐标系下的坐标
  Eigen::Vector3d xis[3];
  Eigen::Matrix<double, 6, 4> Vs[3];
  int id;                      // 在全局平面中的id
  bool optimized_flag = false; // 判断是否参与优化，C是否有效
  bool has_eigen = false;
  double weight = 0.0;
  Eigen::Vector3d normal;
  double param_d;
  int last_frame_id = -1;
  PlaneSimple *first_plane_ptr = nullptr;
  PlaneSimple *last_plane_ptr = nullptr;

  GlobalPlaneSimple()
  {
    C = Eigen::Matrix<double, 4, 4>::Zero();
    center = Eigen::Vector3d::Zero();
  }

  ~GlobalPlaneSimple()
  {
    first_plane_ptr = nullptr;
    last_plane_ptr = nullptr;
    frame_ids.clear();
    C_ijs.clear();
    T_j_C_ijs.clear();
    T_j_C_ij_T_j_Ts.clear();
    plane_ptrs.clear();
    lg_normals.clear();
  }

  void clear()
  {
    frame_ids.clear();
    C_ijs.clear();
    plane_ptrs.clear();
  }

  double get_weight()
  {
    return weight;
  }

  Eigen::Vector3d get_center() const
  {
    return center;
  }

  Eigen::Vector3d get_min_eigen_vec()
  {
    return xis[0];
  }

  void eigen_solve()
  {
    has_eigen = true;
    double N_i = C(3, 3);                              // N_i
    Eigen::Vector3d v_bar = C.block<3, 1>(0, 3) / N_i; // 1/N_i*v_i = 1/N_i*S_p*C_i*F
    center = v_bar;
    // A_i = P_i/N_i - v_i*v_i^T/N_i/N_i =  P_i/N_i - (v_i/N_i)*(v_i/N_i)^T
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) / N_i - v_bar * v_bar.transpose());
    lambdas = saes.eigenvalues();
    if (lambdas[0] < 0 || lambdas[1] < 0 || lambdas[2] < 0 || isnan(lambdas[0]) || isnan(lambdas[1]) || isnan(lambdas[2])) // Bad eigen!
    {
      // std::cout<<"0  lambdas: "<<lambdas.transpose()<<", N_i: "<<N_i<<std::endl;
      lambdas[0] = lambdas[1] = lambdas[2] = 0.0;
      has_eigen = false;
    }
    Eigen::Matrix3d eigenvectors = saes.eigenvectors();
    xis[0] = eigenvectors.col(0);
    xis[1] = eigenvectors.col(1);
    xis[2] = eigenvectors.col(2);
  }

  std::vector<Eigen::Vector3d> get_ellipsoid(double scale = 1.0) const
  {
    if (!has_eigen)
      return std::vector<Eigen::Vector3d>();
    double a = scale * sqrt(lambdas[2]), b = scale * sqrt(lambdas[1]), c = scale * sqrt(lambdas[0]);
    if (isnan(a))
      a = 0.0;
    if (isnan(b))
      b = 0.0;
    if (isnan(c))
      c = 0.0;
    Eigen::Matrix3d Rot;
    Rot.col(0) = xis[2];
    Rot.col(1) = xis[1];
    Rot.col(2) = xis[0];
    // std::cout<<"a,b,c: "<<a<<", "<<b<<", "<<c<<std::endl;
    int sample_n = std::max(int(a * 10), 30);
    std::vector<double> cos_u(sample_n);
    std::vector<double> cos_v(sample_n);
    std::vector<double> sin_u(sample_n);
    std::vector<double> sin_v(sample_n);
    for (int i = 0; i < sample_n; i++)
    {
      double u = double(i) / sample_n * 2 * PI_M;
      double v = double(i) / sample_n * PI_M;
      cos_u[i] = cos(u);
      sin_u[i] = sin(u);
      cos_v[i] = cos(v);
      sin_v[i] = sin(v);
    }
    std::vector<Eigen::Vector3d> xyz(sample_n * sample_n);
    double N_i = C(3, 3);
    // Eigen::Vector3d center = C.block<3, 1>(0, 3)/N_i;
    Eigen::Vector3d center = get_center();
    for (int i = 0; i < sample_n; i++)
    {
      for (int j = 0; j < sample_n; j++)
      {
        xyz[j + sample_n * i][0] = a * cos_u[i] * sin_v[j];
        xyz[j + sample_n * i][1] = b * sin_u[i] * sin_v[j];
        xyz[j + sample_n * i][2] = c * cos_v[j];
        xyz[j + sample_n * i] = Rot * xyz[j + sample_n * i] + center;
      }
    }
    return xyz;
  }

  int get_first_frame_id()
  {
    if (frame_ids.size() > 0)
      return frame_ids[0];
    else
      return -1;
  }

  PlaneSimple *get_first_plane_ptr()
  {
    return first_plane_ptr;
  }

  int frame_size()
  {
    return frame_ids.size();
  }

};

#endif // COMMON_LIB_HPP
