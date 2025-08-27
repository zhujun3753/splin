#include "keyframe_containner.hpp"
#include "TunningPointPairsFactor.h"
#include "common_lib.hpp"
#include "Octree.hpp"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/UInt64.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/exceptions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/Marginals.h>

#include <deque>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <math.h>
#include <unordered_map>
#include <unistd.h>
#include <omp.h>

// #include "common_lib.h"


typedef struct BinaryDescriptor {
	std::vector<bool> occupy_array_;
	unsigned char summary_;
	Eigen::Vector3d location_;
} BinaryDescriptor;

// 1kb,12.8
typedef struct STD {
	Eigen::Vector3d triangle_;
	Eigen::Vector3d angle_;
	Eigen::Vector3d center_;
	unsigned short frame_number_;
	// std::vector<unsigned short> score_frame_;
	// std::vector<Eigen::Matrix3d> position_list_;
	BinaryDescriptor binary_A_;
	BinaryDescriptor binary_B_;
	BinaryDescriptor binary_C_;
} STD;

class STD_LOC {
public:
	int64_t x, y, z, a, b, c;

	STD_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0,
			int64_t vb = 0, int64_t vc = 0)
		: x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

	bool operator==(const STD_LOC &other) const {
	return (x == other.x && y == other.y && z == other.z);
	// return (x == other.x && y == other.y && z == other.z && a == other.a &&
	//         b == other.b && c == other.c);
	}
};

namespace std {
template <> struct hash<STD_LOC> {
	int64_t operator()(const STD_LOC &s) const {
	using std::hash;
	using std::size_t;
	return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
	}
};
} // namespace std
typedef struct STDMatchList {
	std::vector<std::pair<STD, STD>> match_list_;
	int match_frame_;
	double mean_dis_;
} STDMatchList;

namespace loopdetection // 加个名称空间，避免同名类或者函数
{
typedef pcl::PointXYZINormal PointType;

typedef struct Plane {
	PointType p_center_;
	Eigen::Vector3d center_;
	Eigen::Vector3d normal_;
	Eigen::Matrix3d covariance_;
	float radius_ = 0;
	float min_eigen_value_ = 1;
	float d_ = 0;
	int id_ = 0;
	int sub_plane_num_ = 0;
	int points_size_ = 0;
	bool is_plane_ = false;

	std::vector<Eigen::Vector3d> get_ellipsoid(double scale = 1.0) const
	{
		double N_i = points_size_; // N_i
		Eigen::Vector3d center = center_; // 1/N_i*v_i = 1/N_i*S_p*C_i*F
		Eigen::Matrix3d covariance = covariance_;
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);
		Eigen::Vector3d lambdas = saes.eigenvalues();
		if(lambdas[0]<0 || lambdas[1]<0 || lambdas[2]<0 || isnan(lambdas[0]) || isnan(lambdas[1]) || isnan(lambdas[2])) // Bad eigen!
		{
			std::cout<<", lambdas: "<<lambdas.transpose()<<std::endl;
			return std::vector<Eigen::Vector3d>();
		}
		Eigen::Matrix3d eigenvectors = saes.eigenvectors();
		Eigen::Vector3d normal   = eigenvectors.col(0);
		Eigen::Vector3d y_normal = eigenvectors.col(1);
		Eigen::Vector3d x_normal = eigenvectors.col(2);
		float min_eigen_value = lambdas(0);
		float mid_eigen_value = lambdas(1);
		float max_eigen_value = lambdas(2);
		if (max_eigen_value <= 0 || mid_eigen_value <= 0 || min_eigen_value <= 0) return std::vector<Eigen::Vector3d>();
		if (std::isnan(max_eigen_value) || std::isnan(mid_eigen_value) || std::isnan(min_eigen_value)) return std::vector<Eigen::Vector3d>();
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
		for (int j = 0; j < sample_n; j++)
		{
			xyz[j + sample_n * i][0] = a * cos_u[i] * sin_v[j];
			xyz[j + sample_n * i][1] = b * sin_u[i] * sin_v[j];
			xyz[j + sample_n * i][2] = c * cos_v[j];
			xyz[j + sample_n * i] = Rot * xyz[j + sample_n * i] + center;
		}
		return xyz;
	}

} Plane;

class OctoTree {
public:
	std::vector<Eigen::Vector3d> voxel_points_;
	Plane *plane_ptr_;
	int layer_;
	int octo_state_; // 0 is end of tree, 1 is not
	int merge_num_ = 0;
	bool is_project_ = false;
	std::vector<Eigen::Vector3d> project_normal;
	bool is_publish_ = false;
	OctoTree *leaves_[8];
	double voxel_center_[3]; // x, y, z
	float quater_length_;
	bool init_octo_;
	double plane_detection_thre = 0.01; // 平面检测阈值，视为平面的最小特征值
	int voxel_init_num = 10; // 至少10个点才能拟合平面

	OctoTree(double plane_detection_thre = 0.01, int voxel_init_num = 10)
	{
		voxel_points_.clear();
		octo_state_ = 0;
		layer_ = 0;
		init_octo_ = false;
		for (int i = 0; i < 8; i++) {
			leaves_[i] = nullptr;
		}
		plane_ptr_ = new Plane;
	}

	~OctoTree()
	{
		delete plane_ptr_;
	}

	void init_octo_tree()
	{
		if (voxel_points_.size() > voxel_init_num) // 10
		{
			init_plane();
		}
	}

	void init_plane()
	{
		plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
		plane_ptr_->center_ = Eigen::Vector3d::Zero();
		plane_ptr_->normal_ = Eigen::Vector3d::Zero();
		plane_ptr_->points_size_ = voxel_points_.size();
		plane_ptr_->radius_ = 0;
		for (auto pi : voxel_points_) {
			plane_ptr_->covariance_ += pi * pi.transpose();
			plane_ptr_->center_ += pi;
		}
		plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
		plane_ptr_->covariance_ = plane_ptr_->covariance_ / plane_ptr_->points_size_ - plane_ptr_->center_ * plane_ptr_->center_.transpose();
		Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
		Eigen::Matrix3cd evecs = es.eigenvectors();
		Eigen::Vector3cd evals = es.eigenvalues();
		Eigen::Vector3d evalsReal;
		evalsReal = evals.real();
		Eigen::Matrix3d::Index evalsMin, evalsMax;
		evalsReal.rowwise().sum().minCoeff(&evalsMin);
		evalsReal.rowwise().sum().maxCoeff(&evalsMax);
		int evalsMid = 3 - evalsMin - evalsMax;
		if (evalsReal(evalsMin) < plane_detection_thre) { // 0.01
			plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
			plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
			plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
			plane_ptr_->is_plane_ = true;
			plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) + plane_ptr_->normal_(1) * plane_ptr_->center_(1) + plane_ptr_->normal_(2) * plane_ptr_->center_(2));
			plane_ptr_->p_center_.x = plane_ptr_->center_(0);
			plane_ptr_->p_center_.y = plane_ptr_->center_(1);
			plane_ptr_->p_center_.z = plane_ptr_->center_(2);
			plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
			plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
			plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
		} else {
			plane_ptr_->is_plane_ = false;
		}
	}
	
};

struct LoopDetectionResult
{
	Eigen::Matrix4d lc_transform;
	int current_submap_id;
	int target_submap_id;
	double icp_score;
};

struct KeyFramePose
{
	int key_frame_id;
	Eigen::Matrix4d key_frame_pose;
	Eigen::Matrix4d key_frame_pose_opt;
	Eigen::Matrix<double,6,6> key_frame_pose_cov;
	bool pose_opt_set;
	ros::Time key_frame_time;

	KeyFramePose()
	{
		pose_opt_set = false;
		key_frame_pose_cov = Eigen::Matrix<double,6,6>::Zero();
	}
	~KeyFramePose(){}
};

struct SubMap
{
	int start_key_frame_id, end_key_frame_id;
	Eigen::Matrix4d sub_map_pose;
	Eigen::Matrix4d sub_map_pose_opt;
	Eigen::Matrix<double,6,6> sub_map_pose_cov;
	bool pose_opt_set;
	ros::Time sub_map_time;
	pcl::PointCloud<PointType>::Ptr sub_map_cloud;
	std::vector<PlaneSimple*> planes;
	std::vector<Plane> merged_planes;

	SubMap()
	{
		pose_opt_set = false;
		sub_map_cloud.reset(new pcl::PointCloud<PointType>());
		sub_map_pose_cov = Eigen::Matrix<double,6,6>::Zero();
	}
	~SubMap(){}
};

struct LCFactor
{
	gtsam::Pose3 pose_prev, pose_curr;
	Eigen::Matrix4d lc_transform0;
	Eigen::Matrix4d lc_transform;
	Eigen::Matrix4d prev_opt, curr_opt;
	int prev_idx, curr_idx;
	float pose_drift0 = 0.0;
	LCFactor(){}
	~LCFactor(){}
};

struct LCInfo
{
	int prev_idx, curr_idx;
	Eigen::Matrix4d lc_transform0;
	Eigen::Matrix4d lc_transform;
	float overlap_score0 = 0.0;
	float pose_drift0 = 0.0;
	LCInfo(){}
	~LCInfo(){}
};

// 仅仅用于检测回环并计算回环位姿，优化放在其他位置
class LoopClosureDetection {
public:
	ros::NodeHandle nh;
	ros::Subscriber sub_cloud;
	ros::Subscriber sub_odom;
	ros::Subscriber sub_timeCorrection;
	ros::Subscriber sub_aft_gppo_poses_relat;
	ros::Subscriber sub_key_frame_planes;
	ros::Publisher pubKFPathAftPGO;
	ros::Publisher pubPathAftPGO;
	ros::Publisher submap_info_pub;
	std::deque<nav_msgs::Odometry::ConstPtr> odom_buf;
	std::deque<ros::Time> time_buf;
	std::deque<pcl::PointCloud<PointType>::Ptr> lidar_buf;
	Eigen::Matrix4d current_pose;
	Eigen::Matrix4d last_pose;
	Eigen::Matrix4d lc_delta_pose; // 回环更新之后的位姿偏置，作用于在回环更新期间产生的子图
	std::vector<SubMap> sub_maps;
	bool is_sub_msg = false;
	std::mutex mtx_buffer;
	int scan_count = 0;
	std::string save_directory;
	std::fstream debug_file, time_lcd_file;
	std::vector<pcl::PointCloud<PointType>::Ptr> history_plane_list; // save all planes of key frame
	std::vector<std::vector<BinaryDescriptor>> history_binary_list; // save all binary descriptors of key frame
	std::vector<std::vector<STD>> history_STD_list; // save all STD descriptors of key frame
	// std::vector<LidarFrame*> key_frame_list;
	std::vector<LoopDetectionResult> lc_results;
	std::vector<KeyFramePose> key_frame_poses;
	std::unordered_map<STD_LOC, std::vector<STD>> STD_map; // hash table, save all descriptor
	int frame_number = 0;
	int sub_map_id = 0;
	pcl::PointCloud<PointType>::Ptr current_sub_map = nullptr; // 当前子图
	pcl::PointCloud<PointType>::Ptr corners_last_ = nullptr;
	pcl::PointCloud<PointType>::Ptr corners_curr_ = nullptr;
	std::vector<STD> cur_STD_list;
	double position_threshold = 0.2; // 构成子图时，帧位姿变化小于这个值就不记录帧数
	double rotation_threshold = DEG2RAD(5); // 构成子图时，帧位姿变化小于这个值就不记录帧数
	double ds_size = 0.5;   // 下采样网格大小
	int sub_frame_num = 20; // 构成子图的帧数量
	double plane_detection_thre = 0.01; // 平面检测阈值，视为平面的最小特征值
	int voxel_init_num = 10; // 至少10个点才能拟合平面
	double plane_voxel_size = 2.0; // 用于拟合平面的网格大小
	double plane_merge_normal_thre = 0.1; // 平面合并的最大法线差异阈值
	double plane_merge_dis_thre = 0.1; // 平面合并的最大距离差异阈值
	int proj_plane_num = 1; // 子图的投影平面数量
	int useful_corner_num = 30; // 最多保留30个关键点
	double summary_min_thre = 8; // 关键点处存在点的分段的数量至少为8
	double proj_image_resolution = 0.5; // 投影平面划分网格（像素）大小,即每个像素为0.5m*0.5m区域
	double proj_dis_min = 0.2; // 投影平面立柱的高度最小值
	double proj_dis_max = 5; // 投影平面上立柱的高度最大值
	double proj_image_high_inc = 0.1; // 投影平面上立柱的高度划分尺寸，即按照0.1m沿着平面法线方向划分立柱
	double non_max_suppression_radius = 3.0; // 进行非最大值抑制时候考虑的半径范围
	double std_side_resolution = 0.2; // 三角形边长分辨率，三角形边长离散化之前要乘以 1/0.2,即乘以5.0之后再转为整形int
	int descriptor_near_num = 20; // 搜索每个特征点周围的20个点建立三角形
	double descriptor_min_len = 3.0; // 边长范围
	double descriptor_max_len = 50.0; // 边长范围
	double icp_threshold = 0.15; // 几何校验的最小匹配率
	double rough_dis_threshold = 0.01; // 三角形匹配的边长差异 边长差异小于1%
	int skip_near_num = 50; // 匹配的时候不考虑最近的50个子图
	double similarity_threshold = 0.7; // 二值描述子相似度大于0.7
	int candidate_num = 5; // 逐步筛选匹配数量最高的帧，考虑匹配数量最高的5个帧，保存匹配帧id和其中所有匹配的特征
	int ransac_Rt_thr = 4; // 至少4个匹配特征在位姿变化后相差满足条件
	// 搜索当前点通过位姿变化后的点在历史数据中的最近点，如果最近点的法线差异小于0.1，点到面的距离差异小于0.5，则认为成功匹配
	double normal_threshold = 0.1; // 法线差异小于0.1
	double dis_threshold = 0.5; // 点到面的距离差异小于0.5
	float plane_inliner_ratio_thr = 0.4; // 如果平面点的占比超过 0.4  reject, the target cloud is like a plane
	float vs_for_ovlap = 2.0; // cut voxel for counting hit 2.0  构建一个目标子图点云所在位置的2.0m*2.0m空的网格,用于回环子图的重叠率计算
	float overlap_score_thr = 0.5; // check LC cloud pair overlap ratio  0.5 回环子图点云网格重合比例小于 50% 则视为回环失败
	bool associate_consecutive_frame = true; // 是否在因子图中建立相邻子图的顶点关联
	// gtsam
	std::unordered_map<VOXEL_LOC, gtsam::Pose3> lc_pose_uomap;
	std::unordered_map<VOXEL_LOC, std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>> cornerpairs_uomap;
	std::unordered_map<int, bool> val_inserted, val_inserted2;
	std::unordered_map<int, bool> prior_inserted;
	int relocal_status = -1;
	gtsam::FastVector<size_t> factors_invalid_ids, factors_invalid_ids2;
	std::mutex mtx_sub_, mtx_pgo_, mtx_keyf_read_;
	gtsam::NonlinearFactorGraph gts_graph_, gts_graph_2, gts_graph_recover_; // gts_graph_recover_ 不添加当前回环的位姿约束和顶点约束，便于错误回环之后恢复状态
	gtsam::ISAM2 *isam_;
	gtsam::ISAM2 *isam_2; // 根据子图位姿计算所有关键帧位姿
	gtsam::Values gts_init_vals_, gts_init_vals_2;
	gtsam::Values gts_cur_vals_, gts_cur_vals_2;
	gtsam::ISAM2Params parameters;
	gtsam::ISAM2Params parameters2;
	gtsam::NonlinearFactorGraph refined_graph, refined_graph2;
	std::unordered_map<int, bool> factor_invalid_ids_map, factor_invalid_ids_map2;
	gtsam::noiseModel::Diagonal::shared_ptr pose_start_noise_, pose_noise_, pose_noise2_;
	gtsam::noiseModel::Base::shared_ptr loopclousure_noise_, point_noise_, lc_point_noise_;
	gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
	int lc_curr_idx, lc_prev_idx, last_lc_prev_idx, last_lc_curr_idx;
	float overlap_percen;
	bool loop_closure_detected = false;
	int loop_corr_counter = 0;
	bool wait_1more_loopclosure = false;
	int fastlio_notify_type = 0;
	gtsam::Pose3 subpose_beforejump, subpose_afterjump;
	uint64_t jump_time = 0;
	double elapsed_ms_opt = 0.0;
	int opt_count = 0;
	double elapsed_ms_max = 0.0;
	int graphpose_count = 0;
	bool has_add_prior_node = false;
	int pairfactor_num = 6;
	float residual_thr = 2.0;
	double last_lidar_msgtime = 0, last_timestamp_odom=0;
	bool wait_front_end_done = false; // 当回环优化成功之后，等待前端处理完毕的信息，等待期间接收到的位姿直接用回环优化的位姿偏置处理
	double correction_time = -1; // 前端处理回环位姿结束之后的第一个雷达帧的时间戳，小于这个时间戳的位姿都有需要转换一下
	std::deque<nav_msgs::Path::ConstPtr> aft_gppo_poses_relat_buffer;
	std::deque<std_msgs::Float64MultiArray::ConstPtr> key_frame_planes_buffer;
	std::vector<LCFactor> new_added_lc_factors;
	std::vector<LCFactor> valid_added_lc_factors;
	std::vector<LCFactor> invalid_added_lc_factors;
	std::vector<LCInfo> rej_sml_ovl_lcs; // 因重合度较小而拒绝的回环
	bool use_key_frame_planes = false;
	thuni::Octree* octree_kf_plane_center4submap_ptr;
    std::vector<GlobalPlaneSimple*> kf_global_planes4submap;
	std::set<int> submap_ids_w_lc; // 检测到回环的子图id
    bool decouple_front = false; // 解耦前后端，前端的运行不受后端的影响，也就是回环检测之后不更新前端的搜索树和位姿

	LoopClosureDetection()
	{
		{
			nh.param< float > ( "loop_detection/vs_for_ovlap", vs_for_ovlap, 2.0 ); // cut voxel for counting hit 2.0  构建一个目标子图点云所在位置的2.0m*2.0m空的网格,用于回环子图的重叠率计算
			nh.param< double >( "loop_detection/proj_dis_min", proj_dis_min, 0.2 ); // 投影平面立柱的高度最小值
			nh.param< double >( "loop_detection/summary_min_thre", summary_min_thre, 8 ); // 关键点处存在点的分段的数量至少为8
			nh.param< int >   ( "loop_detection/descriptor_near_num", descriptor_near_num, 20 ); // 搜索每个特征点周围的20个点建立三角形
			nh.param< double >( "loop_detection/descriptor_min_len", descriptor_min_len, 3.0 ); // 边长范围
			nh.param< double >( "loop_detection/descriptor_max_len", descriptor_max_len, 50 ); // 边长范围
			nh.param< int >   ( "loop_detection/skip_near_num", skip_near_num, 50 ); // 匹配的时候不考虑最近的50个子图
			nh.param< double >( "loop_detection/normal_threshold", normal_threshold, 0.1 ); // 法线差异小于0.1
			nh.param< bool >  ( "loop_detection/associate_consecutive_frame", associate_consecutive_frame, true ); // 是否在因子图中建立相邻子图的顶点关联
			nh.param< bool >  ( "loop_detection/use_key_frame_planes", use_key_frame_planes, false ); // 是否在因子图中建立相邻子图的顶点关联
			nh.param< float > ( "loop_detection/overlap_score_thr", overlap_score_thr, 0.5 ); // 是否在因子图中建立相邻子图的顶点关联
			nh.param< bool  > ( "poslam_lio/decouple_front", decouple_front, false);
			nh.param< std::string  > ( "SaveDir", save_directory, std::string("/media/zhujun/0DFD06D20DFD06D2/ws_PPSLAM/src/SPLIN/output/tmp"));

		}
		time_lcd_file = std::fstream(save_directory + "/times_loopdetection_LTAOM.txt", std::fstream::out);
		time_lcd_file.precision(std::numeric_limits<double>::max_digits10);
		debug_file = std::fstream(save_directory + "/lc_debug.txt", std::fstream::out);
		last_pose = Eigen::Matrix4d::Identity();
		current_sub_map.reset(new pcl::PointCloud<PointType>);
		// gtsam
		std::cout<<"gtsam init"<<std::endl;
		parameters.relinearizeThreshold = 0.1;
		parameters.relinearizeSkip = 1;
		parameters.enableDetailedResults = true;
		isam_ = new gtsam::ISAM2(parameters);
		parameters2.relinearizeThreshold = 0.1;
		parameters2.relinearizeSkip = 1;
		parameters2.enableDetailedResults = true;
		isam_2 = new gtsam::ISAM2(parameters2);
		gtsam::Vector noise_vec6(6);
		noise_vec6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
		pose_start_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);
		noise_vec6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
		pose_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);
		gtsam::Vector noise_vec3(3);
		double cov1 = 0.000001;
		double cov2 = 0.01;
		double cov3 = 0.0001;
		noise_vec3 << cov1, cov1, cov1;
		point_noise_ = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1),gtsam::noiseModel::Diagonal::Variances(noise_vec3));
		noise_vec3 << cov2, cov2, cov2;
		lc_point_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec3);
		noise_vec6 << cov1, cov1, cov1, cov3, cov3, cov3;
		pose_noise2_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);
		double loopNoiseScore = 0.1; // constant is ok...
		gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
		robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
		// optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
		robustLoopNoise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1),gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
		std::cout<<"gtsam init end"<<std::endl;
		corners_last_.reset(new pcl::PointCloud<PointType>);
		corners_curr_.reset(new pcl::PointCloud<PointType>);
		sub_cloud = nh.subscribe("/key_frame_undistorted", 1000, &LoopClosureDetection::pointCloudCallBack, this, ros::TransportHints().tcpNoDelay());
		sub_odom = nh.subscribe("/key_frame_pose", 1000, &LoopClosureDetection::odomCallBack, this, ros::TransportHints().tcpNoDelay());
		sub_timeCorrection = nh.subscribe<std_msgs::UInt64>("/time_correction", 1000, &LoopClosureDetection::timeCorrectionCallback, this, ros::TransportHints().tcpNoDelay());
		sub_aft_gppo_poses_relat = nh.subscribe<nav_msgs::Path>("/aft_gppo_poses_relat", 1000, &LoopClosureDetection::aft_gppo_poses_relat_cbk, this, ros::TransportHints().tcpNoDelay());
		sub_key_frame_planes = nh.subscribe<std_msgs::Float64MultiArray>("/key_frame_planes", 1000, &LoopClosureDetection::key_frame_planes_cbk, this, ros::TransportHints().tcpNoDelay());
		pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_submaps_path", 100);
		pubKFPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_keyframes_path", 100);
		submap_info_pub = nh.advertise<nav_msgs::Odometry>("/submap_info", 100);
		octree_kf_plane_center4submap_ptr = new thuni::Octree();
		octree_kf_plane_center4submap_ptr->set_min_extent(0.1);
		octree_kf_plane_center4submap_ptr->set_bucket_size(1);
	}

	~LoopClosureDetection() 
	{
		delete isam_;
		delete isam_2;
		time_lcd_file.close();
		debug_file.close();
	}

	void pointCloudCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
	{
		mtx_buffer.lock();
		if (msg->header.stamp.toSec() < last_lidar_msgtime)
		{
			ROS_ERROR("lidar loop back, clear buffer");
			lidar_buf.clear();
			time_buf.clear();
		}
		pcl::PointCloud<PointType>::Ptr ptr(new pcl::PointCloud<PointType>);
		pcl::fromROSMsg(*msg, *ptr);
		lidar_buf.push_back(ptr);
		time_buf.push_back(msg->header.stamp);
		last_lidar_msgtime = msg->header.stamp.toSec();
		mtx_buffer.unlock();
	}

	void odomCallBack(const nav_msgs::Odometry::ConstPtr &msg)
	{
		double timestamp = msg->header.stamp.toSec();
		mtx_buffer.lock();
		if ( timestamp < last_timestamp_odom )
		{
			ROS_ERROR( "odom loop back, clear buffer" );
			odom_buf.clear();
		}
		last_timestamp_odom = timestamp;
		odom_buf.push_back(msg);
		mtx_buffer.unlock();
	}

	void timeCorrectionCallback(const std_msgs::UInt64::ConstPtr &msg)
	{
		correction_time = ros::Time().fromNSec(msg->data).toSec();
	}

	void aft_gppo_poses_relat_cbk(const nav_msgs::Path::ConstPtr &path_in)
    {
		mtx_buffer.lock();
        aft_gppo_poses_relat_buffer.push_back(path_in);
        mtx_buffer.unlock();
    }

	void key_frame_planes_cbk(const std_msgs::Float64MultiArray::ConstPtr& msg)
	{
		if(!use_key_frame_planes)
		{
			std::cout<<"use_key_frame_planes: "<<use_key_frame_planes<<std::endl;
			return;
		}
		mtx_buffer.lock();
		key_frame_planes_buffer.push_back(msg);
        mtx_buffer.unlock();
	}

	template <typename PCType = pcl::PointCloud<PointType>>
	void down_sampling_voxel(const PCType &pc_in, PCType &pc_out, float voxel_size = 1.0, bool no_sort = false, bool clear_data = false)
	{
		std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
		for (int i = 0; i < pc_in.points.size(); i++) {
			int64_t x = std::round(pc_in.points[i].x / voxel_size);
			int64_t y = std::round(pc_in.points[i].y / voxel_size);
			int64_t z = std::round(pc_in.points[i].z / voxel_size);
			VOXEL_LOC position(x, y, z);
			feat_map_tmp[position].push_back(i);
		}
		if (clear_data)
			pc_out.clear();
		// int min_ptn = 1e5;
		for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter)
		{
			if (no_sort)
			{
				pc_out.push_back(pc_in.points[iter->second[0]]);
				continue;
			}
			int best_id = 0;
			float min_dist = 1e8;
			int pt_n = iter->second.size();
			// if(pt_n<10) continue;
			// min_ptn = (min_ptn>pt_n && pt_n>10) ? pt_n : min_ptn;
			Eigen::Vector3f center(iter->first.x, iter->first.y, iter->first.z);
			center *= voxel_size;
			for (int i = 0; i < pt_n; i++)
			{
				int id = iter->second[i];
				float dist = (center - Eigen::Vector3f(pc_in.points[id].x, pc_in.points[id].y, pc_in.points[id].z)).norm();
				if (dist < min_dist)
				{
					min_dist = dist;
					best_id = id;
				}
			}
			pc_out.push_back(pc_in.points[best_id]);
		}
		// std::cout<<"min_ptn: "<<min_ptn<<std::endl;
	}

	template <typename PCType = pcl::PointCloud<PointType>>
	void down_sampling_voxel_eigen(const PCType &pc_in, std::vector<Eigen::Vector3d> &pc_out, float voxel_size = 1.0, bool no_sort = false, std::vector<VOXEL_LOC> *orig_pts_down_eigen_keys_ = nullptr) {
		std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
		for (int i = 0; i < pc_in.points.size(); i++) {
			int64_t x = std::round(pc_in.points[i].x / voxel_size);
			int64_t y = std::round(pc_in.points[i].y / voxel_size);
			int64_t z = std::round(pc_in.points[i].z / voxel_size);
			VOXEL_LOC position(x, y, z);
			feat_map_tmp[position].push_back(i);
		}
		pc_out.clear();
		// int min_ptn = 1e5;
		for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter) {
			if (orig_pts_down_eigen_keys_ != nullptr)
				orig_pts_down_eigen_keys_->push_back(iter->first);
			Eigen::Vector3d pt_ = Eigen::Vector3d(pc_in.points[iter->second[0]].x, pc_in.points[iter->second[0]].y, pc_in.points[iter->second[0]].z);
			if (no_sort) {
				pc_out.push_back(pt_);
				continue;
			}
			Eigen::Vector3d best_pt = pt_;
			float min_dist = 1e8;
			int pt_n = iter->second.size();
			// if(pt_n<10) continue;
			// min_ptn = (min_ptn>pt_n && pt_n>10) ? pt_n : min_ptn;
			Eigen::Vector3d center(iter->first.x, iter->first.y, iter->first.z);
			// center *= voxel_size; // 为什么有这个反而结果变差。。。
			for (int i = 0; i < pt_n; i++) {
				int id = iter->second[i];
				Eigen::Vector3d pt_1 = Eigen::Vector3d(pc_in.points[id].x, pc_in.points[id].y, pc_in.points[id].z);
				float dist = (center - pt_1).norm();
				if (dist < min_dist) {
					min_dist = dist;
					best_pt = pt_1;
				}
			}
			pc_out.push_back(best_pt);
		}
		// std::cout<<"min_ptn: "<<min_ptn<<std::endl;
	}

	double binary_similarity(const BinaryDescriptor &b1, const BinaryDescriptor &b2)
	{
		double dis = 0;
		for (size_t i = 0; i < b1.occupy_array_.size(); i++) {
			// to be debug hanming distance
			if (b1.occupy_array_[i] == true && b2.occupy_array_[i] == true) {
				dis += 1;
			}
		}
		return 2 * dis / (b1.summary_ + b2.summary_);
	}

	void triangle_solver(std::pair<STD, STD> &std_pair, Eigen::Matrix3d &std_rot, Eigen::Vector3d &std_t)
	{
		Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
		Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
		src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
		src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
		src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
		ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
		ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
		ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
		Eigen::Matrix3d covariance = src * ref.transpose();
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		std_rot = V * U.transpose();
		if (std_rot.determinant() < 0) {
			Eigen::Matrix3d K;
			K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
			std_rot = V * K * U.transpose();
		}
		std_t = -std_rot * std_pair.first.center_ + std_pair.second.center_;
	}

	//我习惯用eigen的矩阵，所以加了这个矩阵转换
	gtsam::Pose3 trans2gtsamPose(const Eigen::Matrix4d & matrix_)
	{
		gtsam::Rot3 rot3(matrix_(0,0), matrix_(0,1), matrix_(0,2),\
						matrix_(1,0), matrix_(1,1), matrix_(1,2),\
						matrix_(2,0), matrix_(2,1), matrix_(2,2));
		return gtsam::Pose3(rot3, gtsam::Point3(matrix_(0,3), matrix_(1,3), matrix_(2,3)));
	}

	Eigen::Matrix4d gtsam2transPose(const gtsam::Pose3 & pose)
	{
		Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
		matrix(0,3) = pose.x();
		matrix(1,3) = pose.y();
		matrix(2,3) = pose.z();
		matrix.block<3,3>(0,0) = pose.rotation().matrix().cast<double>();
		return matrix;
	}

	Eigen::Matrix4d calcu_loop_poses(int sub_map_id, int match_frame_id, float & min_dist_aver_, const Eigen::Matrix4d & lc_transform,  bool debug=false, bool display=false)
    {
		const pcl::PointCloud<PointType>::Ptr &currSubMapCloud = sub_maps[sub_map_id].sub_map_cloud;
		const pcl::PointCloud<PointType>::Ptr &targetSubMapCloud = sub_maps[match_frame_id].sub_map_cloud;
        Eigen::Matrix< double, 6, 6 > G, H_T_H, I_STATE, Jh_inv, Jh, H_T_R_inv_H, P_inv;
        G.setZero();
        H_T_H.setZero();
        H_T_R_inv_H.setZero(); // H^T * R^{-1} * H
        I_STATE.setIdentity();
        Jh_inv.setIdentity();
        Jh.setIdentity();
        P_inv.setIdentity();
        int lp_n = currSubMapCloud->size();
        std::vector<Eigen::Vector3d> gnormal_to_glp(lp_n); // 全局坐标系下每个局部平面点在对应的全局平面法线  glp 全局坐标系下的局部平面点
        std::vector<Eigen::Vector3d> gl_pts_lio(lp_n);
        std::vector<Eigen::Matrix3d> point_crossmats(lp_n);
        std::vector<float> gd_to_glp(lp_n); // 全局坐标系下每个局部平面点在对应的全局平面参数d  glp 全局坐标系下的局部平面点
        std::vector<float> dist_to_glp(lp_n); //全局坐标系下每个局部平面的每个点到对应全局平面的距离
        std::vector<float> pts_weights(lp_n); //全局坐标系下每个局部平面的每个点到对应全局平面的距离的权重
        std::vector<int> valid_lidar_pt_ids(lp_n);
        std::vector<int> lp_pts_ns(lp_n);
        double deltaT = 0.0, deltaR = 0.0;
        int rematch_num = 0;
        bool rematch_en = 0;
        int valid_lidar_pts_n = 0;
        P_inv = sub_maps[sub_map_id].sub_map_pose_cov.inverse(); // 迭代过程中不会更新协方差，减少不必要的计算
        Eigen::Matrix3d R_w_l0 = lc_transform.block<3,3>(0,0);
        Eigen::Vector3d t_w_l0 = lc_transform.block<3,1>(0,3);
        Eigen::Matrix3d R_w_l = lc_transform.block<3,3>(0,0);
        Eigen::Vector3d t_w_l = lc_transform.block<3,1>(0,3);
		if(display && t_w_l0.norm()==0) sub_map_id++; // 临时改变
		thuni::Octree search_tree_tmp;
		search_tree_tmp.initialize(*targetSubMapCloud);
        std::cout<<"targetSubMapCloud->size(): "<<targetSubMapCloud->size()<<", search_tree_tmp.size(): "<<search_tree_tmp.size()<<std::endl;
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr octree_pts(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        double solve_time = 0, match_time=0.0, kd_tree_search_time=0.0;
		int num_match_pts = 5, NUM_MAX_ITERATIONS=4;
		double lidar_pt_cov =0.00015;
		double lidar_cov_p = 1.02;
		min_dist_aver_ = 1000.0;
		std::vector<Eigen::Matrix3d> R_w_ls;
		std::vector<Eigen::Vector3d> t_w_ls;
        // ======4. 迭代优化======
        for ( int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ )
        {
            valid_lidar_pts_n = 0;
            if(debug) std::cout<<"iterCount: "<<iterCount<< std::endl;
            // 计算每个点对应的平面法线和有符号距离
			std::vector<std::vector<float>> points_near;
			std::vector<float> pointSearchSqDis_surf;
			float min_dist_total_ = 0.0, min_dist_num_ = 0.0;
			R_w_ls.push_back(R_w_l);
			t_w_ls.push_back(t_w_l);
			for(int pt_i=0; pt_i<lp_n; pt_i++)
			{
				Eigen::Vector3d pt_(currSubMapCloud->points[pt_i].x, currSubMapCloud->points[pt_i].y, currSubMapCloud->points[pt_i].z);
				gl_pts_lio[pt_i] = R_w_l*pt_ + t_w_l;
				if(iterCount == 0 || rematch_en)
				{
					if(iterCount == NUM_MAX_ITERATIONS - 1 && display)
					{
						pcl::PointXYZRGBNormal pt;
						pt.x = gl_pts_lio[pt_i](0);
						pt.y = gl_pts_lio[pt_i](1);
						pt.z = gl_pts_lio[pt_i](2);
						pt.r = 0;
						pt.g = 255;
						pt.b = 0;
						octree_pts->push_back(pt);
					}
					pts_weights[pt_i] = 0.0;
					search_tree_tmp.knnNeighbors_eigen(gl_pts_lio[pt_i], num_match_pts, points_near, pointSearchSqDis_surf);
					float max_distance = pointSearchSqDis_surf[ num_match_pts - 1 ];
					if(max_distance > 4.0) continue; //  超过0.5就无效了
					// 平面拟合
					Eigen::Vector3d normal_fit;
					float pd;
					double pt_weight = lidar_cov_p;
					bool planeValid = true;
					if(1)
					{
						Eigen::MatrixXd A(num_match_pts, 3);
						Eigen::VectorXd B(num_match_pts);
						for (int j = 0; j < num_match_pts; ++j) {
							A(j, 0) = points_near[j][0];  // x
							A(j, 1) = points_near[j][1];  // y
							A(j, 2) = points_near[j][2];  // z
							B(j) = -1;  // 常数项
						}
						Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
						Eigen::VectorXd solution = svd.solve(B);
						normal_fit << solution(0), solution(1), solution(2);
						double norm_ = normal_fit.norm() + 1e-6;
						pd = 1 / norm_;  // 平面到原点的距离（假设平面方程的右侧常数项是 -D）
						normal_fit /= norm_;  // 归一化法向量
						pt_weight=0;
						// 检查拟合平面的好坏，距离超过0.1就无效
						for( int j = 0; j < num_match_pts; j++)
						{
							float dist = fabs( normal_fit[0] * points_near[ j ][0] + normal_fit[1] * points_near[ j ][1] + normal_fit[2] * points_near[ j ][2] + pd );
							if(dist > 0.2) // Raw 0.10
							{
								planeValid = false;
								break;
							}
							pt_weight+=dist;
							if(0 && iterCount == NUM_MAX_ITERATIONS - 1 && display)
							{
								Eigen::Vector3d near_pt_(points_near[ j ][0], points_near[ j ][1], points_near[ j ][2]);
								Eigen::Vector3d dir_ = gl_pts_lio[pt_i]-near_pt_;
								double len = dir_.norm();
								dir_ /= len;
								double scale_ = 0.01;
								int pn = len/scale_;
								for(int k=0; k<pn; k++)
								{
									Eigen::Vector3d pt_eigen = near_pt_ + dir_*scale_*k;
									pcl::PointXYZRGBNormal pt;
									pt.x = pt_eigen(0);
									pt.y = pt_eigen(1);
									pt.z = pt_eigen(2);
									pt.r = 255;
									pt.g = 0;
									pt.b = 255;
									octree_pts->push_back(pt);
								}
							}
						}
						pt_weight = cos(pt_weight/num_match_pts*3.1415926*5);
					}
					if(!planeValid) continue;
					gnormal_to_glp[pt_i] = normal_fit;
					gd_to_glp[pt_i] = pd;
					pts_weights[pt_i] = pt_weight;
				}
				if(pts_weights[pt_i]<1e-9) continue;
				double pd2 = gnormal_to_glp[pt_i].dot(gl_pts_lio[pt_i]) + gd_to_glp[pt_i];
				if(display && abs(pd2)>0.4 && iterCount>=NUM_MAX_ITERATIONS - 2)
				{
					pts_weights[pt_i] = 0.0;
					continue;
				}
				dist_to_glp[pt_i] = pd2;
				valid_lidar_pts_n++;
				min_dist_num_++;
				min_dist_total_ += abs(pd2);
				if(iterCount == NUM_MAX_ITERATIONS - 1 && display)
				{
					Eigen::Vector3d near_pt_ = gl_pts_lio[pt_i] - (gnormal_to_glp[pt_i].dot(gl_pts_lio[pt_i])+gd_to_glp[pt_i])*gnormal_to_glp[pt_i]; // gl_pts[pt_i] 在平面的投影点
					Eigen::Vector3d dir_ = gl_pts_lio[pt_i]-near_pt_;
					double len = dir_.norm();
					// if(pt_i<5) std::cout<<"near_pt_: "<<near_pt_.transpose()<<", len: "<<len<<std::endl;
					dir_ /= len;
					double scale_ = 0.01;
					int pn = len/scale_;
					for(int k=0; k<pn; k++)
					{
						Eigen::Vector3d pt_eigen = near_pt_ + dir_*scale_*k;
						pcl::PointXYZRGBNormal pt;
						pt.x = pt_eigen(0);
						pt.y = pt_eigen(1);
						pt.z = pt_eigen(2);
						pt.r = 255;
						pt.g = 255;
						pt.b = 0;
						octree_pts->push_back(pt);
					}
				}
			}
			if(min_dist_num_>0)
			{
				float min_dist_aver_tmp = min_dist_total_/min_dist_num_;
				min_dist_aver_ = min_dist_aver_tmp;
			}
            // if(debug) std::cout<<"计算雅克比矩阵和测量向量: "<<iterCount<< std::endl;
            // if(process_debug) std::cout<<"EKF_solve_start"<<", valid_lidar_pts_n: "<<valid_lidar_pts_n<< std::endl;
            // 3. 计算雅克比矩阵和测量向量
            // 统计有效点数，并记录每个点在有效向量中的索引id
            Eigen::MatrixXd Hsub( valid_lidar_pts_n, 6 ); // 除了前6维度都是0，这里使用缩减版
            Eigen::VectorXd meas_vec( valid_lidar_pts_n );
            Eigen::MatrixXd H_T_R_inv( 6, valid_lidar_pts_n ); // H^T* R^{-1}
            Hsub.setZero();
            H_T_R_inv.setZero();
            Eigen::Matrix< double, 6, 1 > vec; // \check{x}_{op,k} - \check{x}_{k}
            Eigen::Matrix3d rotd(R_w_l * R_w_l0.transpose());
            vec.block<3, 1>(0, 0) = SO3_LOG(rotd);
            vec.block<3, 1>(3, 0) = t_w_l - rotd*t_w_l0;
            Eigen::Vector3d delta_theta =  vec.block<3,1>(0,0);
            Eigen::Matrix3d J_l_vec_inv = inverse_left_jacobian_of_rotation_matrix(delta_theta); // ==> 对应公式中的 J
            Eigen::Matrix3d t_crossmat;
            Eigen::Vector3d delta_t_tmp = vec.block<3, 1>(3, 0);
            t_crossmat << SKEW_SYM_MATRIX( delta_t_tmp );
            Jh.block<3,3>(0,0) = J_l_vec_inv;
            Jh.block<3,3>(3,0) = -t_crossmat;
            Jh_inv = Jh.inverse();
			int laserCloudSelNum_i = 0;
			for(int pt_i=0; pt_i<lp_n; pt_i++)
			{
				if(pts_weights[pt_i]<1e-9) continue;
				const Eigen::Vector3d & normal_ = gnormal_to_glp[pt_i];
				Eigen::Matrix3d point_crossmat;
				point_crossmat << SKEW_SYM_MATRIX( gl_pts_lio[pt_i] );
				//* 转置，而point_crossmat没转置，就是添加负号！！
				Eigen::Vector3d A = point_crossmat * normal_;
				Hsub.block<1, 6>(laserCloudSelNum_i, 0)  << A[0], A[1], A[2], normal_[0], normal_[1], normal_[2];
				Hsub.block<1, 6>(laserCloudSelNum_i, 0) = Hsub.block<1, 6>(laserCloudSelNum_i, 0)*Jh_inv; // H*J = H*Jh_inv
				H_T_R_inv.block<6, 1>(0, laserCloudSelNum_i) = Hsub.block<1, 6>(laserCloudSelNum_i, 0).transpose()/lidar_pt_cov*pts_weights[pt_i];
				meas_vec( laserCloudSelNum_i ) = -dist_to_glp[pt_i];
				laserCloudSelNum_i++;
			}
            Eigen::MatrixXd K( 6, valid_lidar_pts_n );
            H_T_R_inv_H.block<6,6>(0,0) = H_T_R_inv * Hsub;
            K = ( P_inv*0.01 + H_T_R_inv_H ).inverse().block< 6, 6 >( 0, 0 ) * H_T_R_inv;
            Eigen::Matrix< double, 6, 1 > solution = K * ( meas_vec + Hsub * vec.block< 6, 1 >( 0, 0 ) ); // 见式子 2.81
            R_w_l = Exp(solution(0), solution(1), solution(2))*R_w_l0;
            t_w_l = Exp(solution(0), solution(1), solution(2))*t_w_l0 + solution.block<3, 1>(3, 0);
            Eigen::Vector3d rot_add = solution.block< 3, 1 >( 0, 0 );
            Eigen::Vector3d t_add = solution.block< 3, 1 >( 3, 0 );
            bool flg_EKF_converged = false;
            if( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) ) flg_EKF_converged = true;
            deltaR = rot_add.norm() * 57.3;
            deltaT = t_add.norm() * 100;
            rematch_en = false;
            // if(flg_EKF_converged || ( ( rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) )
            {
                rematch_en = true;
                rematch_num++;
            }
            // if(rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) // Fast lio ori version.
            if(iterCount == NUM_MAX_ITERATIONS - 1 ) // Fast lio ori version.
            {
				std::cout<<"min_dist_aver_: "<<min_dist_aver_<<std::endl;
                break;
            }
            if(debug) std::cout<<"EKF_solve_end"<< std::endl;
        }
		Eigen::Matrix4d lc_transform2 = Eigen::Matrix4d::Identity();
		lc_transform2.block<3,3>(0,0) = R_w_l;
		lc_transform2.block<3,1>(0,3) = t_w_l;
        if(1)
        {
            Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
			T1 = lc_transform;
			T2 = lc_transform2;
            Eigen::Matrix4d delta_T_12_loop = T1.inverse() * T2;
            // std::cout<<"delta_T_12_loop:\n"<<delta_T_12_loop<<std::endl;
            Eigen::Vector3d euler_angles = delta_T_12_loop.block<3,3>(0,0).eulerAngles(2,1,0)*180/M_PI;
            for(int i=0; i<3; i++)
            {
                if(euler_angles[i]>90) euler_angles[i] = 180-euler_angles[i];
                if(euler_angles[i]<-90) euler_angles[i] = 180+euler_angles[i];
            }
            // std::cout<<"delta_T_12_loop.eulerAngles(2,1,0): "<<euler_angles.transpose()<<std::endl;  // yaw pitch roll
            Eigen::Matrix<double,6,1> delta_loop_r;
            delta_loop_r.block<3,1>(0,0) = euler_angles;
            delta_loop_r.block<3,1>(3,0) = delta_T_12_loop.block<3,1>(0,3);
            std::cout<<"delta_r: "<<delta_loop_r.transpose()<<std::endl;  // yaw pitch roll
        }
        return lc_transform2;
    }

	bool CheckIfJustPlane(const pcl::PointCloud<PointType>::Ptr& cloud_in, const float &thr)
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
		float ratio = float(inliners.indices.size())/float(cloud_in->size()) ;
		debug_file <<  "plane_ratio: " << ratio << std::endl;
		if (ratio > thr) return true;
		return false;
	}

	void CutVoxel3d(std::unordered_map<VOXEL_LOC, int> &feat_map, const pcl::PointCloud<PointType>::Ptr pl_feat, float voxel_box_size)
	{
		uint plsize = pl_feat->size();
		for(uint i=0; i<plsize; i++)
		{
			// Transform point to world coordinate
			PointType &p_c = pl_feat->points[i];
			Eigen::Vector3d pvec_tran(p_c.x, p_c.y, p_c.z);
			// Determine the key of hash table
			float loc_xyz[3];
			for(int j=0; j<3; j++)
			{
				loc_xyz[j] = pvec_tran[j] / voxel_box_size;
				if(loc_xyz[j] < 0)
				{
					loc_xyz[j] -= 1.0;
				}
			}
			VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
			auto iter = feat_map.find(position);
			if(iter == feat_map.end()) feat_map[position] = 0;
		}
	}

	void init_voxel_map(const pcl::PointCloud<PointType> &input_cloud, std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map, pcl::PointCloud<PointType>::Ptr &plane_cloud)
	{
		uint plsize = input_cloud.size();
		for (uint i = 0; i < plsize; i++) {
			Eigen::Vector3d p_c(input_cloud[i].x, input_cloud[i].y, input_cloud[i].z);
			double loc_xyz[3];
			for (int j = 0; j < 3; j++) {
				loc_xyz[j] = p_c[j] / plane_voxel_size; // 2.0
				if (loc_xyz[j] < 0) {
					loc_xyz[j] -= 1.0;
				}
			}
			VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
			auto iter = voxel_map.find(position);
			if (iter != voxel_map.end()) {
				voxel_map[position]->voxel_points_.push_back(p_c);
			} else {
				OctoTree *octo_tree = new OctoTree(plane_detection_thre, voxel_init_num);
				voxel_map[position] = octo_tree;
				voxel_map[position]->voxel_points_.push_back(p_c);
			}
		}
		for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
		{
			iter->second->init_octo_tree();
			if (iter->second->plane_ptr_->is_plane_)
			{
				PointType pi;
				pi.x = iter->second->plane_ptr_->center_[0];
				pi.y = iter->second->plane_ptr_->center_[1];
				pi.z = iter->second->plane_ptr_->center_[2];
				pi.normal_x = iter->second->plane_ptr_->normal_[0];
				pi.normal_y = iter->second->plane_ptr_->normal_[1];
				pi.normal_z = iter->second->plane_ptr_->normal_[2];
				plane_cloud->push_back(pi);
			}
		}
	}

	// 根据法线相似性和中心到平面的距离判断是否属于同一平面，然后进行平面融合
	void get_project_plane(std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map, std::vector<Plane *> &project_plane_list)
	{
		std::vector<Plane *> origin_list;
		for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
		{
			if(iter->second->plane_ptr_->is_plane_)
				origin_list.push_back(iter->second->plane_ptr_);
		}
		for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
		int current_id = 1;
		for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--)
		{
			for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++)
			{
				Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
				Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
				double dis1 = fabs((*iter)->normal_(0) * (*iter2)->center_(0) + (*iter)->normal_(1) * (*iter2)->center_(1) + (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
				double dis2 = fabs((*iter2)->normal_(0) * (*iter)->center_(0) + (*iter2)->normal_(1) * (*iter)->center_(1) + (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
				if (normal_diff.norm() < plane_merge_normal_thre || normal_add.norm() < plane_merge_normal_thre) // 0.1
					if (dis1 < plane_merge_dis_thre && dis2 < plane_merge_dis_thre)
					{ // 0.3
						if ((*iter)->id_ == 0 && (*iter2)->id_ == 0)
						{
							(*iter)->id_ = current_id;
							(*iter2)->id_ = current_id;
							current_id++;
						} 
						else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
							(*iter)->id_ = (*iter2)->id_;
						else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
							(*iter2)->id_ = (*iter)->id_;
					}
			}
		}
		std::vector<Plane *> merge_list;
		std::vector<int> merge_flag;
		for (size_t i = 0; i < origin_list.size(); i++)
		{
			auto it = std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
			if (it != merge_flag.end()) continue;
			if (origin_list[i]->id_ == 0) continue;
			Plane *merge_plane = new Plane;
			(*merge_plane) = (*origin_list[i]);
			bool is_merge = false;
			for (size_t j = 0; j < origin_list.size(); j++)
			{
				if (i == j)
					continue;
				if (origin_list[j]->id_ == origin_list[i]->id_) {
					is_merge = true;
					Eigen::Matrix3d P_PT1 = (merge_plane->covariance_ + merge_plane->center_ * merge_plane->center_.transpose()) * merge_plane->points_size_;
					Eigen::Matrix3d P_PT2 = (origin_list[j]->covariance_ + origin_list[j]->center_ * origin_list[j]->center_.transpose()) * origin_list[j]->points_size_;
					Eigen::Vector3d merge_center = (merge_plane->center_ * merge_plane->points_size_ + origin_list[j]->center_ * origin_list[j]->points_size_) / (merge_plane->points_size_ + origin_list[j]->points_size_);
					Eigen::Matrix3d merge_covariance = (P_PT1 + P_PT2) / (merge_plane->points_size_ + origin_list[j]->points_size_) - merge_center * merge_center.transpose();
					merge_plane->covariance_ = merge_covariance;
					merge_plane->center_ = merge_center;
					merge_plane->points_size_ = merge_plane->points_size_ + origin_list[j]->points_size_;
					merge_plane->sub_plane_num_++;
					// for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
					//   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
					// }
					Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
					Eigen::Matrix3cd evecs = es.eigenvectors();
					Eigen::Vector3cd evals = es.eigenvalues();
					Eigen::Vector3d evalsReal;
					evalsReal = evals.real();
					Eigen::Matrix3f::Index evalsMin, evalsMax;
					evalsReal.rowwise().sum().minCoeff(&evalsMin);
					evalsReal.rowwise().sum().maxCoeff(&evalsMax);
					Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
					merge_plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
					merge_plane->radius_ = sqrt(evalsReal(evalsMax));
					merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) + merge_plane->normal_(1) * merge_plane->center_(1) + merge_plane->normal_(2) * merge_plane->center_(2));
					merge_plane->p_center_.x = merge_plane->center_(0);
					merge_plane->p_center_.y = merge_plane->center_(1);
					merge_plane->p_center_.z = merge_plane->center_(2);
					merge_plane->p_center_.normal_x = merge_plane->normal_(0);
					merge_plane->p_center_.normal_y = merge_plane->normal_(1);
					merge_plane->p_center_.normal_z = merge_plane->normal_(2);
				}
			}
			if (is_merge) {
				merge_flag.push_back(merge_plane->id_);
				merge_list.push_back(merge_plane);
			}
		}
		project_plane_list = merge_list;
	}

	void merge_plane(std::vector<Plane *> &origin_list, std::vector<Plane *> &merge_plane_list)
	{
		if (origin_list.size() == 1)
		{
			merge_plane_list = origin_list;
			return;
		}
		for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
		int current_id = 1;
		for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
			for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
				Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
				Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
				double dis1 = fabs((*iter)->normal_(0) * (*iter2)->center_(0) + (*iter)->normal_(1) * (*iter2)->center_(1) + (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
				double dis2 = fabs((*iter2)->normal_(0) * (*iter)->center_(0) + (*iter2)->normal_(1) * (*iter)->center_(1) + (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
				if (normal_diff.norm() < plane_merge_normal_thre || normal_add.norm() < plane_merge_normal_thre)
					if (dis1 < plane_merge_dis_thre && dis2 < plane_merge_dis_thre) {
						if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
							(*iter)->id_ = current_id;
							(*iter2)->id_ = current_id;
							current_id++;
						} else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
							(*iter)->id_ = (*iter2)->id_;
						else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
							(*iter2)->id_ = (*iter)->id_;
					}
			}
		}
		std::vector<int> merge_flag;
		for (size_t i = 0; i < origin_list.size(); i++) {
			auto it = std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
			if (it != merge_flag.end())
				continue;
			if (origin_list[i]->id_ == 0) {
				merge_plane_list.push_back(origin_list[i]);
				continue;
			}
			Plane *merge_plane = new Plane;
			(*merge_plane) = (*origin_list[i]);
			bool is_merge = false;
			for (size_t j = 0; j < origin_list.size(); j++) {
				if (i == j)
					continue;
				if (origin_list[j]->id_ == origin_list[i]->id_) {
					is_merge = true;
					Eigen::Matrix3d P_PT1 = (merge_plane->covariance_ + merge_plane->center_ * merge_plane->center_.transpose()) * merge_plane->points_size_;
					Eigen::Matrix3d P_PT2 = (origin_list[j]->covariance_ + origin_list[j]->center_ * origin_list[j]->center_.transpose()) * origin_list[j]->points_size_;
					Eigen::Vector3d merge_center = (merge_plane->center_ * merge_plane->points_size_ + origin_list[j]->center_ * origin_list[j]->points_size_) / (merge_plane->points_size_ + origin_list[j]->points_size_);
					Eigen::Matrix3d merge_covariance = (P_PT1 + P_PT2) / (merge_plane->points_size_ + origin_list[j]->points_size_) - merge_center * merge_center.transpose();
					merge_plane->covariance_ = merge_covariance;
					merge_plane->center_ = merge_center;
					merge_plane->points_size_ = merge_plane->points_size_ + origin_list[j]->points_size_;
					merge_plane->sub_plane_num_ += origin_list[j]->sub_plane_num_;
					Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
					Eigen::Matrix3cd evecs = es.eigenvectors();
					Eigen::Vector3cd evals = es.eigenvalues();
					Eigen::Vector3d evalsReal;
					evalsReal = evals.real();
					Eigen::Matrix3f::Index evalsMin, evalsMax;
					evalsReal.rowwise().sum().minCoeff(&evalsMin);
					evalsReal.rowwise().sum().maxCoeff(&evalsMax);
					Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
					merge_plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
					merge_plane->radius_ = sqrt(evalsReal(evalsMax));
					merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) + merge_plane->normal_(1) * merge_plane->center_(1) + merge_plane->normal_(2) * merge_plane->center_(2));
					merge_plane->p_center_.x = merge_plane->center_(0);
					merge_plane->p_center_.y = merge_plane->center_(1);
					merge_plane->p_center_.z = merge_plane->center_(2);
					merge_plane->p_center_.normal_x = merge_plane->normal_(0);
					merge_plane->p_center_.normal_y = merge_plane->normal_(1);
					merge_plane->p_center_.normal_z = merge_plane->normal_(2);
				}
			}
			if (is_merge) {
				merge_flag.push_back(merge_plane->id_);
				merge_plane_list.push_back(merge_plane);
			}
		}
	}

	void binary_extractor(const std::vector<Plane *> proj_plane_list, const pcl::PointCloud<PointType>::Ptr &input_cloud, std::vector<BinaryDescriptor> &binary_descriptor_list)
	{
		binary_descriptor_list.clear();
		std::vector<BinaryDescriptor> temp_binary_list;
		Eigen::Vector3d last_normal(0, 0, 0);
		int useful_proj_num = 0;
		for (int i = 0; i < proj_plane_list.size(); i++)
		{
			std::vector<BinaryDescriptor> prepare_binary_list;
			Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
			Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
			if ((proj_normal - last_normal).norm() < 0.3 || (proj_normal + last_normal).norm() > 0.3)
			{
				last_normal = proj_normal;
				// std::cout << "proj normal:" << proj_normal.transpose() << std::endl;
				useful_proj_num++;
				extract_binary(proj_center, proj_normal, input_cloud, prepare_binary_list);
				for (auto bi : prepare_binary_list)
					temp_binary_list.push_back(bi);
				if (useful_proj_num == proj_plane_num) // 1
					break;
			}
		}
		// 搜索半径3米内的点，再次进行最大值抑制
		non_max_suppression(temp_binary_list);
		// 排序，保留值最大的，最多保留30个关键点
		if (useful_corner_num > temp_binary_list.size()) // 30
			binary_descriptor_list = temp_binary_list;
		else 
		{
			std::sort(temp_binary_list.begin(), temp_binary_list.end(), [](BinaryDescriptor a, BinaryDescriptor b){ return (a.summary_ > b.summary_); }); // 排序，保留值最大的
			for (size_t i = 0; i < useful_corner_num; i++)
				binary_descriptor_list.push_back(temp_binary_list[i]);
		}
	}

	void extract_binary(const Eigen::Vector3d &project_center, const Eigen::Vector3d &project_normal, const pcl::PointCloud<PointType>::Ptr &input_cloud, std::vector<BinaryDescriptor> &binary_list)
	{
		binary_list.clear();
		double A = project_normal[0];
		double B = project_normal[1];
		double C = project_normal[2];
		double D = -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
		std::vector<Eigen::Vector3d> projection_points;
		Eigen::Vector3d x_axis(1, 1, 0); // 构造法线的垂直向量
		if (C != 0) {
			x_axis[2] = -(A + B) / C;
		} else if (B != 0) {
			x_axis[1] = -A / B;
		} else {
			x_axis[0] = 0;
			x_axis[1] = 1;
		}
		x_axis.normalize();
		Eigen::Vector3d y_axis = project_normal.cross(x_axis);
		y_axis.normalize();
		double ax = x_axis[0];
		double bx = x_axis[1];
		double cx = x_axis[2];
		double dx = -(ax * project_center[0] + bx * project_center[1] + cx * project_center[2]);
		double ay = y_axis[0];
		double by = y_axis[1];
		double cy = y_axis[2];
		double dy = -(ay * project_center[0] + by * project_center[1] + cy * project_center[2]);
		std::vector<Eigen::Vector2d> point_list_2d;
		pcl::PointCloud<pcl::PointXYZ> point_list_3d;
		std::vector<double> dis_list_2d;
		for (size_t i = 0; i < input_cloud->size(); i++) {
			double x = input_cloud->points[i].x;
			double y = input_cloud->points[i].y;
			double z = input_cloud->points[i].z;
			double dis = fabs(x * A + y * B + z * C + D);
			pcl::PointXYZ pi;
			if (dis < proj_dis_min || dis > proj_dis_max) { // 0.2 5.0
				continue;
			} else {
				if (dis > proj_dis_min && dis <= proj_dis_max) {
					pi.x = x;
					pi.y = y;
					pi.z = z;
				}
			}
			Eigen::Vector3d cur_project; // pi 在平面上的投影点
			cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) / (A * A + B * B + C * C);
			cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) / (A * A + B * B + C * C);
			cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) / (A * A + B * B + C * C);
			pcl::PointXYZ p;
			p.x = cur_project[0];
			p.y = cur_project[1];
			p.z = cur_project[2];
			double project_x = cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
			double project_y = cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
			Eigen::Vector2d p_2d(project_x, project_y);
			point_list_2d.push_back(p_2d);
			dis_list_2d.push_back(dis);
			point_list_3d.points.push_back(pi);
		}
		double min_x = 10;
		double max_x = -10;
		double min_y = 10;
		double max_y = -10;
		if (point_list_2d.size() <= 5) {
			return;
		}
		for (auto pi : point_list_2d) {
			if (pi[0] < min_x) {
				min_x = pi[0];
			}
			if (pi[0] > max_x) {
				max_x = pi[0];
			}
			if (pi[1] < min_y) {
				min_y = pi[1];
			}
			if (pi[1] > max_y) {
				max_y = pi[1];
			}
		}
		// segment project cloud
		int segmen_base_num = 5;
		double segmen_len = segmen_base_num * proj_image_resolution; // 0.5
		int x_segment_num = (max_x - min_x) / segmen_len + 1;
		int y_segment_num = (max_y - min_y) / segmen_len + 1;
		int x_axis_len = (int)((max_x - min_x) / proj_image_resolution + segmen_base_num);
		int y_axis_len = (int)((max_y - min_y) / proj_image_resolution + segmen_base_num);
		std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
		BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
		for (int i = 0; i < x_axis_len; i++) {
			dis_container[i] = new std::vector<double>[y_axis_len];
			binary_container[i] = new BinaryDescriptor[y_axis_len];
		}
		double **img_count = new double *[x_axis_len];
		for (int i = 0; i < x_axis_len; i++) {
			img_count[i] = new double[y_axis_len];
		}
		double **dis_array = new double *[x_axis_len];
		for (int i = 0; i < x_axis_len; i++) {
			dis_array[i] = new double[y_axis_len];
		}
		double **mean_x_list = new double *[x_axis_len];
		for (int i = 0; i < x_axis_len; i++) {
			mean_x_list[i] = new double[y_axis_len];
		}
		double **mean_y_list = new double *[x_axis_len];
		for (int i = 0; i < x_axis_len; i++) {
			mean_y_list[i] = new double[y_axis_len];
		}
		for (int x = 0; x < x_axis_len; x++) {
			for (int y = 0; y < y_axis_len; y++) {
				img_count[x][y] = 0;
				mean_x_list[x][y] = 0;
				mean_y_list[x][y] = 0;
				dis_array[x][y] = 0;
				std::vector<double> single_dis_container;
				dis_container[x][y] = single_dis_container;
			}
		}

		for (size_t i = 0; i < point_list_2d.size(); i++) {
			int x_index = (int)((point_list_2d[i][0] - min_x) / proj_image_resolution);
			int y_index = (int)((point_list_2d[i][1] - min_y) / proj_image_resolution);
			mean_x_list[x_index][y_index] += point_list_2d[i][0];
			mean_y_list[x_index][y_index] += point_list_2d[i][1];
			img_count[x_index][y_index]++;
			dis_container[x_index][y_index].push_back(dis_list_2d[i]);
		}

		for (int x = 0; x < x_axis_len; x++) {
			for (int y = 0; y < y_axis_len; y++) {
				// calc segment dis array
				// 将点到平面的距离进行分段，记录在哪些分段上存在点
				if (img_count[x][y] > 0) {
					int cut_num = (proj_dis_max - proj_dis_min) / proj_image_high_inc; // 5.0 0.2 0.1
					std::vector<bool> occup_list;
					std::vector<double> cnt_list;
					BinaryDescriptor single_binary;
					for (size_t i = 0; i < cut_num; i++) {
						cnt_list.push_back(0);
						occup_list.push_back(false);
					}
					for (size_t j = 0; j < dis_container[x][y].size(); j++) {
						int cnt_index = (dis_container[x][y][j] - proj_dis_min) / proj_image_high_inc;
						cnt_list[cnt_index]++;
					}
					double segmnt_dis = 0;
					for (size_t i = 0; i < cut_num; i++) {
						if (cnt_list[i] >= 1) {
							segmnt_dis++;
							occup_list[i] = true;
						}
					}
					dis_array[x][y] = segmnt_dis; // 存在点的分段的数量
					single_binary.occupy_array_ = occup_list; // 记录在哪些分段上存在点
					single_binary.summary_ = segmnt_dis; // 存在点的分段的数量
					binary_container[x][y] = single_binary;
				}
			}
		}

		// debug image
		// double max_dis_cnt = (proj_dis_max - proj_dis_min) / proj_image_high_inc;
		// cv::Mat proj_image = cv::Mat::zeros(y_axis_len, x_axis_len, CV_8UC1);
		// for (size_t y = 0; y < y_axis_len; y++) {
		//   for (size_t x = 0; x < x_axis_len; x++) {
		//     if (dis_array[x][y] != 0) {
		//       proj_image.at<uchar>(y, x) = dis_array[x][y] * 20 + 50;
		//     }
		//   }
		// }
		// cv::Mat image_max; // 等比例放大图
		// cv::resize(proj_image, image_max,
		//            cv::Size(y_axis_len * 2, x_axis_len * 2)); // 放大操作
		// cv::Mat out;
		// // cv::equalizeHist(proj_image, out);
		// cv::imshow("proj image", proj_image);
		// cv::waitKey();
		// filter by distance
		std::vector<double> max_dis_list;
		std::vector<int> max_dis_x_index_list;
		std::vector<int> max_dis_y_index_list;
		for (int x_segment_index = 0; x_segment_index < x_segment_num; x_segment_index++) {
			for (int y_segment_index = 0; y_segment_index < y_segment_num; y_segment_index++) {
				double max_dis = 0;
				int max_dis_x_index = -10;
				int max_dis_y_index = -10;
				// 寻找5*5小块内的最大值
				for (int x_index = x_segment_index * segmen_base_num; x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
					for (int y_index = y_segment_index * segmen_base_num; y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
						if (dis_array[x_index][y_index] > max_dis) {
							max_dis = dis_array[x_index][y_index];
							max_dis_x_index = x_index;
							max_dis_y_index = y_index;
						}
					}
				}
				if (max_dis >= summary_min_thre) // 8 关键点处存在点的分段的数量至少为8
				{
					bool is_touch = true;
					max_dis_list.push_back(max_dis);
					max_dis_x_index_list.push_back(max_dis_x_index);
					max_dis_y_index_list.push_back(max_dis_y_index);
				}
			}
		}
		// calc line or not
		std::vector<Eigen::Vector2i> direction_list;
		Eigen::Vector2i d(0, 1);
		direction_list.push_back(d);
		d << 1, 0;
		direction_list.push_back(d);
		d << 1, 1;
		direction_list.push_back(d);
		d << 1, -1;
		direction_list.push_back(d);
		for (size_t i = 0; i < max_dis_list.size(); i++) {
			Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
			if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 || p[1] >= y_axis_len - 1) {
				continue;
			}
			bool is_add = true;
			// 如果当前候选关键点周围存在像素点的值接近当前值，则放弃当前候选关键点
			for (int j = 0; j < 4; j++) {
				Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
				if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 || p[1] >= y_axis_len - 1) {
					continue;
				}
				Eigen::Vector2i p1 = p + direction_list[j];
				Eigen::Vector2i p2 = p - direction_list[j];
				double threshold = dis_array[p[0]][p[1]] - 3;
				if (dis_array[p1[0]][p1[1]] >= threshold) {
					if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
						is_add = false;
					}
				}
				if (dis_array[p2[0]][p2[1]] >= threshold) {
					if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
						is_add = false;
					}
				}
				if (dis_array[p1[0]][p1[1]] >= threshold) {
					if (dis_array[p2[0]][p2[1]] >= threshold) {
						is_add = false;
					}
				}
				if (dis_array[p2[0]][p2[1]] >= threshold) {
					if (dis_array[p1[0]][p1[1]] >= threshold) {
						is_add = false;
					}
				}
			}
			if (is_add) {
				// 计算关键像素点对应的三维点的x方向和y方向的均值
				double px = mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] / img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
				double py = mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] / img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
				// 根据均值对中心点进行偏移得到位置
				Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
				pcl::PointXYZ pi;
				pi.x = coord[0];
				pi.y = coord[1];
				pi.z = coord[2];
				BinaryDescriptor single_binary = binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
				single_binary.location_ = coord; // 在哪些分段上存在点 存在点的分段的数量 这些点在平面上的重心
				binary_list.push_back(single_binary); // 将0.5*0.5*(5-0.2)的垂直于平面的立柱沿着法线划分为多段，记录在哪些分段上存在点 occupy_array_、存在点的分段的数量 summary_、这些点在平面上的重心 location_
			}
		}
		for (int i = 0; i < x_axis_len; i++) {
			delete[] binary_container[i];
			delete[] dis_container[i];
			delete[] img_count[i];
			delete[] dis_array[i];
			delete[] mean_x_list[i];
			delete[] mean_y_list[i];
		}
		delete[] binary_container;
		delete[] dis_container;
		delete[] img_count;
		delete[] dis_array;
		delete[] mean_x_list;
		delete[] mean_y_list;
	}

	void non_max_suppression(std::vector<BinaryDescriptor> &binary_list)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
		std::vector<int> pre_count_list;
		std::vector<bool> is_add_list;
		for (auto var : binary_list) {
			pcl::PointXYZ pi;
			pi.x = var.location_[0];
			pi.y = var.location_[1];
			pi.z = var.location_[2];
			prepare_key_cloud->push_back(pi);
			pre_count_list.push_back(var.summary_);
			is_add_list.push_back(true);
		}
		kd_tree.setInputCloud(prepare_key_cloud);
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
			pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
			if (kd_tree.radiusSearch(searchPoint, non_max_suppression_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
				Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
				for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
					Eigen::Vector3d pj(prepare_key_cloud->points[pointIdxRadiusSearch[j]].x, prepare_key_cloud->points[pointIdxRadiusSearch[j]].y, prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
					if (pointIdxRadiusSearch[j] == i) {
						continue;
					}
					if (pre_count_list[i] <= pre_count_list[pointIdxRadiusSearch[j]]) {
						is_add_list[i] = false;
					}
				}
			}
		}
		std::vector<BinaryDescriptor> pass_binary_list;
		for (size_t i = 0; i < is_add_list.size(); i++) {
			if (is_add_list[i]) {
				pass_binary_list.push_back(binary_list[i]);
			}
		}
		binary_list.clear();
		for (auto var : pass_binary_list) {
			binary_list.push_back(var);
		}
		return;
	}

	void generate_std(const std::vector<BinaryDescriptor> &binary_list, std::vector<STD> &std_list)
	{
		double triangle_scale_ = 1.0 / std_side_resolution; // 0.2 ==》 5
		std::unordered_map<VOXEL_LOC, bool> feat_map;
		pcl::PointCloud<pcl::PointXYZ> key_cloud;
		for (auto var : binary_list) {
			pcl::PointXYZ pi;
			pi.x = var.location_[0];
			pi.y = var.location_[1];
			pi.z = var.location_[2];
			key_cloud.push_back(pi);
		}
		pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
		kd_tree->setInputCloud(key_cloud.makeShared());
		int K = descriptor_near_num; // 20
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		for (size_t i = 0; i < key_cloud.size(); i++)
		{
			pcl::PointXYZ searchPoint = key_cloud.points[i];
			if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
			{
				for (int m = 1; m < K - 1; m++)
				{
					for (int n = m + 1; n < K; n++)
					{
						pcl::PointXYZ p1 = searchPoint;
						pcl::PointXYZ p2 = key_cloud.points[pointIdxNKNSearch[m]];
						pcl::PointXYZ p3 = key_cloud.points[pointIdxNKNSearch[n]];
						double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
						double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p3.z, 2));
						double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) + pow(p3.z - p2.z, 2));
						// 边长范围 [3, 50]
						if (a > descriptor_max_len || b > descriptor_max_len || c > descriptor_max_len || a < descriptor_min_len || b < descriptor_min_len || c < descriptor_min_len)
							continue;
						double temp;
						Eigen::Vector3d A, B, C;
						Eigen::Vector3i l1, l2, l3; // 每条边对应的顶点
						Eigen::Vector3i l_temp;
						l1 << 1, 2, 0;
						l2 << 1, 0, 3;
						l3 << 0, 2, 3;
						if (a > b) {
							temp = a;
							a = b;
							b = temp;
							l_temp = l1;
							l1 = l2;
							l2 = l_temp;
						}
						if (b > c) {
							temp = b;
							b = c;
							c = temp;
							l_temp = l2;
							l2 = l3;
							l3 = l_temp;
						}
						if (a > b) {
							temp = a;
							a = b;
							b = temp;
							l_temp = l1;
							l1 = l2;
							l2 = l_temp;
						}
						if (fabs(c - (a + b)) < 0.2) { // 最小的两边之和和最大边不能相差太小
							continue;
						}
						pcl::PointXYZ d_p;
						d_p.x = a * 1000;
						d_p.y = b * 1000;
						d_p.z = c * 1000;
						VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
						auto iter = feat_map.find(position);
						Eigen::Vector3d normal_1, normal_2, normal_3;
						BinaryDescriptor binary_A;
						BinaryDescriptor binary_B;
						BinaryDescriptor binary_C;
						if (iter == feat_map.end()) {
							// 最短的两边大共有的顶点
							if (l1[0] == l2[0]) { // 只能是 1, 2, 0; 和 1, 0, 3  最短的两边大共有的顶点
								A << p1.x, p1.y, p1.z;
								binary_A = binary_list[i];
							} else if (l1[1] == l2[1]) {
								A << p2.x, p2.y, p2.z;
								binary_A = binary_list[pointIdxNKNSearch[m]];
							} else {
								A << p3.x, p3.y, p3.z;
								binary_A = binary_list[pointIdxNKNSearch[n]];
							}
							// 最短边和最长边的共有顶点
							if (l1[0] == l3[0]) {
								B << p1.x, p1.y, p1.z;
								binary_B = binary_list[i];
							} else if (l1[1] == l3[1]) {
								B << p2.x, p2.y, p2.z;
								binary_B = binary_list[pointIdxNKNSearch[m]];
							} else {
								B << p3.x, p3.y, p3.z;
								binary_B = binary_list[pointIdxNKNSearch[n]];
							}
							// 中间边和最长边的共有顶点
							if (l2[0] == l3[0]) {
								C << p1.x, p1.y, p1.z;
								binary_C = binary_list[i];
							} else if (l2[1] == l3[1]) {
								C << p2.x, p2.y, p2.z;
								binary_C = binary_list[pointIdxNKNSearch[m]];
							} else {
								C << p3.x, p3.y, p3.z;
								binary_C = binary_list[pointIdxNKNSearch[n]];
							}
							STD single_descriptor;
							single_descriptor.binary_A_ = binary_A; // 最短的两边大共有的顶点的二值描述子
							single_descriptor.binary_B_ = binary_B; // 最短边和最长边的共有顶点的二值描述子
							single_descriptor.binary_C_ = binary_C; // 中间边和最长边的共有顶点的二值描述子
							single_descriptor.center_ = (A + B + C) / 3; // 三角形的中心
							single_descriptor.triangle_ << triangle_scale_ * a, triangle_scale_ * b, triangle_scale_ * c; // 5 三边长度*5，由小到大
							single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2)); // 没有值？？？？
							single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
							single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
							// single_descriptor.angle << 0, 0, 0;
							single_descriptor.frame_number_ = sub_map_id;
							Eigen::Matrix3d triangle_positon;
							triangle_positon.block<3, 1>(0, 0) = A;
							triangle_positon.block<3, 1>(0, 1) = B;
							triangle_positon.block<3, 1>(0, 2) = C;
							feat_map[position] = true;
							std_list.push_back(single_descriptor);
						}
					}
				}
			}
		}
	}

	float calculateOverlapScore(const pcl::PointCloud<PointType>::Ptr &currKeyframeCloud, const pcl::PointCloud<PointType>::Ptr &targetKeyframeCloud, const Eigen::Matrix4f &lc_transform, const float &voxel_box_size, const int &num_pt_max)
	{
		debug_file << "targetKeyframeCloud size " << targetKeyframeCloud->size() << std::endl;
		debug_file << "currKeyframeCloud size " << currKeyframeCloud->size() << std::endl;
		pcl::PointCloud<PointType>::Ptr currKeyframeCloud_ds(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr targetKeyframeCloud_ds(new pcl::PointCloud<PointType>());
		down_sampling_voxel(*currKeyframeCloud, *currKeyframeCloud_ds, 0.5 * voxel_box_size, true); 
		down_sampling_voxel(*targetKeyframeCloud, *targetKeyframeCloud_ds, 0.5 * voxel_box_size, true); 
		// pcl::VoxelGrid<PointType> sor;
		// sor.setInputCloud(currKeyframeCloud);
		// sor.setLeafSize(0.5 * voxel_box_size, 0.5 * voxel_box_size, 0.5 * voxel_box_size); // 2.0
		// sor.filter(*currKeyframeCloud_ds);
		// sor.setInputCloud(targetKeyframeCloud);
		// sor.setLeafSize(0.5 * voxel_box_size, 0.5 * voxel_box_size, 0.5 * voxel_box_size);
		// sor.filter(*targetKeyframeCloud_ds);
		int num_pt_cur = int(currKeyframeCloud_ds->size());
		std::vector<int> indices;
		int sample_gap;
		if(num_pt_cur > 2 * num_pt_max) sample_gap = ceil(double(num_pt_cur) / double(num_pt_max)); // 150000
		else sample_gap = 1;
		for(int i = 0; i < num_pt_cur; i += sample_gap) indices.push_back(i); // downsample
		int num_pt_target = int(targetKeyframeCloud_ds->size());
		float cloud_size_ratio = std::min(float(num_pt_cur) / float(num_pt_target), float(num_pt_target) / float(num_pt_cur));
		debug_file << "[FPR]: submap pair's pt numbers ratio: " << cloud_size_ratio << std::endl;
		pcl::PointCloud<PointType>::Ptr cloud_in_transed(new pcl::PointCloud<PointType>);
		pcl::transformPointCloud(*currKeyframeCloud_ds, indices, *cloud_in_transed, lc_transform, false); // 将当前帧的点转换到之前的帧对应的世界坐标系
		std::unordered_map<VOXEL_LOC, int> uomp_3d;
		CutVoxel3d(uomp_3d, targetKeyframeCloud_ds, voxel_box_size); // cut voxel for counting hit 2.0  构建一个目标点云所在位置的空的网格
		debug_file<<"CutVoxel3d, cloud_in_transed->size(): "<<cloud_in_transed->size()<<std::endl;
		int count1 = 0; // 占用的网格数量
		int count2 = 0; // 在网格中的点数
		for(int i = 0; i < cloud_in_transed->size(); i++)
		{
			const PointType &a_pt = cloud_in_transed->points[i];
			Eigen::Vector3f pt(a_pt.x, a_pt.y, a_pt.z);
			float loc_xyz[3];
			for(int j = 0; j < 3; j++)
			{
				loc_xyz[j] = pt[j] / voxel_box_size;
				if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
			}
			VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
			auto iter = uomp_3d.find(position);
			if(iter != uomp_3d.end())
			{
				if (iter->second == 0) count1++;
				iter->second++;
				count2++;
			}
		}
		float score = (float(count2) / float(cloud_in_transed->size())) * cloud_size_ratio; // 先对目标点云进行2m*2m的网格划分，然后再看当前点云落在这个网格中的点比例即为 score
		if(score > overlap_score_thr) // 0.5
		{
			debug_file<<"CheckIfJustPlane, score: "<<score<<std::endl;
			bool is_plane = CheckIfJustPlane(cloud_in_transed, plane_inliner_ratio_thr); // 0.4 如果平面点的占比超过 0.4 
			if(is_plane)
			{
				debug_file << "[FPR]: reject, the target cloud is like a plane." << std::endl;
				return 0.0f;
			}
			debug_file<<"CheckIfJustPlane, is_plane: "<<is_plane<<std::endl;
		}
		return score;
	}

	bool detect_loopclosure(std::vector<STD> &STD_list, pcl::PointCloud<PointType>::Ptr &frame_plane_cloud, std::unordered_map<STD_LOC, std::vector<STD>> &STD_map, std::vector<pcl::PointCloud<PointType>::Ptr> &history_plane_list)
	{
		// std::cout<<"============detect_loopclosure============"<<std::endl;
		// std::cout<<"STD_list.size(): "<<STD_list.size()<<std::endl;
		// std::cout<<"frame_plane_cloud->size(): "<<frame_plane_cloud->size()<<std::endl;
		// std::cout<<"STD_map.size(): "<<STD_map.size()<<std::endl;
		// std::cout<<"history_plane_list.size(): "<<history_plane_list.size()<<std::endl;
		// std::cout<<"sub_map_id: "<<sub_map_id<<std::endl;
		auto start2 = std::chrono::system_clock::now();
		auto t_candidate_search_begin = std::chrono::high_resolution_clock::now();
		new_added_lc_factors.clear();
		std::vector<STDMatchList> alternative_match;
		// 1. 通过边长相似性寻找可能得候选特征
		// 2. 候选特征满足帧id与当前id相差超过50，边长差异小于1%，二值描述子相似度超过0.7
		// 3. 逐步筛选匹配数量最高的帧，考虑匹配数量最高的5个帧，保存匹配帧id match_frame_ 和其中所有匹配的特征 match_list_ 于 alternative_match
		candidate_searcher(STD_map, STD_list, alternative_match);
		auto t_candidate_search_end = std::chrono::high_resolution_clock::now();
		// geometrical verification
		auto t_fine_loop_begin = std::chrono::high_resolution_clock::now();
		bool triggle_loop = false;
		Eigen::Vector3d best_t;
		Eigen::Matrix3d best_rot;
		Eigen::Vector3d loop_translation;
		Eigen::Matrix3d loop_rotation;
		std::vector<std::pair<STD, STD>> sucess_match_list;
		std::vector<std::pair<STD, STD>> unsucess_match_list;
		std::vector<std::pair<STD, STD>> sucess_match_list_publish;
		std::vector<std::pair<STD, STD>> unsucess_match_list_publish;
		int match_size = 0;
		int rough_size = 0;
		int candidate_id = -1;
		int match_frame_id = 0;
		double best_score = 0;
		double best_icp_score = 0;
		int best_frame = -1;
		// std::cout << "alternative_match.size(): "  << alternative_match.size() << std::endl;
		debug_file << "alternative_match.size(): "  << alternative_match.size() << std::endl;
		for (int i = 0; i < alternative_match.size(); i++)
		{
			if (alternative_match[i].match_list_.size() < 4) continue; // 匹配帧中至少4个匹配特征才考虑
			bool fine_sucess = false;
			Eigen::Matrix3d std_rot;
			Eigen::Vector3d std_t;
			// std::cout << "[Rough match] rough match frame:" << alternative_match[i].match_frame_ << " match size:" << alternative_match[i].match_list_.size() << std::endl;
			debug_file << "[Rough match] rough match frame:" << alternative_match[i].match_frame_ << " match size:" << alternative_match[i].match_list_.size() << std::endl;
			sucess_match_list.clear();
			// 采用RANSAC的思想，均匀采样50个匹配特征计算位姿变化，并记录特征通过位姿变换后基本能满足匹配的数量
			// 找出最能满足匹配的位姿
			// 至少4个匹配特征在位姿变化后差异满足条件则记录最优匹配，并计算位姿 std_rot, std_t，然后统计满足匹配的特征 sucess_match_list 和不满足的特征 unsucess_match_list
			fine_loop_detection_tbb(alternative_match[i].match_list_, fine_sucess, std_rot, std_t, sucess_match_list, unsucess_match_list);
			if (fine_sucess)
			{
				// 搜索当前点通过位姿变化后的点在历史数据中的最近点，如果最近点的法线差异小于0.1，点到面的距离差异小于0.5，则认为成功匹配，输出匹配成功的比例 icp_score
				double icp_score = geometric_verify(frame_plane_cloud, history_plane_list[alternative_match[i].match_frame_], std_rot, std_t);
				double score = icp_score + sucess_match_list.size() * 1.0 / 1000;
				std::cout << "Fine sucess, Fine size:" << sucess_match_list.size() << "  ,Icp score:" << icp_score << ", score:" << score << std::endl;
				debug_file << "Fine sucess, Fine size:" << sucess_match_list.size() << "  ,Icp score:" << icp_score << ", score:" << score << std::endl;
				if (score > best_score)
				{
					unsucess_match_list_publish = unsucess_match_list;
					sucess_match_list_publish = sucess_match_list;
					best_frame = alternative_match[i].match_frame_;
					best_score = score;
					best_icp_score = icp_score;
					best_rot = std_rot;
					best_t = std_t;
					rough_size = alternative_match[i].match_list_.size();
					match_size = sucess_match_list.size();
					candidate_id = i;
				}
			}
		}
		if (best_icp_score > icp_threshold) // 0.15
		{
			loop_translation = best_t;
			loop_rotation = best_rot;
			match_frame_id = best_frame;
			triggle_loop = true;
			std::cout << "loop translation:" << loop_translation.transpose() << std::endl;
			std::cout << "loop rotation:" << std::endl << loop_rotation << std::endl;
			Eigen::Matrix4d lc_transform = Eigen::Matrix4d::Identity();
			lc_transform.block<3, 3>(0, 0) = loop_rotation;
			lc_transform.block<3, 1>(0, 3) = loop_translation;
			double pose_drift_ = ((lc_transform*sub_maps[sub_map_id].sub_map_pose).block<3,1>(0,3) - (lc_delta_pose*sub_maps[sub_map_id].sub_map_pose).block<3,1>(0,3)).norm();
			debug_file<<"sub_map_id: "<<sub_map_id<<", pose_drift_: "<<pose_drift_<<std::endl;
			std::cout<<"sub_map_id: "<<sub_map_id<<", pose_drift_: "<<pose_drift_<<std::endl;
			// debug_file << "lc_transform:\n" << lc_transform << std::endl;
			// 先对像个点云进行下采样（1m），保留的点最大为150000， 然后对目标点云进行2m*2m的网格划分，然后再看当前点云通过回环位姿lc_transform变换之后落在这个网格中的点比例即为 score
			// 此外，还要检查 如果平面点的占比超过 0.4  则是当前帧为平面，score=0.0
			float overlap_score = calculateOverlapScore(sub_maps[sub_map_id].sub_map_cloud, sub_maps[match_frame_id].sub_map_cloud, lc_transform.cast<float>(), vs_for_ovlap, 150000); // 2.0
			debug_file << "[FPR]: overlap ratio is " << overlap_score << std::endl;
			std::cout << "[FPR]: overlap ratio is " << overlap_score << std::endl;
			if (overlap_score < overlap_score_thr) // check LC cloud pair overlap ratio  0.5 网格重合比例小于 50%
			{
				debug_file << "[FPR]: reject, too small overlap." << std::endl;
				std::cout << "[FPR]: reject, too small overlap." << std::endl;
				triggle_loop = false;
			}
		}
		auto t_fine_loop_end = std::chrono::high_resolution_clock::now();
		auto end2 = std::chrono::system_clock::now();
		std::chrono::duration<double, std::milli> elapsed_ms2 = std::chrono::duration<double, std::milli>(end2 - start2);
		time_lcd_file << elapsed_ms2.count() << " " << (triggle_loop ? 2 : -2) << std::endl;
		if (triggle_loop)
		{
			Eigen::Matrix4d lc_transform = Eigen::Matrix4d::Identity();
			lc_transform.block<3, 3>(0, 0) = loop_rotation;
			lc_transform.block<3, 1>(0, 3) = loop_translation;
			lc_curr_idx = sub_map_id;
			lc_prev_idx = match_frame_id;
			Eigen::Matrix4d lc_transform_updated = lc_transform;
			float min_dist_aver=0.0;
			auto calcu_loop_poses_0 = std::chrono::system_clock::now();
			lc_transform_updated = calcu_loop_poses(sub_map_id, match_frame_id, min_dist_aver, lc_transform);
			auto calcu_loop_poses_1 = std::chrono::system_clock::now();
			auto calcu_loop_poses_ms = std::chrono::duration<double, std::milli>(calcu_loop_poses_1 - calcu_loop_poses_0);
			debug_file << "elapsed_ms aft calcu_loop_poses: " << calcu_loop_poses_ms.count() <<"ms" << std::endl;
			// lc_transform_updated = point_plane_optimize(sub_map_id, match_frame_id, lc_transform);
			if(min_dist_aver>0.2) // 当前子图与目标子图用于成功建立点面约束关系的近邻点中最近的距离的平均值,这个不太靠谱
			{
				debug_file << "[Loop Fail] " << sub_map_id << ", icp:" << best_score << std::endl;
				debug_file<<"[Loop Fail]: Too large min_dist_aver: "<<min_dist_aver<<std::endl;
				std::cout<<"[Loop Fail]: Too large min_dist_aver: "<<min_dist_aver<<std::endl;
				triggle_loop = false;
				double pose_drift_ = ((lc_transform_updated*sub_maps[sub_map_id].sub_map_pose).block<3,1>(0,3) - (lc_delta_pose*sub_maps[sub_map_id].sub_map_pose).block<3,1>(0,3)).norm();
				LCInfo lc_info;
				// lc_info.overlap_score0 = overlap_score;
				lc_info.pose_drift0 = pose_drift_;
				lc_info.lc_transform0 = lc_transform;
				lc_info.lc_transform = lc_transform_updated;
				lc_info.prev_idx = match_frame_id;
				lc_info.curr_idx = sub_map_id;
				rej_sml_ovl_lcs.push_back(lc_info);
				return false;
			}
			loop_rotation = lc_transform_updated.block<3, 3>(0, 0);
			loop_translation = lc_transform_updated.block<3, 1>(0, 3);
			LoopDetectionResult lc_result;
			lc_result.lc_transform = lc_transform_updated;
			lc_result.current_submap_id = sub_map_id;
			lc_result.target_submap_id = match_frame_id;
			lc_result.icp_score = best_icp_score;
			lc_results.push_back(lc_result);
			std::cout << "[Loop Sucess] " << sub_map_id << "--" << match_frame_id << ", candidate id:" << candidate_id << ", icp:" << best_score << std::endl;
			debug_file << "[Loop Sucess] " << sub_map_id << "--" << match_frame_id << ", candidate id:" << candidate_id << ", icp:" << best_score << std::endl;
			std::vector<std::tuple<STD,STD,float>> descriptor_pairs;
			for (auto var : sucess_match_list_publish)
			{
				auto A_tran = loop_rotation * var.first.binary_A_.location_ + loop_translation;
				auto B_tran = loop_rotation * var.first.binary_B_.location_ + loop_translation;
				auto C_tran = loop_rotation * var.first.binary_C_.location_ + loop_translation;
				float ABC_dis = (var.second.binary_A_.location_ - A_tran).norm() + (var.second.binary_B_.location_ - B_tran).norm() + (var.second.binary_C_.location_ - C_tran).norm();
				descriptor_pairs.push_back({var.first, var.second, ABC_dis});
			}
			std::sort(descriptor_pairs.begin(), descriptor_pairs.end(), [](std::tuple<STD,STD,float> a, std::tuple<STD,STD,float> b) { return (std::get<2>(a) < std::get<2>(b)); });
			if(associate_consecutive_frame) addLCDescriptorsFactors(descriptor_pairs);
			else loop_closure_detected = true;
			// 添加回环位姿约束
			gtsam::Pose3 pose_prev, pose_curr;
			pose_prev = trans2gtsamPose(sub_maps[lc_prev_idx].sub_map_pose);
			Eigen::Matrix4d lc_pose_curr = lc_transform_updated*sub_maps[lc_curr_idx].sub_map_pose;
			pose_curr = trans2gtsamPose(lc_pose_curr);
			gts_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lc_prev_idx, lc_curr_idx, pose_prev.between(pose_curr), robustLoopNoise));
			LCFactor lc_factor;
			lc_factor.pose_prev = pose_prev;
			lc_factor.pose_curr = pose_curr;
			lc_factor.lc_transform0 = lc_transform;
			lc_factor.lc_transform = lc_transform_updated;
			lc_factor.prev_idx = lc_prev_idx;
			lc_factor.curr_idx = lc_curr_idx;
			new_added_lc_factors.push_back(lc_factor);
			return true;
		}
		// std::cout << "[Loop Fail] " << sub_map_id << ", icp:" << best_score << std::endl;
		debug_file << "[Loop Fail] " << sub_map_id << ", icp:" << best_score << std::endl;
		return false;
	}

	void addLCDescriptorsFactors(const std::vector<std::tuple<STD,STD,float>> &descriptor_pairs)
	{
		if (descriptor_pairs.empty())
			return;
		std::vector<std::pair<PointType,PointType>> corners_pairs_;
		PointType pi, pj;
		int count = 0;
		int k = 6;
		for (auto &a_pair:descriptor_pairs)
		{
			count++;
			if (count > k) continue;
			pi.x = std::get<0>(a_pair).binary_A_.location_[0];
			pi.y = std::get<0>(a_pair).binary_A_.location_[1];
			pi.z = std::get<0>(a_pair).binary_A_.location_[2];
			pj.x = std::get<1>(a_pair).binary_A_.location_[0];
			pj.y = std::get<1>(a_pair).binary_A_.location_[1];
			pj.z = std::get<1>(a_pair).binary_A_.location_[2];
			corners_pairs_.push_back({pi, pj});
			pi.x = std::get<0>(a_pair).binary_B_.location_[0];
			pi.y = std::get<0>(a_pair).binary_B_.location_[1];
			pi.z = std::get<0>(a_pair).binary_B_.location_[2];
			pj.x = std::get<1>(a_pair).binary_B_.location_[0];
			pj.y = std::get<1>(a_pair).binary_B_.location_[1];
			pj.z = std::get<1>(a_pair).binary_B_.location_[2];
			corners_pairs_.push_back({pi, pj});
			pi.x = std::get<0>(a_pair).binary_C_.location_[0];
			pi.y = std::get<0>(a_pair).binary_C_.location_[1];
			pi.z = std::get<0>(a_pair).binary_C_.location_[2];
			pj.x = std::get<1>(a_pair).binary_C_.location_[0];
			pj.y = std::get<1>(a_pair).binary_C_.location_[1];
			pj.z = std::get<1>(a_pair).binary_C_.location_[2];
			corners_pairs_.push_back({pi, pj});
		}
		int curr_idx_lct = std::get<0>(descriptor_pairs[0]).frame_number_;
		int prev_idx_lct = std::get<1>(descriptor_pairs[0]).frame_number_;
		debug_file << "LC KPF queue front: " << prev_idx_lct << " " << curr_idx_lct;
		loop_closure_detected = true;
		insertKPPairConstraint(corners_pairs_, prev_idx_lct, curr_idx_lct, 2);
	}

	void candidate_searcher(std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map, std::vector<STD> &current_STD_list, std::vector<STDMatchList> &alternative_match)
	{
		int outlier = 0;
		double max_dis = 100;
		double match_array[20000] = {0};
		std::vector<std::pair<STD, STD>> match_list;
		std::vector<int> match_list_index;
		std::vector<Eigen::Vector3i> voxel_round;
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					Eigen::Vector3i voxel_inc(x, y, z);
					voxel_round.push_back(voxel_inc);
				}
			}
		}
		std::vector<bool> useful_match(current_STD_list.size());
		std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
		std::vector<std::vector<STD_LOC>> useful_match_position(current_STD_list.size());
		std::vector<size_t> index(current_STD_list.size());
		for (size_t i = 0; i < index.size(); ++i)
		{
			index[i] = i;
			useful_match[i] = false;
			STD descriptor = current_STD_list[i];
			STD_LOC position;
			int best_index = 0;
			STD_LOC best_position;
			double dis_threshold = descriptor.triangle_.norm() * rough_dis_threshold; //  0.01
			// 1. 通过边长相似性寻找可能得候选特征
			// 2. 候选特征满足帧id与当前id相差超过50，边长差异小于1%，二值描述子相似度超过0.7
			for (auto voxel_inc : voxel_round)
			{
				position.x = (int)(descriptor.triangle_[0] + voxel_inc[0]);
				position.y = (int)(descriptor.triangle_[1] + voxel_inc[1]);
				position.z = (int)(descriptor.triangle_[2] + voxel_inc[2]);
				Eigen::Vector3d voxel_center((double)position.x + 0.5, (double)position.y + 0.5, (double)position.z + 0.5);
				if ((descriptor.triangle_ - voxel_center).norm() >= 1.5) continue;
				auto iter = descriptor_map.find(position);
				if (iter == descriptor_map.end()) continue;
				for (size_t j = 0; j < descriptor_map[position].size(); j++)
				{
					if ((descriptor.frame_number_ - descriptor_map[position][j].frame_number_) <= skip_near_num) continue; // 50
					if ((descriptor.triangle_ - descriptor_map[position][j].triangle_).norm() >= dis_threshold) continue;
					double similarity = (	binary_similarity(descriptor.binary_A_, descriptor_map[position][j].binary_A_) + // 所有存在点的分段匹配的占比
											binary_similarity(descriptor.binary_B_, descriptor_map[position][j].binary_B_) +
											binary_similarity(descriptor.binary_C_, descriptor_map[position][j].binary_C_)) /3;
					if (similarity > similarity_threshold) // 0.7
					{
						useful_match[i] = true;
						useful_match_position[i].push_back(position);
						useful_match_index[i].push_back(j);
					}
				}
			}
		}
		std::mutex mylock;
		auto t0 = std::chrono::high_resolution_clock::now();
		std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> index_recorder;
		auto t1 = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < useful_match.size(); i++)
		{
			if (!useful_match[i]) continue;
			for (size_t j = 0; j < useful_match_index[i].size(); j++)
			{
				match_array[descriptor_map[useful_match_position[i][j]][useful_match_index[i][j]].frame_number_] += 1;
				Eigen::Vector2i match_index(i, j);
				index_recorder.push_back(match_index);
				match_list_index.push_back(descriptor_map[useful_match_position[i][j]][useful_match_index[i][j]].frame_number_);
			}
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		// 逐步筛选匹配数量最高的帧，考虑匹配数量最高的5个帧，保存匹配帧id和其中所有匹配的特征
		for (int cnt = 0; cnt < candidate_num; cnt++) // 5
		{
			double max_vote = 1;
			int max_vote_index = -1;
			for (int i = 0; i < 20000; i++)
			{
				if (match_array[i] > max_vote)
				{
					max_vote = match_array[i];
					max_vote_index = i;
				}
			}
			if (max_vote_index < 0) continue;
			STDMatchList match_triangle_list;
			match_array[max_vote_index] = 0;
			match_triangle_list.match_frame_ = max_vote_index; // 匹配帧id
			double mean_dis = 0;
			for (size_t i = 0; i < index_recorder.size(); i++)
			{
				if (match_list_index[i] == max_vote_index)
				{
					std::pair<STD, STD> single_match_pair;
					single_match_pair.first = current_STD_list[index_recorder[i][0]];
					single_match_pair.second = descriptor_map[useful_match_position[index_recorder[i][0]][index_recorder[i][1]]][useful_match_index[index_recorder[i][0]][index_recorder[i][1]]];
					match_triangle_list.match_list_.push_back(single_match_pair); // 匹配帧中所有匹配的特征
				}
			}
			alternative_match.push_back(match_triangle_list);
		}
		auto t3 = std::chrono::high_resolution_clock::now();
	}

	void fine_loop_detection_tbb(std::vector<std::pair<STD, STD>> &match_list, bool &fine_sucess, Eigen::Matrix3d &std_rot, Eigen::Vector3d &std_t, std::vector<std::pair<STD, STD>> &sucess_match_list, std::vector<std::pair<STD, STD>> &unsucess_match_list)
	{
		sucess_match_list.clear();
		unsucess_match_list.clear();
		double dis_threshold = 3;
		fine_sucess = false;
		std::time_t solve_time = 0;
		std::time_t verify_time = 0;
		int skip_len = (int)(match_list.size() / 50) + 1;
		int use_size = match_list.size() / skip_len;
		std::vector<size_t> index(use_size);
		std::vector<int> vote_list(use_size);
		// 均匀采样50个匹配特征计算位姿变化，并记录特征通过位姿变换后基本能满足匹配的数量
		for (size_t i = 0; i < index.size(); i++)
		{
			index[i] = i;
			auto single_pair = match_list[i * skip_len];
			int vote = 0;
			Eigen::Matrix3d test_rot;
			Eigen::Vector3d test_t;
			triangle_solver(single_pair, test_rot, test_t);
			for (size_t j = 0; j < match_list.size(); j++)
			{
				auto verify_pair = match_list[j];
				Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
				Eigen::Vector3d A_transform = test_rot * A + test_t;
				Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
				Eigen::Vector3d B_transform = test_rot * B + test_t;
				Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
				Eigen::Vector3d C_transform = test_rot * C + test_t;
				double dis_A = (A_transform - verify_pair.second.binary_A_.location_).norm();
				double dis_B = (B_transform - verify_pair.second.binary_B_.location_).norm();
				double dis_C = (C_transform - verify_pair.second.binary_C_.location_).norm();
				if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold) // 3.0
					vote++;
			}
			vote_list[i] = vote;
		}
		auto t0 = std::chrono::high_resolution_clock::now();
		// 找出最能满足匹配的位姿
		int max_vote_index = 0;
		int max_vote = 0;
		for (size_t i = 0; i < vote_list.size(); i++) {
			if (max_vote < vote_list[i]) {
				max_vote_index = i;
				max_vote = vote_list[i];
			}
		}
		// 至少4个匹配特征在位姿变化后相差满足条件
		if (max_vote >= ransac_Rt_thr) // 4 
		{
			fine_sucess = true;
			auto best_pair = match_list[max_vote_index * skip_len];
			int vote = 0;
			Eigen::Matrix3d test_rot;
			Eigen::Vector3d test_t;
			triangle_solver(best_pair, test_rot, test_t);
			std_rot = test_rot;
			std_t = test_t;
			for (size_t j = 0; j < match_list.size(); j++)
			{
				auto verify_pair = match_list[j];
				Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
				Eigen::Vector3d A_transform = test_rot * A + test_t;
				Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
				Eigen::Vector3d B_transform = test_rot * B + test_t;
				Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
				Eigen::Vector3d C_transform = test_rot * C + test_t;
				double dis_A = (A_transform - verify_pair.second.binary_A_.location_).norm();
				double dis_B = (B_transform - verify_pair.second.binary_B_.location_).norm();
				double dis_C = (C_transform - verify_pair.second.binary_C_.location_).norm();
				if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold) // 3.0
					sucess_match_list.push_back(verify_pair);
				else
					unsucess_match_list.push_back(verify_pair);
			}
		} 
	}

	double geometric_verify(const pcl::PointCloud<PointType>::Ptr &source_cloud, const pcl::PointCloud<PointType>::Ptr &target_cloud, const Eigen::Matrix3d &rot, const Eigen::Vector3d &t)
	{
		pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (size_t i = 0; i < target_cloud->size(); i++) {
			pcl::PointXYZ pi;
			pi.x = target_cloud->points[i].x;
			pi.y = target_cloud->points[i].y;
			pi.z = target_cloud->points[i].z;
			input_cloud->push_back(pi);
		}
		kd_tree->setInputCloud(input_cloud);
		// 创建两个向量，分别存放近邻的索引值、近邻的中心距
		std::vector<int> pointIdxNKNSearch(1);
		std::vector<float> pointNKNSquaredDistance(1);
		double useful_match = 0;
		// 搜索当前点通过位姿变化后的点在历史数据中的最近点，如果最近点的法线差异小于0.1，点到面的距离差异小于0.5，则认为成功匹配
		for (size_t i = 0; i < source_cloud->size(); i++) {
			PointType searchPoint = source_cloud->points[i];
			pcl::PointXYZ use_search_point;
			Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
			pi = rot * pi + t;
			use_search_point.x = pi[0];
			use_search_point.y = pi[1];
			use_search_point.z = pi[2];
			Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
			ni = rot * ni;
			if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
				PointType nearstPoint = target_cloud->points[pointIdxNKNSearch[0]];
				Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
				Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);
				Eigen::Vector3d normal_inc = ni - tni;
				Eigen::Vector3d normal_add = ni + tni;
				double point_to_plane = fabs(tni.transpose() * (pi - tpi));
				if ((normal_inc.norm() < normal_threshold || normal_add.norm() < normal_threshold) && point_to_plane < dis_threshold) { // 0.1  0.5
					useful_match++;
				}
			}
		}
		return useful_match / source_cloud->size();
	}

	void add_STD(std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map, std::vector<STD> &STD_list)
	{
		for (auto single_std : STD_list)
		{
			STD_LOC position;
			position.x = (int)(single_std.triangle_[0] + 0.5);
			position.y = (int)(single_std.triangle_[1] + 0.5);
			position.z = (int)(single_std.triangle_[2] + 0.5);
			position.a = (int)(single_std.angle_[0]);
			position.b = (int)(single_std.angle_[1]);
			position.c = (int)(single_std.angle_[2]);
			auto iter = descriptor_map.find(position);
			// single_std.score_frame_.push_back(single_std.frame_number_);
			if (iter != descriptor_map.end()) {
				descriptor_map[position].push_back(single_std);
			} else {
				std::vector<STD> descriptor_list;
				descriptor_list.push_back(single_std);
				descriptor_map[position] = descriptor_list;
			}
		}
	}

	void associate_consecutive_frames(pcl::PointCloud<PointType>::Ptr &corners_curr, pcl::PointCloud<PointType>::Ptr &corners_last, std::vector<std::pair<PointType,PointType>> &corners_pairs)
	{
		if (corners_curr->empty() || corners_last->empty()) return;
		pcl::KdTreeFLANN<PointType>::Ptr kd_tree(new pcl::KdTreeFLANN<PointType>);
		kd_tree->setInputCloud(corners_curr);
		std::vector<int> knn_indices;
		std::vector<float> knn_dis;
		const int k = 1;
		const float pt_dis_thr = 1.0f;
		const float dim_dis_thr = 0.6f;
		// 针对上一帧是回环帧并且优化成功的情况，构建匹配的时候用更新的位姿，但是保存匹配的时候用原来的位姿，这样不会影响后续的操作
		if(!decouple_front && sub_map_id-1>=0 && sub_maps[sub_map_id-1].pose_opt_set)
		{
			Eigen::Matrix4d delta_pose_ = sub_maps[sub_map_id-1].sub_map_pose_opt*sub_maps[sub_map_id-1].sub_map_pose.inverse();
			for (int r = 0; r < corners_last->size(); r++)
			{
				PointType a_pt = corners_last->points[r];
				Eigen::Vector3d pt_(a_pt.x, a_pt.y, a_pt.z);
				pt_ = delta_pose_.block<3,3>(0,0)*pt_ + delta_pose_.block<3,1>(0,3);
				PointType a_pt2;
				a_pt2.x = pt_[0]; a_pt2.y = pt_[1]; a_pt2.z = pt_[2];
				kd_tree->nearestKSearch(a_pt2, k, knn_indices, knn_dis);
				for (int idx = 0; idx < k; idx++)
				{
					PointType pt_nn = corners_curr->points[knn_indices[idx]];
					if (knn_dis[idx] > pt_dis_thr) continue;
					if (fabs(pt_nn.x - a_pt2.x) > dim_dis_thr) continue;
					if (fabs(pt_nn.y - a_pt2.y) > dim_dis_thr) continue;
					if (fabs(pt_nn.z - a_pt2.z) > 0.3) continue;
					corners_pairs.push_back({pt_nn, a_pt});
					break;
				}
			}
		}
		for (int r = 0; r < corners_last->size(); r++)
		{
			PointType a_pt = corners_last->points[r];
			kd_tree->nearestKSearch(a_pt, k, knn_indices, knn_dis);
			for (int idx = 0; idx < k; idx++)
			{
				PointType pt_nn = corners_curr->points[knn_indices[idx]];
				if (knn_dis[idx] > pt_dis_thr) continue;
				if (fabs(pt_nn.x - a_pt.x) > dim_dis_thr) continue;
				if (fabs(pt_nn.y - a_pt.y) > dim_dis_thr) continue;
				if (fabs(pt_nn.z - a_pt.z) > 0.3) continue;
				corners_pairs.push_back({pt_nn, a_pt});
				break;
			}
		}
	}

	void insertKPPairConstraint(const std::vector<std::pair<PointType,PointType>> &corners_pairs, const int prev_idx, const int curr_idx, const int option)
	{
		if (corners_pairs.empty()) return;
		pcl::PointCloud<PointType>::Ptr cloud_prev(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr cloud_curr(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr corners_last(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr corners_curr(new pcl::PointCloud<PointType>());
		for (auto a_pair:corners_pairs)
		{
			corners_curr->push_back(a_pair.first);
			corners_last->push_back(a_pair.second);
		}
		if (option == 1)
		{
			std::vector<std::tuple<gtsam::Vector3, gtsam::Vector3, float>> corner_pairs;
			pcl::transformPointCloud(*corners_curr, *cloud_curr, sub_maps[curr_idx].sub_map_pose.inverse().cast<float>());
			pcl::transformPointCloud(*corners_last, *cloud_prev, sub_maps[prev_idx].sub_map_pose.inverse().cast<float>());
			const int ptn = cloud_curr->size();
			for (int i = 0; i < ptn; i++)
			{
				gtsam::Vector3 pt_curr(cloud_curr->points[i].x, cloud_curr->points[i].y, cloud_curr->points[i].z);
				gtsam::Vector3 pt_prev(cloud_prev->points[i].x, cloud_prev->points[i].y, cloud_prev->points[i].z);
				float dis = pt_prev.norm() + pt_curr.norm();
				corner_pairs.push_back({pt_curr, pt_prev, dis});
			}
			cloud_prev->clear();
			cloud_curr->clear();
			std::sort(corner_pairs.begin(), corner_pairs.end(), [](std::tuple<gtsam::Vector3,gtsam::Vector3,float> a, std::tuple<gtsam::Vector3,gtsam::Vector3,float> b) { return (std::get<2>(a) < std::get<2>(b)); });
			int max_num = corner_pairs.size() > pairfactor_num ? pairfactor_num : corner_pairs.size();
			for (int i = 0; i < max_num; i++)
			{
				gtsam::Point3 pt_curr_body = std::get<0>(corner_pairs[i]);
				gtsam::Point3 pt_prev_body = std::get<1>(corner_pairs[i]);
				boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3>> tmp(new gtsam::TunningPointPairsFactor<gtsam::Pose3>(prev_idx, curr_idx, pt_prev_body, pt_curr_body, point_noise_));
				gts_graph_.add(tmp);
				gts_graph_recover_.add(tmp);
				PointType a_pt;
				a_pt.x = pt_prev_body[0];
				a_pt.y = pt_prev_body[1];
				a_pt.z = pt_prev_body[2];
				cloud_prev->push_back(a_pt);
				a_pt.x = pt_curr_body[0];
				a_pt.y = pt_curr_body[1];
				a_pt.z = pt_curr_body[2];
				cloud_curr->push_back(a_pt);
			}
			debug_file << "[Add Adj KPF]: " << prev_idx << " " << curr_idx << " " << cloud_curr->size() << " " << cloud_prev->size() <<  std::endl;
			// std::cout << "[Add Adj KPF]: " << prev_idx << " " << curr_idx << std::endl;
		} 
		else if (option == 2)
		{
			pcl::transformPointCloud(*corners_last, *cloud_prev, sub_maps[prev_idx].sub_map_pose.inverse().cast<float>());
			pcl::transformPointCloud(*corners_curr, *cloud_curr, sub_maps[curr_idx].sub_map_pose.inverse().cast<float>());
			for (int i = 0; i < cloud_curr->size(); i++)
			{
				gtsam::Point3 pt_prev_body(cloud_prev->points[i].x, cloud_prev->points[i].y, cloud_prev->points[i].z);
				gtsam::Point3 pt_curr_body(cloud_curr->points[i].x, cloud_curr->points[i].y, cloud_curr->points[i].z);
				boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3>> tmp(new gtsam::TunningPointPairsFactor<gtsam::Pose3>(prev_idx, curr_idx, pt_prev_body, pt_curr_body, lc_point_noise_));
				gts_graph_.add(tmp);
			}
			debug_file << "[Add LC KPF] prev_idx: " << prev_idx << " curr_idx: " << curr_idx << " corners_pairs.size(): " << corners_pairs.size() << std::endl;
			std::cout  << "[Add LC KPF] prev_idx: " << prev_idx << " curr_idx: " << curr_idx << " corners_pairs.size(): " << corners_pairs.size() << std::endl;
		}
		VOXEL_LOC prev_curr_idx;
		prev_curr_idx.x = (int)(prev_idx), prev_curr_idx.y = (int)(curr_idx), prev_curr_idx.z = (int)(0);
		auto iter = cornerpairs_uomap.find(prev_curr_idx);
		if (iter == cornerpairs_uomap.end()) // store KP pairs info into containner cornerpairs_uomap
			cornerpairs_uomap[prev_curr_idx] = {cloud_prev, cloud_curr};
	}

	void addOdomFactor()
	{
		if(!has_add_prior_node)
		{
			gtsam::Pose3 pose_prior = trans2gtsamPose(sub_maps[0].sub_map_pose);
			int pose_index = 0;
			if (prior_inserted.find(pose_index) == prior_inserted.end() || !prior_inserted[pose_index])
			{
				gts_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(pose_index, pose_prior, pose_start_noise_));
				gts_graph_recover_.add(gtsam::PriorFactor<gtsam::Pose3>(pose_index, pose_prior, pose_start_noise_));
				prior_inserted[pose_index] = true;
				if (val_inserted.find(pose_index) == val_inserted.end() || !val_inserted[pose_index])
				{
					gts_init_vals_.insert(pose_index, pose_prior);
					val_inserted[pose_index] = true;
				}
				debug_file << "[Add Prior Factor]: prior at " << pose_index << std::endl;
				std::cout << "[Add Prior Factor]: prior at " << pose_index << std::endl;
			}
			graphpose_count++;
			has_add_prior_node = true;
		}
		else
		{
			const int prev_node_idx = graphpose_count - 1;
			const int curr_node_idx = graphpose_count;
			// std::cout << "[Add Odom Factor]: " << prev_node_idx << " " << curr_node_idx << " " << std::endl;
			debug_file << "[Add Odom Factor]: " << prev_node_idx << " " << curr_node_idx << " " << std::endl;
			gtsam::Pose3 pose_prev, pose_curr;
			pose_prev = trans2gtsamPose(sub_maps[prev_node_idx].sub_map_pose);
			if(!decouple_front && sub_maps[prev_node_idx].pose_opt_set) pose_prev = trans2gtsamPose(sub_maps[prev_node_idx].sub_map_pose_opt);
			pose_curr = trans2gtsamPose(sub_maps[curr_node_idx].sub_map_pose);
			Eigen::Matrix4d delta_T_ = gtsam2transPose(pose_prev).inverse()*gtsam2transPose(pose_curr);
			Eigen::Matrix<double, 6,6> w_2 = Eigen::Matrix<double, 6,6>::Identity();
			Eigen::Matrix3d delta_R_ = delta_T_.block<3,3>(0,0);
			Eigen::Vector3d delta_t_ = delta_T_.block<3,1>(0,3);
			w_2.block<3,3>(0,0) = -delta_R_.transpose();
			w_2.block<3,3>(3,0) = delta_R_.transpose()*vec_to_hat(delta_t_);
			w_2.block<3,3>(3,3) = -delta_R_.transpose();
			Eigen::Matrix<double, 6,6> pose_cov = sub_maps[curr_node_idx].sub_map_pose_cov + w_2*sub_maps[prev_node_idx].sub_map_pose_cov*w_2.transpose();
			double p_scale = 1e5;
			pose_cov *= p_scale;
			// std::cout<<"pose_cov: "<<pose_cov.diagonal().transpose()<<std::endl;;
			gtsam::SharedNoiseModel noise_model_ = gtsam::noiseModel::Gaussian::Covariance(pose_cov);
			gts_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_prev.between(pose_curr), noise_model_));
			gts_graph_recover_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_prev.between(pose_curr), noise_model_));
			// gts_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_prev.between(pose_curr), pose_noise_));
			// gts_graph_recover_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_prev.between(pose_curr), pose_noise_));
			if (val_inserted.find(curr_node_idx) == val_inserted.end() || !val_inserted[curr_node_idx])
			{
				gts_init_vals_.insert(curr_node_idx, pose_curr);
				val_inserted[curr_node_idx] = true;
			}
			graphpose_count++;
		}
	}

	void gtsam_optimize(bool pose_rel_opt = false)
	{
		if(pose_rel_opt) lc_prev_idx = -1;
		using namespace std;
		if (!pose_rel_opt && gts_init_vals_.empty() || gts_graph_.empty()) return;
		// std::cout << "[Optimize]: running isam2 optimization ..." << endl;
		debug_file << "[Optimize]: running isam2 optimization ..." << endl;
		auto start1 = std::chrono::system_clock::now();
		// std::cout << "[Optimize]: isam_->backup() ..." << endl;
		isam_->backup(); // self-defined isam function, you need to compile gtsam with provided note in Readme
		// 1. 移除因子 (factors_invalid_ids) ---> 2. 添加新因子 (gts_graph_) ---> 3. 添加新变量初始值 (gts_init_vals_) ---> 4. 进行增量优化
		// 所谓的移除因子并不是真的移除，只是设置为无效，在优化的时候不参与，并没有直接删除，getFactorsUnsafe还能返回
		// 虽然因子可以置为无效，但是变量的添加要严格保证唯一性
		// std::cout << "[Optimize]: isam_->update() ..." << endl;
		if (!factors_invalid_ids.empty() && loop_closure_detected) isam_->update(gts_graph_, gts_init_vals_, factors_invalid_ids);
		else isam_->update(gts_graph_, gts_init_vals_); // if isam->update() triggers segmentation fault, try parameters.factorization = gtsam::ISAM2Params::QR;
		auto isam_result = isam_->update();
		if (loop_closure_detected)
		{
			isam_result = isam_->update();
			isam_result = isam_->update();
			isam_result = isam_->update(); // result more stable if repeat
		}
		auto end1 = std::chrono::system_clock::now();
		auto elapsed_ms = (std::chrono::duration<double, std::milli>(end1 - start1)).count();
		gts_cur_vals_ = isam_->calculateEstimate(); // 从当前优化的因子图中提取变量的最新估计值。
		correctPosesAndPubAndSaveTxt(false, pose_rel_opt);
		if (!loop_closure_detected)
		{
			gts_graph_.resize(0);
			gts_graph_recover_.resize(0);
			gts_init_vals_.clear();
			// debug_file << "[Optimize]: gts_init_vals_.clear() " << std::endl;
			return;
		}
		elapsed_ms_opt += elapsed_ms;
		opt_count++;
		elapsed_ms_max = elapsed_ms > elapsed_ms_max ? elapsed_ms : elapsed_ms_max;
		debug_file << "[Optimize]: optimization 1 takes " << elapsed_ms << "ms" << std::endl;
		debug_file << "[Optimize]: optimization with lc: -----------" << lc_prev_idx << " " << lc_curr_idx << std::endl;
		debug_file << "[Optimize]: optimization avg takes " << elapsed_ms_opt / opt_count << "ms" << std::endl;
		debug_file << "[Optimize]: optimization max takes " << elapsed_ms_max << "ms" << std::endl;
		bool bad_loopclosure = checkResiduals();
		// 如果回环偏差太小，是不是就不用发不了？？？
		if(!associate_consecutive_frame) // 对于nclt数据集，前端的位姿精度较高，有时候偏差太小的回环反而有不好的效果。。。
		{
			gtsam::Pose3 pose_optimized = gts_cur_vals_.at<gtsam::Pose3>(gtsam::Symbol(lc_curr_idx));
			Eigen::Matrix4d sub_map_pose_opt = gtsam2transPose(pose_optimized);
			double pose_drift_ = (sub_map_pose_opt.block<3,1>(0,3) - (lc_delta_pose*sub_maps[lc_curr_idx].sub_map_pose).block<3,1>(0,3)).norm();
			debug_file<<"lc_curr_idx: "<<lc_curr_idx<<", pose_drift_: "<<pose_drift_<<std::endl;
			std::cout<<"lc_curr_idx: "<<lc_curr_idx<<", pose_drift_: "<<pose_drift_<<std::endl;
			new_added_lc_factors.back().pose_drift0 = pose_drift_;
			if(pose_drift_<=0.01)
			{
				debug_file<<"Too small pos drift, reject LC"<<std::endl;
				std::cout<<"Too small pos drift, reject LC"<<std::endl;
				bad_loopclosure = true;
			}
		}
		if (bad_loopclosure)
		{
			debug_file << "[FPR]: reject, large residual appear." << std::endl;
			std::cout << "[FPR]: reject, large residual appear." << std::endl;
			new_added_lc_factors.back().prev_opt = sub_maps[lc_prev_idx].sub_map_pose_opt;
			new_added_lc_factors.back().curr_opt = sub_maps[lc_curr_idx].sub_map_pose_opt;
			if(new_added_lc_factors.size()>0)
			{
				invalid_added_lc_factors.insert(invalid_added_lc_factors.end(), new_added_lc_factors.begin(), new_added_lc_factors.end());
			}
			isam_->recover(); // self-defined isam function, you need to compile gtsam with provided note in Readme
			{
				isam_->update(gts_graph_recover_, gts_init_vals_);
				gts_cur_vals_ = isam_->calculateEstimate();
				correctPosesAndPubAndSaveTxt();
				gts_graph_.resize(0);
				gts_graph_recover_.resize(0);
				gts_init_vals_.clear();
				if(lc_results.back().current_submap_id==lc_curr_idx && lc_results.back().target_submap_id==lc_prev_idx) lc_results.pop_back();
			}
			new_added_lc_factors.clear();
		} 
		else
		{
			debug_file << "[FPR]: accept, everything is ok." << std::endl;
			std::cout << "[FPR]: loop closure accept, isam loop optimize end." << std::endl;
			loop_corr_counter++;
			factors_invalid_ids.clear();
			gts_graph_.resize(0);
			gts_graph_recover_.resize(0);
			gts_init_vals_.clear();
			submap_ids_w_lc.insert(lc_curr_idx);
			if(new_added_lc_factors.size()>0)
			{
				valid_added_lc_factors.insert(valid_added_lc_factors.end(), new_added_lc_factors.begin(), new_added_lc_factors.end());
				new_added_lc_factors.clear();
			}
			correctPosesAndPubAndSaveTxt(true, pose_rel_opt);
			if(!pose_rel_opt)
			{
				lc_delta_pose = sub_maps[lc_curr_idx].sub_map_pose_opt*sub_maps[lc_curr_idx].sub_map_pose.inverse(); // 计算回环偏置，用于前端更新之前发过来的帧位姿
				if(associate_consecutive_frame && lc_prev_idx >= 0) marginalizeKPPairFactor(lc_prev_idx, lc_curr_idx); // Marginalize KPF for efficiency when some correct LC are accepted
				if(!decouple_front)
				{
					wait_front_end_done = true;
					last_pose = lc_delta_pose*last_pose; // 也更新一下上一个关键帧位姿，便于构建子图,对于前端结构的就不需要了
				}
				// corners_last_->clear(); // 暂时不建立顶点关联，不过后期可以考虑(TODO)
				if (lc_prev_idx >= 0) optimizePubKeyFramePoses(sub_maps[lc_prev_idx].end_key_frame_id, sub_maps[lc_curr_idx].end_key_frame_id, true);
			}
		}
		last_lc_prev_idx = lc_prev_idx;
		last_lc_curr_idx = lc_curr_idx;
		loop_closure_detected = false;
	}

	void correctPosesAndPubAndSaveTxt(bool pub_pgo_path = false, bool pose_rel_opt = false)
	{
		std::fstream poses_opt_file;
		if(pose_rel_opt) poses_opt_file = std::fstream(save_directory + "/poses_ppsam_pose_rel_opt.txt", std::fstream::out);
		else poses_opt_file = std::fstream(save_directory + "/poses_ppsam.txt", std::fstream::out);
		poses_opt_file.precision(std::numeric_limits<double>::max_digits10);
		int sub_map_poses_n = sub_maps.size();
		// std::cout << "save pose txt, sub_map_poses_n: "<<sub_map_poses_n << std::endl;
		// gts_cur_vals_.print("gts_cur_vals_: ");
		nav_msgs::PathPtr pathAftPGO (new nav_msgs::Path());
		pathAftPGO->header.frame_id = "/world"; //"/world";
		pathAftPGO->header.stamp = sub_maps[sub_map_poses_n-1].sub_map_time;
		for (int i = 0; i < sub_map_poses_n; i++)
		{
			gtsam::Pose3 pose_optimized = gts_cur_vals_.at<gtsam::Pose3>(gtsam::Symbol(i));
			sub_maps[i].sub_map_pose_opt = gtsam2transPose(pose_optimized);
			if(valid_added_lc_factors.size()==0) pose_optimized = trans2gtsamPose(sub_maps[i].sub_map_pose); // 如果没有有效回环，就直接用原始位姿
			Eigen::Quaterniond q(pose_optimized.rotation().matrix());
			poses_opt_file << sub_maps[i].sub_map_time.toSec() << " " << pose_optimized.x() << " " << pose_optimized.y() << " " << pose_optimized.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
			if(!pub_pgo_path) continue;
			sub_maps[i].pose_opt_set = true; // 回环优化成功之后才置为 true
			geometry_msgs::PoseStamped poseStampAftPGO;
			poseStampAftPGO.header.frame_id = "/world";
			poseStampAftPGO.header.stamp = sub_maps[i].sub_map_time;
			poseStampAftPGO.header.seq = sub_maps[i].end_key_frame_id;
			poseStampAftPGO.pose.position.x = pose_optimized.x();
			poseStampAftPGO.pose.position.y = pose_optimized.y();
			poseStampAftPGO.pose.position.z = pose_optimized.z();
			poseStampAftPGO.pose.orientation.x = q.x();
			poseStampAftPGO.pose.orientation.y = q.y();
			poseStampAftPGO.pose.orientation.z = q.z();
			poseStampAftPGO.pose.orientation.w = q.w();
			pathAftPGO->poses.push_back(poseStampAftPGO);
		}
		if(pub_pgo_path) pubPathAftPGO.publish(pathAftPGO);
		debug_file << "pub and save pose txt end " << std::endl;
		// std::cout << "save pose txt end " << std::endl;
		poses_opt_file.close();
	}

	bool checkResiduals()
	{
		std::unordered_map<VOXEL_LOC, bool> tmp;
		bool outflag = false;
		const gtsam::NonlinearFactorGraph graph_cur = isam_->getFactorsUnsafe();
		for (const boost::shared_ptr<gtsam::NonlinearFactor> &factor : graph_cur) // Iterate over currect factor in the graph
		{
			boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3>> triBetween = boost::dynamic_pointer_cast<gtsam::TunningPointPairsFactor<gtsam::Pose3>>(factor);
			// 这里还是有点风险，不知道标记为无效的因子能否转化
			if(!triBetween) continue; // If it is a Adjacent KPF
			gtsam::Key key1 = triBetween->key1();
			gtsam::Key key2 = triBetween->key2();
			Eigen::Matrix4d T_2T1_ = sub_maps[key2].sub_map_pose_opt.inverse()*sub_maps[key1].sub_map_pose_opt;
			VOXEL_LOC prev_curr_idx;
			prev_curr_idx.x = (int)(key1), prev_curr_idx.y = (int)(key2), prev_curr_idx.z = (int)(0);
			if (tmp.find(prev_curr_idx) != tmp.end()) continue;
			if (cornerpairs_uomap.find(prev_curr_idx) == cornerpairs_uomap.end()) continue;
			pcl::PointCloud<PointType>::Ptr cloud_prev = cornerpairs_uomap[prev_curr_idx].first;
			pcl::PointCloud<PointType>::Ptr cloud_curr = cornerpairs_uomap[prev_curr_idx].second;
			if (cloud_prev->empty() || cloud_curr->empty()) continue;
			tmp[prev_curr_idx] = true;
			pcl::PointCloud<PointType>::Ptr cloud_prev_tran(new pcl::PointCloud<PointType>());
			pcl::transformPointCloud(*(cloud_prev), *cloud_prev_tran, T_2T1_.cast<float>());
			int pairs_size = int(cloud_prev_tran->size());
			float residual_max = 0;
			for (int j = 0; j < pairs_size; j++)
			{
				const gtsam::Point3 q(cloud_prev_tran->points[j].x, cloud_prev_tran->points[j].y, cloud_prev_tran->points[j].z);
				const gtsam::Point3 p2(cloud_curr->points[j].x, cloud_curr->points[j].y, cloud_curr->points[j].z);
				float residual = (q - p2).norm();
				residual_max = residual > residual_max ? residual : residual_max;
			}
			// debug_file << "key1: "<<key1<<", key2: "<< key2 << ", residual_max: " << residual_max << std::endl;
			if (residual_max > residual_thr)
			{
				debug_file << "!!!! Too large Residual, checkResidual fails !!!"<<std::endl;
				// std::cout << "!!!! Too large Residual, checkResidual fails !!!"<<std::endl;
				outflag = true;
				return outflag; // can uncomment this line for efficiency
			}
		}
		return outflag;
	}

	// 除了当前回环对应的顶点约束之外，其他回环全部转化为相对位姿约束，当前回环之间的顶点约束和相对位姿约束也转化为相对位姿约束
	void marginalizeKPPairFactor(const int lc_start, const int lc_end) // Marginalize KPF for efficiency when some correct LC are accepted
	{
		const gtsam::NonlinearFactorGraph graph_cur = isam_->getFactorsUnsafe(); // 名称中包含 Unsafe，意味着调用此函数时可能会绕过某些安全检查
		std::unordered_map<VOXEL_LOC, bool> tmp;
		for (int i = 0; i < graph_cur.size(); i++)
		{
			if (factor_invalid_ids_map.find(i) != factor_invalid_ids_map.end()) continue;
			auto afactor_keys = graph_cur[i]->keys();
			if (afactor_keys.size() != 2) continue;
			if (afactor_keys[0] >= lc_start && afactor_keys[1] <= lc_end && afactor_keys[1] - afactor_keys[0] <= 2) // 边缘化掉当前回环帧之间相邻帧的约束（包括顶点约束和相对位姿约束）
			{
				factors_invalid_ids.push_back(i);
				factor_invalid_ids_map[i] = true;
			}
			if (afactor_keys[0] != lc_start && afactor_keys[1] != lc_end && afactor_keys[1] - afactor_keys[0] > 2) // 边缘化掉非当前回环帧的回环帧之间的约束（只有顶点约束，因为没有添加相对位姿约束）
			{
				factors_invalid_ids.push_back(i);
				factor_invalid_ids_map[i] = true;
				VOXEL_LOC position;
				position.x = (int)(afactor_keys[0]), position.y = (int)(afactor_keys[1]), position.z = (int)(0);
				if (tmp.find(position) == tmp.end()) tmp[position] = true;
			}
		}
		for (int i = lc_start; i < lc_end; i++) // 当前回环之间的帧建立相邻帧的相对位姿约束
		{
			const gtsam::Value &estimation_last = isam_->calculateEstimate(i);
			const gtsam::Value &estimation_curr = isam_->calculateEstimate(i + 1);
			const gtsam::Pose3 &pose_curr = estimation_curr.cast<gtsam::Pose3>();
			const gtsam::Pose3 &pose_last = estimation_last.cast<gtsam::Pose3>();
			refined_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i, i + 1, pose_last.between(pose_curr), pose_noise_));
		}
		for (auto iter = tmp.begin(); iter != tmp.end(); ++iter) // 非当前回环帧之间建立相对位姿约束
		{
			const gtsam::Value &estimation_last = isam_->calculateEstimate(iter->first.x);
			const gtsam::Value &estimation_curr = isam_->calculateEstimate(iter->first.y);
			const gtsam::Pose3 &pose_last = estimation_last.cast<gtsam::Pose3>();
			const gtsam::Pose3 &pose_curr = estimation_curr.cast<gtsam::Pose3>();
			refined_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(iter->first.x, iter->first.y, pose_last.between(pose_curr), pose_noise2_));
		}
		gts_graph_ = refined_graph;
		gts_graph_recover_ = refined_graph;
		refined_graph.resize(0);
	}

	// 根据子图位姿优化关键帧之间的关键帧的位姿
	void optimizePubKeyFramePoses(const int lc_start, const int lc_end, bool pub_kf_path = false)
	{
		const int sub_map_poses_n = sub_maps.size();
		const int key_frame_pose_n = key_frame_poses.size();
		for (int i = 0; i < sub_map_poses_n; i++)
		{
			gtsam::Pose3 pose_prior = trans2gtsamPose(sub_maps[i].sub_map_pose_opt);
			gts_graph_2.add(gtsam::PriorFactor<gtsam::Pose3>(sub_maps[i].end_key_frame_id, pose_prior, pose_start_noise_));
		}
		// for(auto keyValue : gts_init_vals_2)	std::cout << "Key: " << keyValue.key << ", ";
		// std::cout << std::endl;
		// 开始优化
		std::cout<<"开始优化"<<std::endl;
		if (!factors_invalid_ids2.empty())
			isam_2->update(gts_graph_2, gts_init_vals_2, factors_invalid_ids2);
		else
			isam_2->update(gts_graph_2, gts_init_vals_2); // if isam->update() triggers segmentation fault, try parameters.factorization = gtsam::ISAM2Params::QR;
		auto isam_result = isam_2->update();
		{
			isam_result = isam_2->update();
			isam_result = isam_2->update();
			isam_result = isam_2->update(); // result more stable if repeat
		}
		// 保存结果
		std::cout<<"保存和发布结果"<<std::endl;
		nav_msgs::PathPtr pathAftPGO(new nav_msgs::Path());
		pathAftPGO->header.frame_id = "/world";
		pathAftPGO->header.stamp = sub_maps.back().sub_map_time;
		gts_cur_vals_2 = isam_2->calculateEstimate(); // 从当前优化的因子图中提取变量的最新估计值。
		if(pub_kf_path) // 记录回环信息
		{
			geometry_msgs::PoseStamped poseStampAftPGO;
			poseStampAftPGO.header.frame_id = "/world";
			poseStampAftPGO.header.stamp = key_frame_poses[0].key_frame_time;
			poseStampAftPGO.header.seq = 0;
			poseStampAftPGO.pose.position.x = lc_prev_idx; // 回环目标子图id
			poseStampAftPGO.pose.position.y = lc_curr_idx; // 回环当前子图id
			poseStampAftPGO.pose.position.z = lc_start; // 回环目标关键帧id
			poseStampAftPGO.pose.orientation.x = lc_end; // 回环当前关键帧id
			pathAftPGO->poses.push_back(poseStampAftPGO);
		}
		for (int i = 0; i < key_frame_pose_n; i++)
		{
			if(i>lc_end)
			{
				std::cout<<"i: "<<i<<", lc_end: "<<lc_end<<", key_frame_pose_n: "<<key_frame_pose_n<<std::endl;
				ROS_INFO("shutting down!");
				ros::shutdown();
			}
			const gtsam::Pose3 &pose_optimized = gts_cur_vals_2.at<gtsam::Pose3>(gtsam::Symbol(i));
			key_frame_poses[i].key_frame_pose_opt = gtsam2transPose(pose_optimized);
			key_frame_poses[i].pose_opt_set = true;
			if(pub_kf_path)
			{
				Eigen::Quaterniond q(pose_optimized.rotation().matrix());
				geometry_msgs::PoseStamped poseStampAftPGO;
				poseStampAftPGO.header.frame_id = "/world";
				poseStampAftPGO.header.stamp = key_frame_poses[i].key_frame_time;
				poseStampAftPGO.header.seq = key_frame_poses[i].key_frame_id;
				poseStampAftPGO.pose.position.x = pose_optimized.x();
				poseStampAftPGO.pose.position.y = pose_optimized.y();
				poseStampAftPGO.pose.position.z = pose_optimized.z();
				poseStampAftPGO.pose.orientation.x = q.x();
				poseStampAftPGO.pose.orientation.y = q.y();
				poseStampAftPGO.pose.orientation.z = q.z();
				poseStampAftPGO.pose.orientation.w = q.w();
				pathAftPGO->poses.push_back(poseStampAftPGO);
			}
		}
		if(pub_kf_path) pubKFPathAftPGO.publish(pathAftPGO);
		factors_invalid_ids2.clear();
		gts_graph_2.resize(0);
		gts_init_vals_2.clear();
		// 更新因子图
		std::cout<<"更新因子图"<<std::endl;
		const gtsam::NonlinearFactorGraph graph_cur = isam_2->getFactorsUnsafe(); // 名称中包含 Unsafe，意味着调用此函数时可能会绕过某些安全检查
		for (int i = 0; i < graph_cur.size(); i++)
		{
			if(!graph_cur[i] || factor_invalid_ids_map2.find(i) != factor_invalid_ids_map2.end()) continue;
			boost::shared_ptr<gtsam::PriorFactor<gtsam::Pose3>> priorPose = boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(graph_cur[i]);
			if (priorPose && priorPose->key()>=lc_start && priorPose->key()<=lc_end)
			{
				factors_invalid_ids2.push_back(i);
				factor_invalid_ids_map2[i] = true;
				continue;
			}
			auto afactor_keys = graph_cur[i]->keys();
			if (afactor_keys.size() != 2) continue;
			if (afactor_keys[0] >= lc_start && afactor_keys[1] <= lc_end) // 边缘化掉当前回环帧之间相邻帧的约束（包括顶点约束和相对位姿约束）
			{
				factors_invalid_ids2.push_back(i);
				factor_invalid_ids_map2[i] = true;
			}
		}
		for (int i = lc_start; i < lc_end; i++) // 当前回环之间的帧建立相邻帧的相对位姿约束
		{
			const gtsam::Value &estimation_last = isam_2->calculateEstimate(i);
			const gtsam::Value &estimation_curr = isam_2->calculateEstimate(i + 1);
			const gtsam::Pose3 &pose_curr = estimation_curr.cast<gtsam::Pose3>();
			const gtsam::Pose3 &pose_last = estimation_last.cast<gtsam::Pose3>();
			refined_graph2.add(gtsam::BetweenFactor<gtsam::Pose3>(i, i + 1, pose_last.between(pose_curr), pose_noise_));
		}
		gts_graph_2 = refined_graph2;
		refined_graph2.resize(0);
	}

	// 将子图中包含的每帧的平面进行融合，得到子图自己的平面数据
    void merge_plane_for_submap(KeyFramePose & kf_pose, std_msgs::Float64MultiArray::ConstPtr msg, bool debug = true)
    {
		// debug_file << "merge_plane_for_submap" << std::endl;
		auto merge_plane_start = std::chrono::system_clock::now();
		if(msg->layout.data_offset != kf_pose.key_frame_id)
		{
			std::cout<<"kf_pose.key_frame_id: "<<kf_pose.key_frame_id<<", msg->layout.data_offset: "<<msg->layout.data_offset<<std::endl;
			return;
		}
		int plane_n_ = msg->layout.dim[0].size;
        int merge_plane_n_ = msg->layout.dim[2].size;
        int plane_grid_centers_n = msg->layout.dim[3].size;
		int C_ij_float_num = plane_n_*16;
        int center_0_float_num = merge_plane_n_*3;
		int find_merge_plane_n = 0, find_comm_plane_n = 0;
		const int key_frame_id = kf_pose.key_frame_id;
        Eigen::Matrix3d R_w_l = kf_pose.key_frame_pose.block<3,3>(0,0);
        Eigen::Vector3d t_w_l = kf_pose.key_frame_pose.block<3,1>(0,3);
        Eigen::Matrix<double, 4,4> T_j = kf_pose.key_frame_pose;
		std::vector<PlaneSimple*> planes(plane_n_, nullptr);
		for (size_t pl_i = 0; pl_i < plane_n_; ++pl_i)
		{
			PlaneSimple * local_plane_ = new PlaneSimple();
			local_plane_->local_plane_id = pl_i;
			planes[pl_i] = local_plane_;
			Eigen::Matrix4d C_ij;
			for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				C_ij(r, c) = msg->data[pl_i * 16 + r * 4 + c];
			double N_i = C_ij(3, 3); // N_i
			if(N_i<3) continue;
			Eigen::Vector3d center = C_ij.block<3, 1>(0, 3)/N_i; // 1/N_i*v_i = 1/N_i*S_p*C_i*F
			Eigen::Matrix3d covariance = C_ij.block<3, 3>(0, 0)/N_i - center * center.transpose();
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);
			Eigen::Vector3d lambdas = saes.eigenvalues();
			if(lambdas[0]<0 || lambdas[1]<0 || lambdas[2]<0 || isnan(lambdas[0]) || isnan(lambdas[1]) || isnan(lambdas[2])) // Bad eigen!
			{
				std::cout<<"pl_i: "<<pl_i<<", lambdas: "<<lambdas.transpose()<<std::endl;
				continue;
			}
			Eigen::Matrix3d eigenvectors = saes.eigenvectors();
			local_plane_->points_size = N_i;
			local_plane_->center = center;
			local_plane_->covariance = covariance;
			local_plane_->ready_to_pca = true;
			local_plane_->normal = eigenvectors.col(0);
			local_plane_->y_normal = eigenvectors.col(1);
			local_plane_->x_normal = eigenvectors.col(2);
			local_plane_->min_eigen_value = lambdas(0);
			local_plane_->mid_eigen_value = lambdas(1);
			local_plane_->max_eigen_value = lambdas(2);
			local_plane_->d = - local_plane_->normal.dot(center);
			if(local_plane_->d<0)
			{
				local_plane_->d *= -1;
				local_plane_->normal *= -1;
			}
			local_plane_->is_plane = true;
            Eigen::Matrix<double, 4,4> T_j_C_ij_T_j_T = T_j*C_ij*T_j.transpose();
            Eigen::Vector3d normal_g = R_w_l * local_plane_->normal; // 全局坐标系下的平面法线
            float d_g = local_plane_->d - normal_g.transpose() * t_w_l; // 全局坐标系下的平面参数d
            Eigen::Vector3d center_ = R_w_l * local_plane_->center + t_w_l;
            int near_gplane_id = -1;
            Eigen::Vector4d cos_dist_best = Eigen::Vector4d::Zero();
            Eigen::Vector3d near_pt_best = Eigen::Vector3d::Zero();
            int gp_id_best = -1, gp_id_best2 = -1;
            double cost_best = 0;
            double max_eigen_dist_ = local_plane_->get_max_eigen_dist();
            double mid_eigen_dist_ = local_plane_->get_mid_eigen_dist();
            Eigen::Vector3d x_normal_g = R_w_l*local_plane_->x_normal;
            Eigen::Vector3d y_normal_g = R_w_l*local_plane_->y_normal;
            bool special_plane_flag = false;
            if(pl_i>=merge_plane_n_) // 单独针对某些特定的普通平面，这些平面的法线与中心点到原点的连线的夹角接近90度，在进行平面匹配时候特殊对待，构成匹配的平面在主方向上不应该偏差太多
            {
                double cos_angle_w_z = abs(local_plane_->normal.dot(local_plane_->center))/local_plane_->center.norm();
                double angle_w_z = 90-acos(cos_angle_w_z)/3.1415926*180;
                if(angle_w_z<15) special_plane_flag=true;
            }
            if(debug) std::cout<<"pl_i: "<<pl_i<<", plane_n_: "<<plane_n_<<std::endl;
            if(debug && key_frame_id==1) std::cout<<"octree_kf_plane_center4submap_ptr->size(): "<<octree_kf_plane_center4submap_ptr->size()<<std::endl;
            if(octree_kf_plane_center4submap_ptr->size()>0) // 寻找已经存在的全局平面
            {
                std::vector<std::vector<float>> resultIndices;
                std::vector<float> distances;
                bool find_global_plane = false;
                if(local_plane_->local_plane_id<merge_plane_n_) // 防止融合平面中心处于非平面块区域
                {
					for (int c = 0; c < 3; ++c)
                        local_plane_->center_0[c] = msg->data[C_ij_float_num + pl_i * 3 + c];
                    Eigen::Vector3d center_g_ = R_w_l*local_plane_->center_0 + t_w_l;
                    octree_kf_plane_center4submap_ptr->knnNeighbors_eigen(center_g_, 6, resultIndices, distances);
                }
                else
                {
                    octree_kf_plane_center4submap_ptr->knnNeighbors_eigen(center_, 6, resultIndices, distances);
                }
                double best_cost_ = -2.0, best_dist_ = 100.0, best_angle_ = 20.0, best_cover_ = 100.0;
                std::vector<Eigen::Vector4d> candi_cos_dist;
                std::vector<Eigen::Vector3d> candi_nearest_pt;
                std::vector<int> candi_gpid;
                Eigen::Vector4d cos_weights = {1/15.0, 1.0, 0.5, 1/20.0};
                if(debug) std::cout<<"pl_i: "<<pl_i<<", distances.size(): "<<distances.size()<<std::endl;
                // 考察当前平面与全局平面的第一个平面之间的相似性，全局平面的第一个平面的参数由 R_w_l 确定
                for(int re1=0; re1<distances.size(); re1++)
                {
                    // std::cout<<"distances\n";
                    int gp_id_ = int(resultIndices[re1][4]);
                    if(gp_id_<0 || kf_global_planes4submap[gp_id_]==nullptr) continue;
                    Eigen::Vector3d near_pt = {resultIndices[re1][0], resultIndices[re1][1], resultIndices[re1][2]};
                    GlobalPlaneSimple* near_global_plane_ = kf_global_planes4submap[gp_id_];
                    const Eigen::Vector3d & normal_s = near_global_plane_->normal;
                    Eigen::Vector3d center_l = near_global_plane_->center_g0;
                    // 对于平面关联，最关键的几个参数是： 夹角、最近点在三个特征向量方向的投影距离（三个距离）
                    double angle_ = acos(normal_s.transpose()*normal_g)/3.1415926*180; // 法线夹角
                    // 一般来说 center_l 和 near_pt 是在同一个平面上的，基于平面覆盖率的考虑采用二维距离计算最近点
                    Eigen::Vector3d nearest_pt_ = ((center_l-center_).cross(normal_g)).norm()<((near_pt-center_).cross(normal_g)).norm() ? center_l : near_pt; // 最近点
                    Eigen::Vector3d delta_ = nearest_pt_ - center_;
                    double x_dist_ = abs(x_normal_g.dot(delta_))/max_eigen_dist_; // 最近点在当前平面上的投影在最大特征向量方向与最大特征根距离的比值
                    double y_dist_ = abs(y_normal_g.dot(delta_))/mid_eigen_dist_; // 最近点在当前平面上的投影在中间特征向量方向与中间特征根距离的比值
                    double z_dist_ = abs(normal_g.dot(delta_)); // 最近点到当前平面的距离
                    Eigen::Vector4d cos_dist = {angle_, x_dist_, y_dist_, z_dist_};
                    if(debug && 0)
                    {
                        std::cout<<"cos_dist: "<<cos_dist.transpose()<<std::endl;
                        std::cout<<"normal_g: "<<normal_g.transpose()<<std::endl;
                        std::cout<<"normal_s: "<<normal_s.transpose()<<std::endl;
                        std::cout<<"delta_: "<<delta_.transpose()<<std::endl;
                        std::cout<<"center_: "<<center_.transpose()<<std::endl;
                        std::cout<<"center_l: "<<center_l.transpose()<<std::endl;
                        std::cout<<"near_pt: "<<near_pt.transpose()<<std::endl;
                        std::cout<<"local_plane_->normal: "<<local_plane_->normal.transpose()<<std::endl;
                        std::cout<<"T_j:\n"<<T_j<<std::endl;
                    }
                    if(re1==0)
                    {
                        near_gplane_id = gp_id_;
                        cos_dist_best = cos_dist;
                        near_pt_best = near_pt;
                        gp_id_best = gp_id_;
                    }
                    if(x_dist_>2 || y_dist_>5 || angle_>20 || z_dist_>0.10) continue; // 0.25=>0.10
                    if(special_plane_flag && x_dist_>1) continue;
                    double cost_ = cos(angle_/20.0*3.1415926/2.0)*cos(z_dist_/0.15*3.1415926/2.0);
                    // 平面分类：融合平面、常规平面、特殊平面
                    if(    (special_plane_flag && y_dist_<4.5) // 特殊平面
                        || (angle_<6 && x_dist_<1.5 && y_dist_<3) // 小夹角
                        || (cost_>0.75) // 高cost
                        || (cost_>0.5 && x_dist_<1.5 && y_dist_<1.5) // 中cost
                        || (x_dist_<1.0 && y_dist_<1.0) // 只考虑覆盖
                        )
                    {
                        candi_gpid.push_back(gp_id_);
                        candi_nearest_pt.push_back(nearest_pt_);
                        candi_cos_dist.push_back(cos_dist);
                    }
                }
                if(debug) std::cout<<"candi_gpid.size(): "<<candi_gpid.size()<<std::endl;
                if(candi_gpid.size()>0)
                {
                    gp_id_best = candi_gpid[0];
                    cos_dist_best = candi_cos_dist[0];
                    near_pt_best = candi_nearest_pt[0];
                    double best_cost_ = cos_weights.dot(candi_cos_dist[0]);
                    for(int candi_i=1; candi_i<candi_gpid.size(); candi_i++)
                    {
                        double cost_ = cos_weights.dot(candi_cos_dist[candi_i]);
                        if(cost_<best_cost_)
                        {
                            gp_id_best = candi_gpid[candi_i];
                            cos_dist_best = candi_cos_dist[candi_i];
                            near_pt_best = candi_nearest_pt[candi_i];
                            best_cost_ = cost_;
                        }
                    }
                    find_global_plane = true;
                }
                if(find_global_plane)
                {
                    if(pl_i<merge_plane_n_) find_merge_plane_n += 1;// 对于融合平面
                    else find_comm_plane_n += 1; // 对于一般平面
                    // std::cout<<"global_plane_\n";
                    GlobalPlaneSimple* global_plane_ = kf_global_planes4submap[gp_id_best];
                    global_plane_->C_ijs.push_back(C_ij);
                    global_plane_->T_j_C_ij_T_j_Ts.push_back(T_j_C_ij_T_j_T);
                    global_plane_->C += T_j_C_ij_T_j_T;
                    global_plane_->frame_ids.push_back(key_frame_id);
                    global_plane_->plane_ptrs.push_back(local_plane_);
                    global_plane_->weight += local_plane_->get_weight();
                    local_plane_->global_plane_id = gp_id_best;
                    if(global_plane_->last_frame_id==key_frame_id) // 避免将同一帧中本来不属于同一平面的平面匹配到同一平面
                    {
                        double d1 = abs(local_plane_->normal.dot(global_plane_->last_plane_ptr->center) + local_plane_->d); // 上一个平面中心到当前平面的距离
                        double d2 = abs(global_plane_->last_plane_ptr->normal.dot(local_plane_->center) + global_plane_->last_plane_ptr->d); // 当前平面中心到上一个平面的距离
                        double d3 = (local_plane_->center-global_plane_->last_plane_ptr->center).norm();
                        if(d1>0.03 && d2>0.03 && d1/d3>0.1 && d2/d3>0.1 || local_plane_->local_plane_id<merge_plane_n_ && d1>0.01 && d2>0.01)
                        {
                            Eigen::Vector3d center_g = R_w_l * local_plane_->center + t_w_l;
                            Eigen::Vector3d last_center_g = R_w_l * global_plane_->last_plane_ptr->center + t_w_l;
                            Eigen::Vector3d last_normal_g1 = R_w_l * global_plane_->last_plane_ptr->normal;
                            Eigen::Vector3d normal_g1 = R_w_l * global_plane_->last_plane_ptr->normal;
                            double d4 = abs(global_plane_->normal.dot(center_g) + global_plane_->param_d); // 当前平面中心到全局平面的距离
                            double d5 = abs(global_plane_->normal.dot(last_center_g) + global_plane_->param_d); // 上一个平面中心到全局平面的距离
                            double angle1 = acos(global_plane_->last_plane_ptr->normal.transpose()*local_plane_->normal)/3.1415926*180; // 当前平面与上一个平面法线夹角
                            double angle2 = acos(normal_g1.transpose()*global_plane_->normal)/3.1415926*180; // 当前平面与全局平面法线夹角
                            double angle3 = acos(last_normal_g1.transpose()*global_plane_->normal)/3.1415926*180; // 上一个平面与全局平面法线夹角
                            // std::cout<<"gp id: "<<global_plane_->id<<", d1: "<<d1<<", d2: "<<d2<<", angle1: "<<angle1<<", angle2: "<<angle2<<", angle3: "<<angle3<<", d4: "<<d4<<", d5: "<<d5<<std::endl;
                            std::vector<int> drop_ids;
                            std::vector<double> drop_dists;
                            int min_dist_id = -1;
                            double min_dist_ = 0.0;
                            for(int j=0; j<global_plane_->frame_ids.size(); j++)
                            {
                                if(global_plane_->frame_ids[j] != key_frame_id) continue;
                                Eigen::Vector3d center_g_j = R_w_l * global_plane_->plane_ptrs[j]->center + t_w_l;
                                double dist_ = abs(global_plane_->normal.dot(center_g_j) + global_plane_->param_d); // 平面中心到全局平面的距离
                                if(min_dist_id==-1)
                                {
                                    min_dist_id = j;
                                    min_dist_ = dist_;
                                }
                                else
                                {
                                    if(dist_<min_dist_)
                                    {
                                        drop_ids.push_back(min_dist_id);
                                        drop_dists.push_back(min_dist_);
                                        min_dist_id = j;
                                        min_dist_ = dist_;
                                    }
                                    else
                                    {
                                        drop_ids.push_back(j);
                                        drop_dists.push_back(dist_);
                                    }
                                }
                            }
                            // std::cout<<"min_dist id: "<<min_dist_id<<", min_dist_: "<<min_dist_<<std::endl;
                            global_plane_->last_plane_ptr = global_plane_->plane_ptrs[min_dist_id];;
                            for(int j=drop_ids.size()-1; j>=0; j--)
                            {
                                PlaneSimple * local_plane_0 = global_plane_->plane_ptrs[drop_ids[j]];
                                local_plane_0->global_plane_id = -1;
                                if(1 && local_plane_0->local_plane_id<merge_plane_n_ && local_plane_0->get_max_eigen_dist()>=0.05) // 加入全局平面
                                {
                                    int plane_id_ = kf_global_planes4submap.size();
                                    GlobalPlaneSimple* global_plane_2 = new GlobalPlaneSimple();
                                    Eigen::Vector3d normal_g_ = R_w_l * local_plane_0->normal; // 全局坐标系下的平面法线
                                    float d_g_ = local_plane_0->d - normal_g_.transpose() * t_w_l; // 全局坐标系下的平面参数d
                                    global_plane_2->id = plane_id_;
                                    global_plane_2->normal = normal_g_; // 全局坐标系下的平面法线
                                    global_plane_2->param_d = d_g_;
                                    global_plane_2->C_ijs.push_back(global_plane_->C_ijs[drop_ids[j]]);
                                    global_plane_2->T_j_C_ij_T_j_Ts.push_back(global_plane_->T_j_C_ij_T_j_Ts[drop_ids[j]]);
                                    global_plane_2->C += global_plane_->T_j_C_ij_T_j_Ts[drop_ids[j]];
                                    global_plane_2->frame_ids.push_back(key_frame_id);
                                    global_plane_2->plane_ptrs.push_back(local_plane_0);
                                    global_plane_2->weight += local_plane_0->get_weight();
                                    kf_global_planes4submap.push_back(global_plane_2);
                                    local_plane_0->global_plane_id = plane_id_;
                                }
                                global_plane_->C_ijs.erase(global_plane_->C_ijs.begin() + drop_ids[j]);
                                global_plane_->C -= global_plane_->T_j_C_ij_T_j_Ts[drop_ids[j]];
                                global_plane_->T_j_C_ij_T_j_Ts.erase(global_plane_->T_j_C_ij_T_j_Ts.begin() + drop_ids[j]);
                                global_plane_->frame_ids.erase(global_plane_->frame_ids.begin() + drop_ids[j]);
                                global_plane_->plane_ptrs.erase(global_plane_->plane_ptrs.begin() + drop_ids[j]);
                                global_plane_->weight -= local_plane_0->get_weight();
                                // std::cout<<"drop id: "<<drop_ids[j]<<", dist: "<<drop_dists[j]<<std::endl;
                            }
                        }
                    }
                    else
                    {
                        global_plane_->last_frame_id = key_frame_id;
                        global_plane_->last_plane_ptr = local_plane_;
                    }
                    if(debug) std::cout<<"find_global_plane"<<std::endl;
                    continue;
                }
            }
            // 添加新平面
            if(1)
            {
                if(local_plane_->get_max_eigen_dist()<0.05) continue; // 最大特征值对应的距离太小的直接放弃构建全局平面，误差太大了。。。
                int plane_id_ = kf_global_planes4submap.size();
                GlobalPlaneSimple* global_plane_ = new GlobalPlaneSimple();
                global_plane_->id = plane_id_;
                global_plane_->normal = normal_g;
                global_plane_->param_d = d_g;
				global_plane_->center_g0 = center_;
                global_plane_->C_ijs.push_back(C_ij);
                global_plane_->T_j_C_ij_T_j_Ts.push_back(T_j_C_ij_T_j_T);
                global_plane_->C += T_j_C_ij_T_j_T;
                global_plane_->frame_ids.push_back(key_frame_id);
                global_plane_->plane_ptrs.push_back(local_plane_);
                global_plane_->weight += local_plane_->get_weight();
                kf_global_planes4submap.push_back(global_plane_);
                local_plane_->global_plane_id = plane_id_;
                if(debug) std::cout<<"add plane"<<std::endl;
            }
        }
        if(debug) std::cout<<"merged_planes find ratio: "<<(float)find_merge_plane_n/merge_plane_n_<<", "<<(float)find_merge_plane_n/plane_n_<<std::endl;
        if(debug) std::cout<<"common_planes find ratio: "<<(float)find_comm_plane_n/merge_plane_n_<<", "<<(float)find_comm_plane_n/plane_n_<<std::endl;
        // std::cout<<"update_with_attr\n";
		// debug_file << "merge_plane_for_submap 更新平面中心搜索树" << std::endl;
        // 更新平面中心搜索树
        std::vector<Eigen::Vector3d> plane_grid_centers_(plane_grid_centers_n);
        std::vector<float> init_attr_={float(key_frame_id), -1.0};
        std::vector<std::vector<float>> plane_attrs_(plane_grid_centers_n, init_attr_);
        for(int j=0; j<plane_grid_centers_n; j++)
        {
			for (int c = 0; c < 3; ++c)
                plane_grid_centers_[j][c] = msg->data[C_ij_float_num + center_0_float_num + j * 4 + c];
            plane_grid_centers_[j] = R_w_l*plane_grid_centers_[j] + t_w_l;
			// 将帧平面id转换为全局平面id
            int local_plane_id_ = int(msg->data[C_ij_float_num +center_0_float_num + j * 4 + 3]);
			if(local_plane_id_<0) continue;
            PlaneSimple* local_plane_ =  planes[local_plane_id_];
            plane_attrs_[j][1] = (float)local_plane_->global_plane_id;
        }
        // std::cout<<"plane_grid_centers_n: "<<plane_grid_centers_n<<std::endl;
        octree_kf_plane_center4submap_ptr->update_with_attr(plane_grid_centers_, plane_attrs_, false);
        // std::cout<<"update_with_attr"<<std::endl;
		for (size_t pl_i = 0; pl_i < plane_n_; ++pl_i) // 这里仅仅删除没有加入全局平面的子平面，全局平面和其他平面在后面创新子图平面的时候再删除
		{
			if(planes[pl_i]->global_plane_id<0) delete planes[pl_i];
		}
		auto merge_plane_end = std::chrono::system_clock::now();
		auto merged_planes_ms = std::chrono::duration<double, std::milli>(merge_plane_end - merge_plane_start);
        if(debug) std::cout<<"total time: "<<merged_planes_ms.count()<<std::endl;
		// debug_file << "merge_plane_for_submap end" << std::endl;
    }

	// 通过平面约束构建类似于回环的约束，然后再反馈给因子进行优化，补充回环之间的位姿约束，此外还可以作为小场景的位姿约束
    Eigen::Matrix4d point_plane_optimize(int cur_submap_id = -1, int nearest_submap_id = -1, Eigen::Matrix4d pose_delta = Eigen::Matrix4d::Identity(), bool debug = true)
    {
		int ref_submap_radius = 2;
        int nms_submap_n = 10; // 根据z轴偏差进行非最大值抑制轴偏差进行非最大值抑制的长度
        float search_radius = 20.0; // 子图搜索树半径
        int submap_gap_th = 10; // 关联子图之间的id差异
		int sub_maps_n = sub_maps.size();
        if(sub_maps_n<submap_gap_th) return pose_delta;
        // 2. 构造子图位姿搜索树==========
		if(cur_submap_id<0) cur_submap_id = sub_maps_n-1;
		debug_file<<"======point_plane_optimize_debug=========="<<std::endl;
		debug_file<<"cur_submap_id: "<<cur_submap_id<<", nearest_submap_id: "<<nearest_submap_id<<std::endl;
		Eigen::Matrix4d cur_submap_pose_lc = pose_delta*sub_maps[cur_submap_id].sub_map_pose; // 当前子图的回环位姿
		Eigen::Matrix4d cur_submap_pose_inv = sub_maps[cur_submap_id].sub_map_pose.inverse(); // 当前子图原始位姿的逆
        // 3. 通过搜索树建立子图之间的关联关系==========
        if(nearest_submap_id<0)
		{
			std::vector<Eigen::Vector3d> pose_pts;
			std::vector<std::vector<float>> pose_attrs;
			for(int submap_id=0; submap_id<sub_maps_n-submap_gap_th; submap_id++) // 只将可能的子图位置加入搜索树
			{
				Eigen::Vector3d t_w_l = sub_maps[submap_id].sub_map_pose.block<3,1>(0,3);
				std::vector<float> pose_attr = {submap_id};
				pose_pts.emplace_back(t_w_l);
				pose_attrs.emplace_back(pose_attr);
			}
			debug_file<<"pose_pts.size(): "<<pose_pts.size()<<std::endl;
			if(pose_pts.size()<5) return pose_delta;
			thuni::Octree octree_submap_poses;
			octree_submap_poses.update_with_attr(pose_pts, pose_attrs);
			double nearest_submap_z_dist = -1.0;
            std::vector<std::vector<float>> points_near;
            std::vector<float> pointSearchSqDis_surf;
            Eigen::Vector3d current_trans = cur_submap_pose_lc.block<3,1>(0,3);
			octree_submap_poses.radiusNeighbors_eigen(current_trans, search_radius, points_near, pointSearchSqDis_surf);
            if(points_near.size()<1)
			{
				debug_file<<"points_near.size(): "<<points_near.size()<<std::endl;
				return pose_delta;
			}
            int points_near_n = points_near.size();
            std::vector<std::pair<int, float>> near_ids_dists_;
            for(int i=0; i<points_near_n; i++) near_ids_dists_.emplace_back(std::pair<int, float>{int(points_near[i][3]), pointSearchSqDis_surf[i]});
            std::sort(near_ids_dists_.begin(), near_ids_dists_.end(), [](std::pair<int, float> & a1, std::pair<int, float> & a2){return a1.first<a2.first;}); // 按照id大小排序
            debug_file<<"submap_id2: ";
            int last_submap_id2_ = near_ids_dists_[0].first;
            // 希望找到的是距离近、索引小，且z轴存在偏差，这里索引小是优先级比较高的，使得其他子图对齐到初始子图
            // 先通过id完成第一次筛选，选择id最小的序列
            std::vector<std::pair<int, float>> near_ids_dists_2;
            for(int i=0; i<points_near_n; i++)
            {
                int & submap_id2 = near_ids_dists_[i].first;
                if(cur_submap_id-submap_gap_th<submap_id2) break; // 子图相近的就舍弃，确保 submap_id 比 submap_id2 大至少 submap_gap_th
                if(submap_id2-last_submap_id2_>2) break; // 确保与id最小的那段子图建立匹配
                last_submap_id2_ = submap_id2;
                near_ids_dists_2.emplace_back(near_ids_dists_[i]);
                debug_file<<"("<<submap_id2<<", "<<near_ids_dists_[i].second<<"), ";
            }
            if(near_ids_dists_2.size()<1)
			{
				debug_file<<"near_ids_dists_2.size(): "<<near_ids_dists_2.size()<<std::endl;
				return pose_delta;
			}
            // 再通过距离筛选最近的
            std::sort(near_ids_dists_2.begin(), near_ids_dists_2.end(), [](std::pair<int, float> & a1, std::pair<int, float> & a2){return a1.second<a2.second;}); // 按照距离排序
            int & submap_id2 = near_ids_dists_2[0].first;
            Eigen::Vector3d delta_dir3 = sub_maps[submap_id2].sub_map_pose.block<3,1>(0,3)-cur_submap_pose_lc.block<3,1>(0,3);
            Eigen::Vector3d grd_normal_approx = cur_submap_pose_lc.block<3,1>(0,2); // 近似地面法线
            grd_normal_approx /= grd_normal_approx.norm()+1e-6;
            double z_dist_ = abs(delta_dir3.dot(grd_normal_approx));
            double z_dist_ratio_ = abs(delta_dir3.dot(grd_normal_approx))/(delta_dir3.norm()+1e-6);
            nearest_submap_id = submap_id2;
            nearest_submap_z_dist = z_dist_ratio_;
            debug_file<<"\n("<<submap_id2<<", "<<z_dist_<<", "<<sqrt(near_ids_dists_2[0].second)<<", "<<z_dist_ratio_<<")\n\n";
			if(nearest_submap_id<0 || nearest_submap_z_dist<0)
			{
				// debug_file<<"nearest_submap_id: "<<nearest_submap_id<<", nearest_submap_z_dist: "<<nearest_submap_z_dist<<std::endl;
				return pose_delta;
			}
		}
        // 4. 建立关联子图之间的平面约束==========同时优化多帧，然后建立当前帧与距离相近时间相远帧的关联关系，作为因子图的因子
        // 待优化的超级子图还是要小一点表较好，同时基础超级子图应该包围待优化超级子图，极端一点比如直接建立单个子图对多个子图的约束关系，补充回环检测缺失的问题
        // 根据z轴偏差进行非最大值抑制
        std::vector<int> ref_ids;
        std::unordered_map<int, int> submap_id2sel_plane_id;
        if(1)
		{
            // 再通过平面距离筛选
            // 建立与最近且最大平面点的关联
            Eigen::Vector3d last_normal_g = Eigen::Vector3d::Zero(); // 全局坐标系下的平面法线
            float last_d_g = 0; // 全局坐标系下的平面参数d
            Eigen::Vector3d last_center_g = Eigen::Vector3d::Zero();
            std::vector<int> submap_id2process = {cur_submap_id, nearest_submap_id};
            std::vector<Eigen::Matrix4d> pose_Ts = std::vector<Eigen::Matrix4d>{cur_submap_pose_lc, sub_maps[nearest_submap_id].sub_map_pose};
            bool need_opti = false;
			double cos_init_th_ = cos(30.0/180.0*3.1415926);
            for(int sii=0; sii<submap_id2process.size(); sii++)
            {
                const SubMap & submap_info = sub_maps[submap_id2process[sii]];
                Eigen::Matrix4d pose_T = pose_Ts[sii];
                std::vector<std::pair<int, double>> plane_id_areas_;
                int planes_n = submap_info.planes.size();
                // 先选择近似地面的平面
                for(int pid_=0; pid_<planes_n; pid_++)
                {
                    PlaneSimple* local_plane_ = submap_info.planes[pid_];
                    Eigen::Vector3d center_ = local_plane_->center;
                    if(local_plane_->center.norm()>10.0) continue; // 与中心点距离大于10
                    if(abs(local_plane_->normal[2])<cos_init_th_) continue; // 与z轴夹角大于30度
                    if(local_plane_->get_max_eigen_dist()>6.0*local_plane_->get_mid_eigen_dist()) continue; // 椭圆长短轴只差超过5
                    plane_id_areas_.emplace_back(std::pair<int, double>{pid_, local_plane_->get_max_eigen_dist()*local_plane_->get_mid_eigen_dist()});
                }
                if(plane_id_areas_.size()<1) break;
                std::sort(plane_id_areas_.begin(), plane_id_areas_.end(), [](std::pair<int, double> & a1, std::pair<int, double> & a2){return a1.second>a2.second;} );
                double min_dist_ = 100.0;
                int min_dist_pid_ = plane_id_areas_[0].first;
                // 从面积最大的前4个平面中选择距离中心点最近的一个
                for(int piai_=0; piai_<plane_id_areas_.size(); piai_++)
                {
                    int pid_ = plane_id_areas_[piai_].first;
                    PlaneSimple* local_plane_ = submap_info.planes[pid_];
                    double dist_ = local_plane_->center.norm();
                    if(dist_<min_dist_)
                    {
                        min_dist_ = dist_;
                        min_dist_pid_ = pid_;
                    }
                    // if(piai_>3)
					break;
                }
                submap_id2sel_plane_id[submap_id2process[sii]] = min_dist_pid_;
                PlaneSimple* local_plane_ = submap_info.planes[min_dist_pid_];
                Eigen::Vector3d normal_g =  pose_T.block<3,3>(0,0) * local_plane_->normal; // 全局坐标系下的平面法线
                float d_g = local_plane_->d - normal_g.transpose() * pose_T.block<3,1>(0,3); // 全局坐标系下的平面参数d
                Eigen::Vector3d center_g = pose_T.block<3,3>(0,0)*local_plane_->center + pose_T.block<3,1>(0,3);
                if(last_normal_g.norm()>0.5)
                {
                    float angle_ = acos(abs(normal_g.dot(last_normal_g)))/3.1415926*180.0;
                    float dist1_ = abs(normal_g.dot(last_center_g) + d_g);
                    float dist2_ = abs(last_normal_g.dot(center_g) + last_d_g);
                    std::cout<<"cur_submap_id: "<<cur_submap_id<<", nearest_submap_id: "<<nearest_submap_id<<", angle_: "<<angle_<<", dist1_: "<<dist1_<<", dist2_: "<<dist2_<<std::endl;
                    debug_file<<"cur_submap_id: "<<cur_submap_id<<", nearest_submap_id: "<<nearest_submap_id<<", angle_: "<<angle_<<", dist1_: "<<dist1_<<", dist2_: "<<dist2_<<std::endl;
                    // if(angle_<5 && dist1_>0.05 && dist2_>0.05) need_opti=true; // 角度好像不是太必要
                    if(dist1_>0.05 && dist2_>0.05) need_opti = true;
                }
                else
                {
                    last_normal_g = normal_g;
                    last_d_g = d_g;
                    last_center_g = center_g;
                }
            }
            // if(!need_opti)
			// {
			// 	debug_file<<"need_opti: "<<need_opti<<std::endl;
			// 	return;
			// }
            ref_ids.emplace_back(nearest_submap_id);
			if(submap_ids_w_lc.count(nearest_submap_id)<1)
			{
				for(int si = 1; si<=ref_submap_radius; si++)
				{
					int submap_id_ = nearest_submap_id+si;
					if(submap_id_<0 || submap_id_>=sub_maps_n) continue;
					if(submap_ids_w_lc.count(submap_id_)>0) break;
					ref_ids.emplace_back(submap_id_);
				}
				for(int si = 1; si<=ref_submap_radius; si++)
				{
					int submap_id_ = nearest_submap_id-si;
					if(submap_id_<0 || submap_id_>=sub_maps_n) continue;
					if(submap_ids_w_lc.count(submap_id_)>0) break;
					ref_ids.emplace_back(submap_id_);
				}
			}
        }
        std::unordered_map<int, Eigen::Matrix4d> submap_id2updated_poses; // 子图id到子图更新位姿的映射
        if(1)
        {
            std::string log_name = save_directory + "/optimize_seg_" +std::to_string(cur_submap_id)+ ".txt";
            // 将当前基础分段子图对应的需要优化的子图整理为超级子图，超级子图的相邻两子图之间的id不超过3
            std::vector<int> super_submap_ids; // 保存当前分段存在的超级子图数据，至少一个超级子图，可能存在多个
            // super_submap_ids.emplace_back(cur_submap_id); // 暂时只用一个子图，后面如果约束不足再考虑添加子图，只用一个子图感觉构建的平面约束有点少
			super_submap_ids.emplace_back(cur_submap_id);
			if(submap_ids_w_lc.count(cur_submap_id)<1)
			{
				for(int si = 1; si<=ref_submap_radius; si++)
				{
					int submap_id_ = cur_submap_id+si;
					if(submap_id_<0 || submap_id_>=sub_maps_n) continue;
					if(submap_ids_w_lc.count(submap_id_)>0) break;
					super_submap_ids.emplace_back(submap_id_);
				}
				for(int si = 1; si<=ref_submap_radius; si++)
				{
					int submap_id_ = cur_submap_id-si;
					if(submap_id_<0 || submap_id_>=sub_maps_n) continue;
					if(submap_ids_w_lc.count(submap_id_)>0) break;
					super_submap_ids.emplace_back(submap_id_);
				}
			}
			debug_file<<"submap_ids_w_lc "<<": ";
            for(int id_tmp: submap_ids_w_lc) debug_file<<id_tmp<<", ";
            debug_file<<std::endl;
			debug_file<<"super_submap_ids "<<": ";
            for(int id_tmp: super_submap_ids) debug_file<<id_tmp<<", ";
            debug_file<<std::endl;
            debug_file<<"ref_ids "<<": ";
            for(int id_tmp: ref_ids) debug_file<<id_tmp<<", ";
            debug_file<<std::endl;
            thuni::Octree * octree_basic_plane_center_ptr = new thuni::Octree();
            thuni::Octree * octree_basic_nonplane_pts_ptr = new thuni::Octree();
            octree_basic_nonplane_pts_ptr->set_min_extent(0.5);
            octree_basic_nonplane_pts_ptr->set_bucket_size(1);
            octree_basic_plane_center_ptr->set_min_extent(0.1);
            octree_basic_plane_center_ptr->set_bucket_size(1);
            std::vector<GlobalPlaneSimple*> submap_global_planes;
            std::unordered_map<int, int> submap_id2slide_win_id; // 子图id到划窗id的映射
            std::vector<Eigen::Matrix4d> slide_window_T_js; // 划窗中的帧位姿
            std::vector<int> slide_window_submap_ids; // 划窗中的帧
            std::set<int> slide_window_plane_ids_set; // 划窗中的平面
            std::vector<Eigen::Vector3d> plane_grid_centers_;
            std::vector<std::vector<float>> plane_attrs_;
            // 构建基础全局平面，只新建平面，不建立关联， 并将当前分段基础子图的非平面点加入非平面点搜索树中
            for(int rii_=0; rii_<ref_ids.size(); rii_++)
            {
                int submap_id_ = ref_ids[rii_];
                Eigen::Matrix<double, 4,4> T_j = sub_maps[submap_id_].sub_map_pose;
                Eigen::Matrix3d R_w_l= T_j.block<3,3>(0,0);
                Eigen::Vector3d t_w_l= T_j.block<3,1>(0,3);
                int plane_n_ = sub_maps[submap_id_].planes.size();
                for(int pi_=0; pi_<plane_n_; pi_++)
                {
                    PlaneSimple * local_plane_ = sub_maps[submap_id_].planes[pi_];
                    if(local_plane_->get_max_eigen_dist()<0.05) continue;
                    local_plane_->recover_for_insert();
                    Eigen::Matrix<double, 4,4> C_ij;
                    C_ij << local_plane_->covariance, local_plane_->center, local_plane_->center.transpose(), local_plane_->points_size;
                    local_plane_->cal_cov_and_center();
                    Eigen::Vector3d normal_g = R_w_l * local_plane_->normal; // 全局坐标系下的平面法线
                    float d_g = local_plane_->d - normal_g.transpose() * t_w_l; // 全局坐标系下的平面参数d
                    int plane_id_ = submap_global_planes.size();
                    GlobalPlaneSimple* global_plane_ = new GlobalPlaneSimple();
                    global_plane_->id = plane_id_;
                    global_plane_->normal = normal_g;
                    global_plane_->param_d = d_g;
                    global_plane_->C_ijs.push_back(C_ij);
                    global_plane_->frame_ids.push_back(submap_id_);
                    global_plane_->lg_normals.push_back(normal_g);
                    global_plane_->weight += local_plane_->get_weight();
                    submap_global_planes.push_back(global_plane_);
                    local_plane_->global_plane_id = plane_id_;
                    Eigen::Vector3d center_g = R_w_l*local_plane_->center + t_w_l;
                    std::vector<float> init_attr_={float(submap_id_), float(plane_id_)};
                    plane_grid_centers_.emplace_back(center_g);
                    plane_attrs_.emplace_back(init_attr_);
                }
                octree_basic_plane_center_ptr->update_with_attr(plane_grid_centers_, plane_attrs_);
                // 将当前分段基础子图的非平面点加入非平面点搜索树中
                SubMap & submap_info = sub_maps[submap_id_];
                Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
				pcl::PointCloud<PointType>::Ptr current_cloud_ = submap_info.sub_map_cloud;
				int pt_n =  current_cloud_->size();
                std::vector<Eigen::Vector3d> tree_pts_(pt_n);
				for(int pt_i=0; pt_i<pt_n; pt_i++)
				{
					Eigen::Vector3d pt_(current_cloud_->points[pt_i].x, current_cloud_->points[pt_i].y, current_cloud_->points[pt_i].z);
					Eigen::Vector3d ptE = delta_T.block<3,3>(0,0)*pt_ + delta_T.block<3,1>(0,3);
                    tree_pts_[pt_i] = ptE;
                }
                octree_basic_nonplane_pts_ptr->update_eigen(tree_pts_, true);
            }
            // 只建立关联不新增平面
            bool first_submap_flag = true;
            // int cur_submap_id = super_submap_ids[0]; // 以超级子图中的第一个子图作为参考，将其余子图全部投影到第一个子图
            std::vector<GlobalPlaneSimple*> slide_window_planes;
            double initial_residual = 0;
            double np_start_t, np_end_t, np_cost_t;
            double all_preppare_t=0, all_calcu_t=0;
            np_cost_t = 0;
            int iter_i = 0;
            int max_iter = 10;
            int win_size;
            Eigen::Matrix<double, 4,4> super_submap_T_j = cur_submap_pose_lc;
            Eigen::Matrix<double, 4,4> super_submap_T_j_inv = super_submap_T_j.inverse();
            submap_id2slide_win_id[cur_submap_id] = slide_window_T_js.size();
            slide_window_submap_ids.emplace_back(cur_submap_id);
            slide_window_T_js.emplace_back(super_submap_T_j);
            win_size = slide_window_T_js.size();
            bool need_rematch = false;
            double plane_merge_max_dist_ = 3.0, plane_merge_max_angle_=20.0;
			bool display = false;
			Eigen::Matrix< double, 6, 6 > G, H_T_H, I_STATE, Jh_inv, Jh, H_T_R_inv_H, P_inv;
			G.setZero(); H_T_H.setZero(); H_T_R_inv_H.setZero(); // H^T * R^{-1} * H
			I_STATE.setIdentity(); Jh_inv.setIdentity(); Jh.setIdentity(); P_inv.setIdentity();
			// ============1. 将其他子图的点转移到超级子图下，统一使用超级子图的位姿进行更新============
			std::vector<Eigen::Vector3d> nonplane_pts;
			for(int ssi_=0; ssi_<super_submap_ids.size(); ssi_++)
			{
				int submap_id2 = super_submap_ids[ssi_];
				const SubMap & submap_info = sub_maps[submap_id2];
				pcl::PointCloud<PointType>::Ptr current_cloud_ = submap_info.sub_map_cloud;
				int pt_n =  current_cloud_->size();
				std::vector<Eigen::Vector3d> nonplane_pts_(pt_n);
				for(int pt_i=0; pt_i<pt_n; pt_i++)
				{
					Eigen::Vector3d pt_(current_cloud_->points[pt_i].x, current_cloud_->points[pt_i].y, current_cloud_->points[pt_i].z);
					Eigen::Vector3d ptE = cur_submap_pose_inv.block<3,3>(0,0)*pt_ + cur_submap_pose_inv.block<3,1>(0,3);
					nonplane_pts_[pt_i] = ptE; // 初始全局坐标系==>子图局部坐标系==>更新后的全局坐标系==>到超级子图所在的局部坐标系
				}
				nonplane_pts.insert(nonplane_pts.end(), nonplane_pts_.begin(),nonplane_pts_.end());
			}
			int lp_n = nonplane_pts.size();
			std::vector<Eigen::Vector3d> gnormal_to_glp(lp_n);
			std::vector<double> gd_to_glp(lp_n);
			std::vector<double> pts_weights(lp_n);
			std::vector<Eigen::Vector3d> gl_pts_lio(lp_n);
			std::vector<float> dist_to_glp(lp_n); //全局坐标系下每个局部平面的每个点到对应全局平面的距离
			double deltaT = 0.0, deltaR = 0.0;
			int rematch_num = 0, valid_lidar_pts_n = 0;
			bool rematch_en = 0;
			Eigen::Matrix3d R_w_l0 = super_submap_T_j.block<3,3>(0,0);
			Eigen::Vector3d t_w_l0 = super_submap_T_j.block<3,1>(0,3);
			Eigen::Matrix3d R_w_l  = super_submap_T_j.block<3,3>(0,0);
			Eigen::Vector3d t_w_l  = super_submap_T_j.block<3,1>(0,3);
			// 增量差异带来的协方差变化 右乘==>左乘
			Eigen::Matrix<double, 6,6> W_r = Eigen::Matrix<double, 6,6>::Identity();
			W_r.block<3,3>(0,0) = R_w_l.transpose();
			W_r.block<3,3>(3,0) = -R_w_l.transpose()*vec_to_hat(t_w_l);
			W_r.block<3,3>(3,3) = R_w_l.transpose();
			Eigen::Matrix<double, 6,6> W_l = W_r.inverse();
			Eigen::Matrix<double, 6,6> cov1 = W_l*sub_maps[cur_submap_id].sub_map_pose_cov*W_l.transpose();
			P_inv = cov1.inverse(); // 这里的协方差是右乘扰动构建的，还需要转换到左乘。。。。
			double solve_time = 0, match_time=0.0, kd_tree_search_time=0.0;
			int num_match_pts = 4, NUM_MAX_ITERATIONS=3;
			double lidar_pt_cov =0.000015;
			double LASER_PLANE_COV =1e-5;
			double lidar_cov_p = 1.02;
			// ======2. 迭代优化======
			for ( int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ )
			{
				valid_lidar_pts_n = 0;
				// 计算每个点对应的平面法线和有符号距离
				std::vector<std::vector<float>> points_near;
				std::vector<float> pointSearchSqDis_surf;
				for(int pt_i=0; pt_i<lp_n; pt_i++)
				{
					gl_pts_lio[pt_i] = R_w_l*nonplane_pts[pt_i] + t_w_l;
					if(iterCount == 0 || rematch_en)
					{
						pts_weights[pt_i] = 0.0;
						octree_basic_nonplane_pts_ptr->knnNeighbors_eigen(gl_pts_lio[pt_i], num_match_pts, points_near, pointSearchSqDis_surf);
						float max_distance = pointSearchSqDis_surf[ num_match_pts - 1 ];
						if(max_distance > 4.0) continue; //  超过0.5就无效了
						// 平面拟合
						Eigen::Vector3d normal_fit;
						float pd;
						double pt_weight = lidar_cov_p;
						bool planeValid = true;
						if(1)
						{
							Eigen::MatrixXd A(num_match_pts, 3);
							Eigen::VectorXd B(num_match_pts);
							for (int j = 0; j < num_match_pts; ++j) {
								A(j, 0) = points_near[j][0];  // x
								A(j, 1) = points_near[j][1];  // y
								A(j, 2) = points_near[j][2];  // z
								B(j) = -1;  // 常数项
							}
							Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
							Eigen::VectorXd solution = svd.solve(B);
							normal_fit << solution(0), solution(1), solution(2);
							double norm_ = normal_fit.norm() + 1e-6;
							pd = 1 / norm_;  // 平面到原点的距离（假设平面方程的右侧常数项是 -D）
							normal_fit /= norm_;  // 归一化法向量
							pt_weight=0;
							// 检查拟合平面的好坏，距离超过0.1就无效
							for( int j = 0; j < num_match_pts; j++)
							{
								float dist = fabs( normal_fit[0] * points_near[ j ][0] + normal_fit[1] * points_near[ j ][1] + normal_fit[2] * points_near[ j ][2] + pd );
								if(dist > 0.2) // Raw 0.10
								{
									planeValid = false;
									break;
								}
								pt_weight+=dist;
							}
							pt_weight = cos(pt_weight/num_match_pts*3.1415926*5);
						}
						if(!planeValid) continue;
						gnormal_to_glp[pt_i] = normal_fit;
						gd_to_glp[pt_i] = pd;
						pts_weights[pt_i] = pt_weight;
					}
					if(pts_weights[pt_i]<1e-9) continue;
					double pd2 = gnormal_to_glp[pt_i].dot(gl_pts_lio[pt_i]) + gd_to_glp[pt_i];
					float s = 1 - 0.9 * fabs(pd2) / sqrt(nonplane_pts[pt_i].norm());
					if(s<=0.9)
					{
						pts_weights[pt_i] = 0.0;
						continue;
					}
					dist_to_glp[pt_i] = pd2;
					valid_lidar_pts_n++;
				}
				int find_plane_n = 0;
				Eigen::Matrix<double, 4,4> delta_T_u = slide_window_T_js[0]*cur_submap_pose_inv;
				// 构建待优化子图平面点与基础子图平面点之间的平面约束
				if(iterCount == 0)
				{
					for(int swpi=0; swpi<slide_window_planes.size(); swpi++) delete slide_window_planes[swpi];
					slide_window_planes.clear();
				}
				for(int ssi_=0; ssi_<super_submap_ids.size() && (iterCount == 0); ssi_++)
				{
					int submap_id2 = super_submap_ids[ssi_];
					const SubMap & submap_info = sub_maps[submap_id2];
					Eigen::Matrix<double, 4,4> T_j = delta_T_u*submap_info.sub_map_pose;
					Eigen::Matrix4d delta_T_super = cur_submap_pose_inv*submap_info.sub_map_pose;
					Eigen::Matrix3d R_w_l2= T_j.block<3,3>(0,0);
					Eigen::Vector3d t_w_l2= T_j.block<3,1>(0,3);
					std::vector<GlobalPlaneSimple*> plane_added_tmp;
					int plane_n_ = sub_maps[submap_id2].planes.size();
					for(int pi_=0; pi_<plane_n_; pi_++)
					{
						// std::cout<<"cubes_map_iter\n";
						PlaneSimple * local_plane_ = sub_maps[submap_id2].planes[pi_];
						local_plane_->recover_for_insert();
						Eigen::Matrix<double, 4,4> C_ij;
						C_ij << local_plane_->covariance, local_plane_->center, local_plane_->center.transpose(), local_plane_->points_size;
						C_ij = delta_T_super*C_ij*delta_T_super.transpose(); // 将平面从当前子图转移到超级子图
						local_plane_->cal_cov_and_center();
						Eigen::Vector3d normal_g = R_w_l2 * local_plane_->normal; // 全局坐标系下的平面法线
						float d_g = local_plane_->d - normal_g.transpose() * t_w_l2; // 全局坐标系下的平面参数d
						Eigen::Vector3d center_ = R_w_l2 * local_plane_->center + t_w_l2;
						double max_eigen_dist_ = local_plane_->get_max_eigen_dist();
						double mid_eigen_dist_ = local_plane_->get_mid_eigen_dist();
						Eigen::Vector3d x_normal_g = R_w_l2*local_plane_->x_normal;
						Eigen::Vector3d y_normal_g = R_w_l2*local_plane_->y_normal;
						Eigen::Vector4d cos_dist_best = Eigen::Vector4d::Zero();
						Eigen::Vector3d near_pt_best = Eigen::Vector3d::Zero();
						int gp_id_best = -1;
						std::vector<std::vector<float>> resultIndices;
						std::vector<float> distances;
						bool find_global_plane = false;
						octree_basic_plane_center_ptr->knnNeighbors_eigen(center_, 3, resultIndices, distances);
						std::vector<Eigen::Vector4d> candi_cos_dist;
						std::vector<Eigen::Vector3d> candi_nearest_pt;
						std::vector<int> candi_gpid;
						Eigen::Vector4d cos_weights = {1/15.0, 1.0, 0.5, 1/20.0};
						for(int re1=0; re1<distances.size(); re1++)
						{
							// std::cout<<"distances\n";
							int gp_id_ = int(resultIndices[re1][4]);
							if(gp_id_<0 || submap_global_planes[gp_id_]==nullptr) continue;
							GlobalPlaneSimple* near_global_plane_ = submap_global_planes[gp_id_];
							Eigen::Vector3d near_pt = {resultIndices[re1][0], resultIndices[re1][1], resultIndices[re1][2]};
							const Eigen::Vector3d & normal_s = near_global_plane_->normal;
							// 对于平面关联，最关键的几个参数是： 夹角、最近点在三个特征向量方向的投影距离（三个距离）
							double angle_ = acos(normal_s.transpose()*normal_g)/3.1415926*180; // 法线夹角
							Eigen::Vector3d nearest_pt_ = near_pt; // 最近点
							Eigen::Vector3d delta_ = nearest_pt_ - center_;
							double x_dist_ = abs(x_normal_g.dot(delta_))/max_eigen_dist_; // 最近点在当前平面上的投影在最大特征向量方向与最大特征根距离的比值
							double y_dist_ = abs(y_normal_g.dot(delta_))/mid_eigen_dist_; // 最近点在当前平面上的投影在中间特征向量方向与中间特征根距离的比值
							double z_dist_ = abs(normal_g.dot(delta_)); // 最近点到当前平面的距离
							Eigen::Vector4d cos_dist = {angle_, x_dist_, y_dist_, z_dist_};
							if(re1==0)
							{
								cos_dist_best = cos_dist;
								near_pt_best = near_pt;
								gp_id_best = gp_id_;
							}
							if(z_dist_>plane_merge_max_dist_ || angle_>plane_merge_max_angle_) continue;
							if(x_dist_>3 || y_dist_>6)
							{
								// log_file<<"x_dist_: "<<x_dist_<<", y_dist_: "<<y_dist_<<", angle_: "<<angle_<<", z_dist_: "<<z_dist_<<std::endl;
								continue; // 0.25=>0.10
							}
							double cost_ = cos(angle_/plane_merge_max_angle_*3.1415926/2.0)*cos(z_dist_/plane_merge_max_dist_*3.1415926/2.0);
							// 平面分类：融合平面、常规平面
							if(    (angle_<6 && x_dist_<3 && y_dist_<5) // 小夹角
								|| (cost_>0.75) // 高cost
								|| (cost_>0.5 && x_dist_<3 && y_dist_<3) // 中cost
								|| (angle_<10 && x_dist_<1.0 && y_dist_<1.0) // 只考虑覆盖
								)
							{
								candi_gpid.push_back(gp_id_);
								candi_nearest_pt.push_back(nearest_pt_);
								candi_cos_dist.push_back(cos_dist);
							}
							else
							{
								// log_file<<"x_dist_: "<<x_dist_<<", y_dist_: "<<y_dist_<<", angle_: "<<angle_<<", z_dist_: "<<z_dist_<<", cost_: "<<cost_<<std::endl;
							}
						}
						if(candi_gpid.size()>0)
						{
							gp_id_best = candi_gpid[0];
							cos_dist_best = candi_cos_dist[0];
							near_pt_best = candi_nearest_pt[0];
							double best_cost_ = cos_weights.dot(candi_cos_dist[0]);
							for(int candi_i=1; candi_i<candi_gpid.size(); candi_i++)
							{
								double cost_ = cos_weights.dot(candi_cos_dist[candi_i]);
								if(cost_<best_cost_)
								{
									gp_id_best = candi_gpid[candi_i];
									cos_dist_best = candi_cos_dist[candi_i];
									near_pt_best = candi_nearest_pt[candi_i];
									best_cost_ = cost_;
								}
							}
							find_global_plane = true;
						}
						if(find_global_plane)
						{
							find_plane_n += 1; // 对于一般平面
							GlobalPlaneSimple* global_plane_ = new GlobalPlaneSimple();
							*global_plane_ = *(submap_global_planes[gp_id_best]);
							global_plane_->C_ijs.push_back(C_ij);
							global_plane_->frame_ids.push_back(cur_submap_id);
							global_plane_->lg_normals.push_back(normal_g);
							global_plane_->weight += local_plane_->get_weight();
							plane_added_tmp.push_back(global_plane_);
						}
					}
					float find_ratio = (float)find_plane_n/plane_n_;
					if(find_ratio<0.15)
					{
						int plane_added_tmp_n = plane_added_tmp.size();
						for(int i=plane_added_tmp_n-1; i>-1; i--) delete plane_added_tmp[i];
					}
					else
					{
						int plane_added_tmp_n = plane_added_tmp.size();
						for(int i=plane_added_tmp_n-1; i>-1; i--)
						{
							if(!plane_added_tmp[i]) continue;
							slide_window_planes.push_back(plane_added_tmp[i]);
						}
					}
				}
				// 更新全局平面状态
				const int plane_n = slide_window_planes.size();
				std::vector<Eigen::Matrix<double, 1, 6>> J_pl_Ts(plane_n);
				std::vector<double> meas_vec_pl(plane_n, 0.0);
				std::vector<double> pl_weights(plane_n, 0.0);
				Eigen::MatrixXd H_l, D, I_e;
				Eigen::VectorXd J_l_T, dxi;
				int l=0;
				I_e.resize(6*win_size,6*win_size); H_l.resize(6*win_size,6*win_size); D.resize(6*win_size,6*win_size);
				J_l_T.resize(6*win_size); dxi.resize(6*win_size);
				H_l.setZero(); J_l_T.setZero();D.setZero(); dxi.setZero();
				I_e.setIdentity(); 
				double u = 0.01;
				double coe = 1.0;
				double residual = 0.0;
				for(int pi=0; pi<plane_n; pi++)
				{
					GlobalPlaneSimple* global_plane_ = slide_window_planes[pi];
					if(!global_plane_ || global_plane_->frame_size()<2) continue;
					global_plane_->C = Eigen::Matrix<double, 4,4>::Zero();
					const std::vector<int> & gp_all_frame_ids = global_plane_->frame_ids;
					const std::vector<Eigen::Matrix<double, 4,4>> & all_C_ijs = global_plane_->C_ijs;
					const int frame_ids_n = gp_all_frame_ids.size();
					bool re_calculate_flag = false;
					if(global_plane_->T_j_C_ijs.size() != frame_ids_n || global_plane_->T_j_C_ij_T_j_Ts.size() != frame_ids_n)
					{
						global_plane_->T_j_C_ijs.resize(frame_ids_n);
						global_plane_->T_j_C_ij_T_j_Ts.resize(frame_ids_n);
						for(int si=0; si<frame_ids_n; si++)
						{
							global_plane_->T_j_C_ijs[si] = Eigen::Matrix<double, 4,4>::Zero();
							global_plane_->T_j_C_ij_T_j_Ts[si] = Eigen::Matrix<double, 4,4>::Zero();
						}
						re_calculate_flag = true;
					}
					for(int si=0; si<frame_ids_n; si++)
					{
						const int & frame_id_ = gp_all_frame_ids[si];
						if(submap_id2slide_win_id.find(frame_id_) == submap_id2slide_win_id.end())
						{
							if(re_calculate_flag)
							{
								SubMap submap_ = sub_maps[frame_id_];
								Eigen::Matrix<double, 4,4> T_j = submap_.sub_map_pose;
								const Eigen::Matrix<double, 4,4> & C_ij = all_C_ijs[si];
								Eigen::Matrix<double, 4,4> T_j_C_ij, T_j_C_ij_T_j_T;
								T_j_C_ij = T_j*C_ij;
								T_j_C_ij_T_j_T = T_j_C_ij*T_j.transpose();
								global_plane_->T_j_C_ijs[si] = T_j_C_ij;
								global_plane_->T_j_C_ij_T_j_Ts[si] = T_j_C_ij_T_j_T;
							}
							global_plane_->C += global_plane_->T_j_C_ij_T_j_Ts[si]; // 作为边缘化因子，约束当前平面
						}
						else
						{
							int j = submap_id2slide_win_id.at(frame_id_); // 帧id转化为划窗中的id
							const Eigen::Matrix<double, 4,4> & C_ij = all_C_ijs[si];
							const Eigen::Matrix<double, 4,4> & T_j = slide_window_T_js[j];
							Eigen::Matrix<double, 4,4> T_j_C_ij, T_j_C_ij_T_j_T;
							T_j_C_ij = T_j*C_ij;
							T_j_C_ij_T_j_T = T_j_C_ij*T_j.transpose();
							global_plane_->T_j_C_ijs[si] = T_j_C_ij;
							global_plane_->T_j_C_ij_T_j_Ts[si] = T_j_C_ij_T_j_T;
							global_plane_->C += T_j_C_ij_T_j_T;
						}
					}
					Eigen::Vector3d xis[3];
					Eigen::Matrix<double, 6, 4> Vs[3];
					std::vector<Eigen::Matrix<double, 1, 6>> g_i_ml[3];
					double N_i = global_plane_->C(3, 3); // N_i
					global_plane_->eigen_solve();
					if(!global_plane_->has_eigen) continue;
					const Eigen::Vector3d & lambdas = global_plane_->lambdas;
					double tilt_weight = 1.0;
					if(1)
					{
						std::vector<double> angles_;
						double max_angle_ = 0;
						double mean_cos_ = 0;
						for(int li=0; li<global_plane_->lg_normals.size(); li++)
						{
							double cos_ = abs(global_plane_->xis[0].transpose()*global_plane_->lg_normals[li]);
							double angle_ = acos(cos_)/3.14159*180;
							mean_cos_ += abs(cos_);
							if(angle_>90) angle_=180-angle_;
							angles_.push_back(angle_);
							if(max_angle_<angle_) max_angle_=angle_;
						}
						if(max_angle_>30) tilt_weight = mean_cos_ / angles_.size();
					}
					if(lambdas[0]<0) tilt_weight=0.0;
					coe = global_plane_->weight*tilt_weight;
					pl_weights[pi] = coe;
					meas_vec_pl[pi] = lambdas[l];
					residual += coe*lambdas[l];
					for (int k = 0; k < 3; k++)
					{
						xis[k] = global_plane_->xis[k]; // 特征向量
						g_i_ml[k].resize(win_size);
						Vs[k].setZero();
						Vs[k].block<3, 3>(0, 0) = hat_d(-xis[k]);
						Vs[k].block<3, 1>(3, 3) = xis[k];
					}
					std::vector<Eigen::Matrix<double, 6, 1>> VlTCS_vT(win_size, Eigen::Matrix<double, 6, 1>::Zero()); // V_l*T_j*C_{ij}*S_v^T
					Eigen::Vector3d v_bar = global_plane_->C.block<3, 1>(0, 3)/N_i; // 1/N_i*v_i = 1/N_i*S_p*C_i*F
					// std::cout<<"Jacobian矩阵、hessian矩阵对角元素\n";
					// Jacobian矩阵、hessian矩阵对角元素
					for(int si=0; si<frame_ids_n; si++)
					{
						if(submap_id2slide_win_id.find(gp_all_frame_ids[si]) == submap_id2slide_win_id.end()) continue;
						int j = submap_id2slide_win_id[gp_all_frame_ids[si]]; // 帧id转化为划窗中的id
						Eigen::Matrix<double, 3, 4> temp = slide_window_T_js[j].block<3, 4>(0, 0); // S_p*T_j
						temp.block<3, 1>(0, 3) -= v_bar; // S_p*(T_j-1/N_i*C_i*F)
						// temp = S_p*(T_j-1/N_i*C_i*F)
						Eigen::Matrix<double, 4, 3> TC_TCFSp = global_plane_->T_j_C_ijs[si] * temp.transpose(); // T_j * C_ij * (S_p*(T_j-1/N_i*C*F))^T
						for (int m = 0; m < 3; m++)
						{
							Eigen::Matrix<double, 6, 1> g1, g2;
							g1 = Vs[m] * TC_TCFSp * xis[l]; // V_k* T_j * C_j * (S_p*(T_j-1/N_i*C*F))^T * \xi_l  见式 34
							g2 = Vs[l] * TC_TCFSp * xis[m];
							g_i_ml[m][j] = (g1 + g2).transpose() / N_i;
						}
						J_l_T.block<6,1>(6*j,0) += coe * g_i_ml[l][j].transpose();
						if(cur_submap_id==gp_all_frame_ids[si]) J_pl_Ts[pi] = g_i_ml[l][j];
						VlTCS_vT[j] = (Vs[l] * global_plane_->T_j_C_ijs[si]).block<6, 1>(0, 3); // V_l*T_j*C_{ij}*S_v^T
						Eigen::Matrix<double, 6, 6> Ha(-2.0 / N_i / N_i * VlTCS_vT[j] * VlTCS_vT[j].transpose());
						// Q^{ijj}_{ll} = -2/(N_i*N_i) V_l * T_j *C_ij * F * C_ij * T_j^T * V_l^T
						Eigen::Matrix3d Ell = 1.0 / N_i * hat_d(TC_TCFSp.block<3, 3>(0, 0) * xis[l]) * hat_d(xis[l]);
						// 1/N_i * (S_p * T_j * C_ij * (S_p*(T_j-1/N_i*C_i*F))^T*S_p *\xi_l)^\land (\xi_l)^\land
						Ha.block<3, 3>(0, 0) += Ell + Ell.transpose(); // K^{ij}_{ll}
						for (int m = 0; m < 3; m++)
							if (m != l)
								Ha += 2.0 / (lambdas[l] - lambdas[m]) * g_i_ml[m][j].transpose() * g_i_ml[m][j];
								// (31) \sum_{m=1,m\ne l}^{3} 2/(\lambda_l-\lambda_m) * g_{ml}^T * g_{ml}
						H_l.block<6, 6>(6 * j, 6 * j) += coe * Ha;
						Eigen::Matrix<double, 6, 6> Hb = Vs[l] * global_plane_->T_j_C_ij_T_j_Ts[si] * Vs[l].transpose();
						H_l.block<6, 6>(6 * j, 6 * j) += 2.0 / N_i * coe * Hb;
					}
					// std::cout<<"hessian矩阵非对角元素\n";
					// hessian矩阵非对角元素
					for(int si=0; si<frame_ids_n-1; si++)
					{
						if(submap_id2slide_win_id.find(gp_all_frame_ids[si]) == submap_id2slide_win_id.end()) continue;
						int i = submap_id2slide_win_id[gp_all_frame_ids[si]]; // 帧id转化为划窗中的id
						for(int sj=si+1; sj<frame_ids_n; sj++)
						{
							if(submap_id2slide_win_id.find(gp_all_frame_ids[sj]) == submap_id2slide_win_id.end()) continue;
							int j = submap_id2slide_win_id[gp_all_frame_ids[sj]]; // 帧id转化为划窗中的id
							Eigen::Matrix<double, 6, 6> Ha = -2.0 / N_i / N_i * VlTCS_vT[i] * VlTCS_vT[j].transpose();
							for (int m = 0; m < 3; m++)
								if (m != l)
									Ha += 2.0 / (lambdas[l] - lambdas[m]) * g_i_ml[m][i].transpose() * g_i_ml[m][j];
							H_l.block<6, 6>(6 * i, 6 * j) += coe * Ha;
						}
					}
					// std::cout<<"hessian矩阵非对角元素2\n";
				}
				for (int i = 0; i < win_size; i++)
				{
					for (int j = 0; j < i; j++)
						H_l.block<6, 6>(6 * i, 6 * j) = H_l.block<6, 6>(6 * j, 6 * i).transpose();
					Eigen::Matrix<double, 6,6> cov_inv = sub_maps[slide_window_submap_ids[i]].sub_map_pose_cov.inverse();
					// H_l.block<6, 6>(6 * i, 6 * i) += cov_inv * 0.01;
					// std::cout<<"i: "<<i<<", H_l + cov_inv: "<<H_l.block<6, 6>(6 * i, 6 * i).diagonal().transpose()<<std::endl;
				}
				// if(debug) std::cout<<"计算雅克比矩阵和测量向量: "<<iterCount<< std::endl;
				// if(process_debug) std::cout<<"EKF_solve_start"<<", valid_lidar_pts_n: "<<valid_lidar_pts_n<< std::endl;
				// 3. 计算雅克比矩阵和测量向量
				Eigen::Matrix< double, 6, 1 > vec; // \check{x}_{op,k} - \check{x}_{k}
				Eigen::Matrix3d rotd(R_w_l * R_w_l0.transpose());
				vec.block<3, 1>(0, 0) = SO3_LOG(rotd);
				vec.block<3, 1>(3, 0) = t_w_l - rotd*t_w_l0;
				Eigen::Vector3d delta_theta =  vec.block<3,1>(0,0);
				Eigen::Matrix3d J_l_vec_inv = inverse_left_jacobian_of_rotation_matrix(delta_theta); // ==> 对应公式中的 J
				Eigen::Matrix3d t_crossmat;
				Eigen::Vector3d delta_t_tmp = vec.block<3, 1>(3, 0);
				t_crossmat << SKEW_SYM_MATRIX( delta_t_tmp );
				Jh.block<3,3>(0,0) = J_l_vec_inv;
				Jh.block<3,3>(3,0) = -t_crossmat;
				Jh_inv = Jh.inverse();
				Eigen::Matrix< double, 6, 1 > solution;
				solution.setZero();
				bool pure_po_flag = true, pure_pl_flag = true, po_pl_flag = true;
				if(pure_po_flag && 0) // 纯点面约束
				{
					Eigen::MatrixXd Hsub( valid_lidar_pts_n, 6 ); // 除了前6维度都是0，这里使用缩减版
					Eigen::VectorXd meas_vec( valid_lidar_pts_n );
					Eigen::MatrixXd H_T_R_inv( 6, valid_lidar_pts_n ); // H^T* R^{-1}
					Hsub.setZero(); H_T_R_inv.setZero(); meas_vec.setZero();
					int Num_i = 0;
					double po_total_meas_ = 0.0;
					double po_total_weight_ = 0.0, max_weight_ = 0.0, min_weight = 1e5;
					for(int pt_i=0; pt_i<lp_n; pt_i++)
					{
						if(pts_weights[pt_i]<1e-9) continue;
						if(Num_i>=valid_lidar_pts_n)
						{
							break;
						}
						const Eigen::Vector3d & normal_ = gnormal_to_glp[pt_i];
						Eigen::Matrix3d point_crossmat;
						point_crossmat << SKEW_SYM_MATRIX( gl_pts_lio[pt_i] );
						//* 转置，而point_crossmat没转置，就是添加负号！！
						Eigen::Vector3d A = point_crossmat * normal_;
						Hsub.block<1, 6>(Num_i, 0)  << A[0], A[1], A[2], normal_[0], normal_[1], normal_[2];
						Hsub.block<1, 6>(Num_i, 0) = Hsub.block<1, 6>(Num_i, 0)*Jh_inv; // H*J = H*Jh_inv
						H_T_R_inv.block<6, 1>(0, Num_i) = Hsub.block<1, 6>(Num_i, 0).transpose()/lidar_pt_cov*pts_weights[pt_i];
						meas_vec( Num_i ) = dist_to_glp[pt_i];
						po_total_meas_ += dist_to_glp[pt_i];
						po_total_weight_ += pts_weights[pt_i];
						if(pts_weights[pt_i]>max_weight_) max_weight_ = pts_weights[pt_i];
						if(pts_weights[pt_i]<min_weight) min_weight = pts_weights[pt_i];
						Num_i++;
					}
					po_total_meas_ /= Num_i+1;
					po_total_weight_ /= Num_i+1;
					Eigen::MatrixXd K( 6, valid_lidar_pts_n );
					H_T_R_inv_H.block<6,6>(0,0) = H_T_R_inv * Hsub;
					K = ( P_inv + H_T_R_inv_H ).inverse().block< 6, 6 >( 0, 0 ) * H_T_R_inv;
					solution = K * ( -meas_vec + Hsub * vec.block< 6, 1 >( 0, 0 ) ); // 见“基于EKF的雷达位姿更新” 式子 3.81
				}
				if(pure_pl_flag && 0) // 纯面面约束
				{
					Eigen::MatrixXd Hsub( plane_n, 6 ); // 除了前6维度都是0，这里使用缩减版
					Eigen::VectorXd meas_vec( plane_n );
					Eigen::VectorXd meas_vec_T_R_inv( plane_n );
					Eigen::MatrixXd H_T_R_inv( 6, plane_n ); // H^T* R^{-1}
					Hsub.setZero(); H_T_R_inv.setZero(); meas_vec.setZero();
					int Num_i = 0;
					double pl_total_meas_ = 0.0;
					double pl_total_weight_ = 0.0, max_weight_ = 0.0, min_weight = 1e5;
					for(int pl_i=0; pl_i<plane_n; pl_i++)
					{
						if(pl_weights[pl_i]<1e-9) continue;
						Hsub.block<1, 6>(Num_i, 0) = J_pl_Ts[pl_i]*Jh_inv; // H*J = H*Jh_inv
						H_T_R_inv.block<6, 1>(0, Num_i) = Hsub.block<1, 6>(Num_i, 0).transpose()/LASER_PLANE_COV*pl_weights[pl_i];
						meas_vec( Num_i ) = meas_vec_pl[pl_i];
						meas_vec_T_R_inv(Num_i) = meas_vec_pl[pl_i]/LASER_PLANE_COV*pl_weights[pl_i];
						pl_total_meas_ += meas_vec_pl[pl_i];
						pl_total_weight_ += pl_weights[pl_i];
						if(pl_weights[pl_i]>max_weight_) max_weight_ = pl_weights[pl_i];
						if(pl_weights[pl_i]<min_weight) min_weight = pl_weights[pl_i];
						Num_i++;
					}
					pl_total_meas_ /= Num_i+1;
					pl_total_weight_ /= Num_i+1;
					Eigen::MatrixXd K( 6, plane_n );
					H_T_R_inv_H.block<6,6>(0,0) = H_T_R_inv * Hsub;
					K = ( P_inv + H_T_R_inv_H ).inverse().block< 6, 6 >( 0, 0 ) * H_T_R_inv;
					solution = K * ( -meas_vec + Hsub * vec.block< 6, 1 >( 0, 0 ) ); // 见“基于EKF的雷达位姿更新” 式子 3.81
					// if(iterCount==0)
					{
						Eigen::Matrix< double, 6, 1 > vec_; // \check{x}_{op,k} - \check{x}_{k}
						Eigen::Matrix3d R_w_l_ = Exp(solution(0), solution(1), solution(2))*R_w_l0;
						Eigen::Vector3d t_w_l_ = Exp(solution(0), solution(1), solution(2))*t_w_l0 + solution.block<3, 1>(3, 0);
						Eigen::Matrix3d rotd_(R_w_l_ * R_w_l.transpose());
						vec_.block<3, 1>(0, 0) = SO3_LOG(rotd_);
						vec_.block<3, 1>(3, 0) = t_w_l_ - rotd_*t_w_l;
						double residual_ = (meas_vec_T_R_inv.transpose())*(meas_vec);
						double a = vec_.transpose()*(H_T_R_inv_H)*vec_;
						double b = 2*(meas_vec_T_R_inv.transpose())*Hsub*vec_;
						double c = residual_;
						double residual_approx = a+b+c;
						double scale_ = 1.0;
						if(abs(residual_approx)>c)
						{
							scale_=0.0;
							residual_approx=c;
						}
						if(residual_approx>0)
						{
							double tmp_1 = -b/(2*a);
							// if(tmp_1>0 && tmp_1<2.0) //* 不能取消这个限制，距离线性化点稍远误差就会很大！！！
							{
								double residual_temp = c + b*tmp_1 + a*tmp_1*tmp_1;
								// std::cout<<"-b/(2*a): "<<tmp_1<<", residual_temp: "<<residual_temp<<", residual_approx: "<<residual_approx<<std::endl;
								if(residual_temp<residual_approx)
								{
									residual_approx = residual_temp;
									scale_ = tmp_1;
								}
							}
						}
						if(residual_approx<0)
						{
							double tmp_1 = -b/(2*a);
							double tmp_2 = sqrt(b*b-4*a*c)/(2*a);
							double t1 = tmp_1-tmp_2, t2 = tmp_1+tmp_2;
							if(t1>0 && t1<1)
							{
								double residual_temp = a*t1*t1 + b*t1 + c;
								if(abs(residual_temp)<abs(residual_approx))
								{
									residual_approx = residual_temp;
									scale_ = t1;
								}
							}
							if(t2>0 && t2<1)
							{
								double residual_temp = a*t2*t2 + b*t2 + c;
								if(abs(residual_temp)<abs(residual_approx))
								{
									residual_approx = residual_temp;
									scale_ = t2;
								}
							}
						}
					}
				}
				if(po_pl_flag && 1) // 纯点面约束
				{
					Eigen::MatrixXd Hsub( valid_lidar_pts_n+plane_n, 6 ); // 除了前6维度都是0，这里使用缩减版
					Eigen::VectorXd meas_vec( valid_lidar_pts_n+plane_n );
					Eigen::VectorXd meas_vec_T_R_inv( valid_lidar_pts_n+plane_n );
					Eigen::MatrixXd H_T_R_inv( 6, valid_lidar_pts_n+plane_n ); // H^T* R^{-1}
					Hsub.setZero(); H_T_R_inv.setZero(); meas_vec.setZero();
					int Num_i = 0;
					double po_total_meas_ = 0.0;
					double po_total_weight_ = 0.0, max_weight_ = 0.0, min_weight = 1e5;
					for(int pt_i=0; pt_i<lp_n; pt_i++)
					{
						if(pts_weights[pt_i]<1e-9) continue;
						if(Num_i>=valid_lidar_pts_n)
						{
							break;
						}
						const Eigen::Vector3d & normal_ = gnormal_to_glp[pt_i];
						Eigen::Matrix3d point_crossmat;
						point_crossmat << SKEW_SYM_MATRIX( gl_pts_lio[pt_i] );
						//* 转置，而point_crossmat没转置，就是添加负号！！
						Eigen::Vector3d A = point_crossmat * normal_;
						Hsub.block<1, 6>(Num_i, 0)  << A[0], A[1], A[2], normal_[0], normal_[1], normal_[2];
						Hsub.block<1, 6>(Num_i, 0) = Hsub.block<1, 6>(Num_i, 0)*Jh_inv; // H*J = H*Jh_inv
						H_T_R_inv.block<6, 1>(0, Num_i) = Hsub.block<1, 6>(Num_i, 0).transpose()/lidar_pt_cov*pts_weights[pt_i];
						meas_vec( Num_i ) = dist_to_glp[pt_i];
						meas_vec_T_R_inv(Num_i) = dist_to_glp[pt_i]/lidar_pt_cov*pts_weights[pt_i];
						po_total_meas_ += dist_to_glp[pt_i];
						po_total_weight_ += pts_weights[pt_i];
						if(pts_weights[pt_i]>max_weight_) max_weight_ = pts_weights[pt_i];
						if(pts_weights[pt_i]<min_weight) min_weight = pts_weights[pt_i];
						Num_i++;
					}
					po_total_meas_ /= Num_i+1;
					po_total_weight_ /= Num_i+1;
					Num_i = valid_lidar_pts_n;
					double pl_total_meas_ = 0.0;
					double pl_total_weight_ = 0.0;
					for(int pl_i=0; pl_i<plane_n; pl_i++)
					{
						if(pl_weights[pl_i]<1e-9) continue;
						Hsub.block<1, 6>(Num_i, 0) = J_pl_Ts[pl_i]*Jh_inv; // H*J = H*Jh_inv
						H_T_R_inv.block<6, 1>(0, Num_i) = Hsub.block<1, 6>(Num_i, 0).transpose()/LASER_PLANE_COV*pl_weights[pl_i];
						meas_vec( Num_i ) = meas_vec_pl[pl_i];
						meas_vec_T_R_inv(Num_i) = meas_vec_pl[pl_i]/LASER_PLANE_COV*pl_weights[pl_i];
						pl_total_meas_ += meas_vec_pl[pl_i];
						pl_total_weight_ += pl_weights[pl_i];
						Num_i++;
					}
					Num_i -= valid_lidar_pts_n;
					pl_total_meas_ /= Num_i+1;
					pl_total_weight_ /= Num_i+1;
					Eigen::MatrixXd K( 6, valid_lidar_pts_n+plane_n );
					H_T_R_inv_H.block<6,6>(0,0) = H_T_R_inv * Hsub;
					K = ( P_inv + H_T_R_inv_H ).inverse().block< 6, 6 >( 0, 0 ) * H_T_R_inv;
					solution = K * ( -meas_vec + Hsub * vec.block< 6, 1 >( 0, 0 ) ); // 见“基于EKF的雷达位姿更新” 式子 3.81
					// if(iterCount==0)
					{
						Eigen::Matrix< double, 6, 1 > vec_; // \check{x}_{op,k} - \check{x}_{k}
						Eigen::Matrix3d R_w_l_ = Exp(solution(0), solution(1), solution(2))*R_w_l0;
						Eigen::Vector3d t_w_l_ = Exp(solution(0), solution(1), solution(2))*t_w_l0 + solution.block<3, 1>(3, 0);
						Eigen::Matrix3d rotd_(R_w_l_ * R_w_l.transpose());
						vec_.block<3, 1>(0, 0) = SO3_LOG(rotd_);
						vec_.block<3, 1>(3, 0) = t_w_l_ - rotd_*t_w_l;
						double residual_ = (meas_vec_T_R_inv.transpose())*(meas_vec);
						double a = vec_.transpose()*(H_T_R_inv_H)*vec_;
						double b = 2*(meas_vec_T_R_inv.transpose())*Hsub*vec_;
						double c = residual_;
						double residual_approx = a+b+c;
						double scale_ = 1.0;
						if(abs(residual_approx)>c)
						{
							scale_=0.0;
							residual_approx=c;
						}
						if(residual_approx>0)
						{
							double tmp_1 = -b/(2*a);
							// if(tmp_1>0 && tmp_1<2.0) //* 不能取消这个限制，距离线性化点稍远误差就会很大！！！
							{
								double residual_temp = c + b*tmp_1 + a*tmp_1*tmp_1;
								// std::cout<<"-b/(2*a): "<<tmp_1<<", residual_temp: "<<residual_temp<<", residual_approx: "<<residual_approx<<std::endl;
								if(residual_temp<residual_approx)
								{
									residual_approx = residual_temp;
									scale_ = tmp_1;
								}
							}
						}
						if(residual_approx<0)
						{
							double tmp_1 = -b/(2*a);
							double tmp_2 = sqrt(b*b-4*a*c)/(2*a);
							double t1 = tmp_1-tmp_2, t2 = tmp_1+tmp_2;
							if(t1>0 && t1<1)
							{
								double residual_temp = a*t1*t1 + b*t1 + c;
								if(abs(residual_temp)<abs(residual_approx))
								{
									residual_approx = residual_temp;
									scale_ = t1;
								}
							}
							if(t2>0 && t2<1)
							{
								double residual_temp = a*t2*t2 + b*t2 + c;
								if(abs(residual_temp)<abs(residual_approx))
								{
									residual_approx = residual_temp;
									scale_ = t2;
								}
							}
						}
						solution *= scale_;
					}
				}
				R_w_l = Exp(solution(0), solution(1), solution(2))*R_w_l0;
				t_w_l = Exp(solution(0), solution(1), solution(2))*t_w_l0 + solution.block<3, 1>(3, 0);
				slide_window_T_js[0].block<3,3>(0,0) = R_w_l;
				slide_window_T_js[0].block<3,1>(0,3) = t_w_l;
				Eigen::Vector3d rot_add = solution.block< 3, 1 >( 0, 0 );
				Eigen::Vector3d t_add = solution.block< 3, 1 >( 3, 0 );
				bool flg_EKF_converged = false;
				if( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) ) flg_EKF_converged = true;
				deltaR = rot_add.norm() * 57.3;
				deltaT = t_add.norm() * 100;
				rematch_en = false;
				// if(flg_EKF_converged || ( ( rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) )
				{
					rematch_en = true;
					rematch_num++;
				}
				// if(rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) break; // Fast lio ori version.
				if(iterCount == NUM_MAX_ITERATIONS - 1 ) break; // Fast lio ori version.
				// if(debug) std::cout<<"EKF_solve_end"<< std::endl;
			}
            int pp_residual_iter_i = 0;
            // std::cout<<"all_prepare_t: "<<all_preppare_t<<", all_calcu_t: "<<all_calcu_t<<", all time: "<<all_preppare_t+all_calcu_t<<std::endl;
            Eigen::Matrix<double, 4,4> super_submap_T_j_u = slide_window_T_js[0];
            Eigen::Matrix<double, 4,4> delta_T_u = super_submap_T_j_u*cur_submap_pose_inv;
            for(int i=0; i<submap_global_planes.size(); i++) delete submap_global_planes[i];
            submap_global_planes.clear();
            delete octree_basic_plane_center_ptr;
			return delta_T_u;
        }
	}

	void loop_run()
	{
		ros::Rate loop(50000);
		int start_key_frame_id;
		while (ros::ok())
		{
			ros::spinOnce();
			bool save_submaps_path_ = false;
			nh.getParam("/save_submaps_path", save_submaps_path_);    // you can do [$rosparam set /save_map true] on terminal to manually save map
			if(save_submaps_path_)
			{
				nh.setParam("/save_submaps_path", false);
				save_submaps_path_ = false;
				debug_file<<"save_submaps_path"<<std::endl;
				correctPosesAndPubAndSaveTxt(true);
			}
			bool save_one_time = false;
			nh.getParam("/save_map", save_one_time);    // you can do [$rosparam set /save_map true] on terminal to manually save map
			if(save_one_time)
			{
				nh.setParam("/save_map", false);
				save_one_time = false;
				debug_file<<"display"<<std::endl;
				// display();
			}
			if (lidar_buf.empty() || odom_buf.empty() || use_key_frame_planes && key_frame_planes_buffer.empty()) continue;
			// debug_file<<"lidar_buf.size(): "<<lidar_buf.size()<<", odom_buf.size(): "<<odom_buf.size()<<std::endl;
			if(!decouple_front && wait_front_end_done && correction_time>0 && time_buf.front().toSec() >= correction_time) // 前端回环更新完毕，结束等待状态
			{
				wait_front_end_done = false;
				correction_time = -1;
			}
			mtx_buffer.lock();
			while (!odom_buf.empty() && odom_buf.front()->header.stamp.toSec() < time_buf.front().toSec()) odom_buf.pop_front();
			if(odom_buf.front()->header.stamp.toSec() != time_buf.front().toSec()) continue;
			current_pose = Eigen::Matrix4d::Identity();
			auto cur_pose = odom_buf.front()->pose;
			int key_frame_id = int(odom_buf.front()->twist.covariance[0]);
			ros::Time cur_time = time_buf.front();
			pcl::PointCloud<PointType>::Ptr cloud_body_ptr(new pcl::PointCloud<PointType>);
			*cloud_body_ptr = *(lidar_buf.front());
			std_msgs::Float64MultiArray::ConstPtr key_frame_planes_msg = nullptr;
			if(use_key_frame_planes)
			{
				key_frame_planes_msg = key_frame_planes_buffer.front();
				key_frame_planes_buffer.pop_front();
			}
			lidar_buf.pop_front(); time_buf.pop_front(); odom_buf.pop_front();
			mtx_buffer.unlock();
			Eigen::Quaterniond current_quaternion(cur_pose.pose.orientation.w, cur_pose.pose.orientation.x, cur_pose.pose.orientation.y, cur_pose.pose.orientation.z);
			current_pose.block<3,3>(0,0) = current_quaternion.toRotationMatrix();
			current_pose.block<3,1>(0,3) = Eigen::Vector3d(cur_pose.pose.position.x, cur_pose.pose.position.y, cur_pose.pose.position.z);
			if(!decouple_front && wait_front_end_done) current_pose = lc_delta_pose*current_pose; // 回环优化之后的等待时间进来的帧位姿都需要加个偏置
			// std::cout<<"key_frame_id: "<<key_frame_id<<", cur_p: "<<current_pose.block<3,1>(0,3).transpose()<<", wait_front_end_done: "<<wait_front_end_done<<std::endl;
			if(1) // 保存关键帧位姿并添加相邻两帧之间的位姿约束
			{
				KeyFramePose kf_pose;
				kf_pose.key_frame_id = key_frame_id;
				kf_pose.key_frame_pose = current_pose;
				kf_pose.key_frame_pose_opt = current_pose;
				for(int i=0; i<6; i++)
					for(int j=0; j<6; j++)
						kf_pose.key_frame_pose_cov(i,j) = cur_pose.covariance[i*6+j];
				kf_pose.pose_opt_set = false;
				kf_pose.key_frame_time = cur_time;
				if(key_frame_poses.size()==0)
				{
					last_pose = current_pose;
					lc_delta_pose = Eigen::Matrix4d::Identity();
				}
				key_frame_poses.push_back(kf_pose);
				if(use_key_frame_planes) merge_plane_for_submap(kf_pose, key_frame_planes_msg, false);
				if(key_frame_poses.size()==1)
				{
					gtsam::Pose3 pose_prior = trans2gtsamPose(current_pose);
					gts_graph_2.add(gtsam::PriorFactor<gtsam::Pose3>(key_frame_id, pose_prior, pose_start_noise_));
					if (val_inserted2.find(key_frame_id) == val_inserted2.end() || !val_inserted2[key_frame_id])
					{
						gts_init_vals_2.insert(key_frame_id, pose_prior);
						val_inserted2[key_frame_id] = true;
					}
				}
				else
				{
					const KeyFramePose & last_kf_pose = key_frame_poses[key_frame_poses.size()-2];
					const gtsam::Pose3 & pose_curr = trans2gtsamPose(current_pose);
					Eigen::Matrix4d last_kf_T = last_kf_pose.key_frame_pose;
					if(!decouple_front && wait_front_end_done && last_kf_pose.pose_opt_set) last_kf_T = last_kf_pose.key_frame_pose_opt; // 针对回环优化之后的第一帧
					const gtsam::Pose3 & pose_last = trans2gtsamPose(last_kf_T);
					gts_graph_2.add(gtsam::BetweenFactor<gtsam::Pose3>(last_kf_pose.key_frame_id, key_frame_id, pose_last.between(pose_curr), pose_noise_));
					if (val_inserted2.find(key_frame_id) == val_inserted2.end() || !val_inserted2[key_frame_id])
					{
						gts_init_vals_2.insert(key_frame_id, pose_curr);
						val_inserted2[key_frame_id] = true;
					}
				}
			}
			if (frame_number == 0)
			{
				current_sub_map->clear();
				start_key_frame_id = key_frame_id;
			}
			bool is_build_descriptor = false;
			// 转换到世界坐标系
			pcl::PointCloud<PointType>::Ptr cloud_ptr(new pcl::PointCloud<PointType>);
			for (int i = 0; i < cloud_body_ptr->points.size(); i++)
			{
				PointType pt = cloud_body_ptr->points[i];
				Eigen::Vector3d pte(pt.x, pt.y, pt.z);
				pte = current_pose.block<3, 3>(0, 0)*pte + current_pose.block<3, 1>(0, 3);
				pt.x = pte[0];
				pt.y = pte[1];
				pt.z = pte[2];
				cloud_ptr->push_back(pt);
			}
			// debug_file<<"down_sampling_voxel" << ", position_inc.norm(): "<<position_inc.norm()<<std::endl;
			down_sampling_voxel(*cloud_ptr, *current_sub_map, ds_size); // 下采样并保存到 current_sub_map 中 0.5m
			Eigen::Quaterniond last_quaternion(last_pose.block<3, 3>(0, 0));
			double rotation_inc = current_quaternion.angularDistance(last_quaternion);
			Eigen::Vector3d position_inc = current_pose.block<3,1>(0,3) - last_pose.block<3,1>(0,3);
			if (position_inc.norm() > 5)
			{
				last_pose = current_pose;
				continue;
			}
			if (position_inc.norm() < position_threshold && rotation_inc < rotation_threshold) continue; // 0.2  DEG2RAD(5)
			if (frame_number < sub_frame_num - 1) frame_number++; // 20
			else
			{
				debug_file << "lidar_buf size: " << lidar_buf.size() << " waiting to process" << std::endl;
				frame_number = 0;
				is_build_descriptor = true;
			}
			last_pose = current_pose;
			// debug_file <<"frame_number: "<<frame_number<<", cloud size:" << current_sub_map->size() <<" is_build_descriptor: "<< is_build_descriptor << std::endl<< std::endl;
			if(!is_build_descriptor) continue;
			if(1) // 发布子图信息
			{
				nav_msgs::OdometryPtr odom_pub(new nav_msgs::Odometry());
				odom_pub->header.frame_id = "/world";
				odom_pub->header.stamp = cur_time;
				odom_pub->pose.pose.position = cur_pose.pose.position;
				odom_pub->pose.pose.orientation = cur_pose.pose.orientation;
				odom_pub->pose.covariance = cur_pose.covariance;
				odom_pub->twist.covariance[0] = sub_map_id;
				odom_pub->twist.covariance[1] = key_frame_id;
				odom_pub->twist.covariance[2] = start_key_frame_id;
				submap_info_pub.publish(odom_pub);
			}
			std::cout << std::endl;
			// std::cout << "loop_detection sub_map_id:" << sub_map_id << ", cloud size:" << current_sub_map->size() << std::endl;
			debug_file << std::endl;
			debug_file << "loop_detection sub_map_id:" << sub_map_id << ", cloud size:" << current_sub_map->size() << std::endl;
			auto start1 = std::chrono::system_clock::now();
			auto t1 = std::chrono::high_resolution_clock::now();
			auto t_build_descriptor_begin = std::chrono::high_resolution_clock::now();
			std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
			pcl::PointCloud<PointType>::Ptr frame_plane_cloud(new pcl::PointCloud<PointType>);
			init_voxel_map(*current_sub_map, voxel_map, frame_plane_cloud); // 按照网格棱长2.0划分点云，对于点数量大于10的网格初步计算平面，最小特征值小于0.01即为平面，并获取平面的中心和法线（两者保存于一点中）
			history_plane_list.push_back(frame_plane_cloud); // 保存每个子图拟合的平面点云
			// key_frame_list.push_back(lidar_frame_); // 保存对应的雷达关键帧
			pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
			down_sampling_voxel(*current_sub_map, *temp_cloud, 0.5);
			if(1) // 保存子图位姿信息，平面信息
			{
				SubMap sub_map;
				sub_map.start_key_frame_id = start_key_frame_id;
				sub_map.end_key_frame_id = key_frame_id; // 当前子图包含的雷达帧的最后一帧id
				sub_map.sub_map_pose = current_pose;
				sub_map.sub_map_pose_opt = current_pose;
				sub_map.pose_opt_set = false;
				sub_map.sub_map_time = cur_time;
				sub_map.sub_map_cloud = temp_cloud; // 保存下采样之后的子图点云
				for(int i=0; i<6; i++)
				for(int j=0; j<6; j++)
					sub_map.sub_map_pose_cov(i,j) = cur_pose.covariance[i*6+j];
				if(use_key_frame_planes)
				{
					Eigen::Matrix4d pose_inv = current_pose.inverse();
                    int submap_global_planes_n = kf_global_planes4submap.size();
                    for(int pi=0; pi<submap_global_planes_n; pi++)
                    {
                        GlobalPlaneSimple* global_plane_ = kf_global_planes4submap[pi];
                        PlaneSimple * local_plane_ = new PlaneSimple();
                        Eigen::Matrix4d C_local = pose_inv*global_plane_->C*pose_inv.transpose();
                        double N_i = C_local(3, 3); // N_i
                        Eigen::Vector3d center = C_local.block<3, 1>(0, 3)/N_i; // 1/N_i*v_i = 1/N_i*S_p*C_i*F
		                Eigen::Matrix3d covariance = C_local.block<3, 3>(0, 0)/N_i - center * center.transpose();
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covariance);
                        Eigen::Vector3d lambdas = saes.eigenvalues();
                        if(lambdas[0]<0 || lambdas[1]<0 || lambdas[2]<0 || isnan(lambdas[0]) || isnan(lambdas[1]) || isnan(lambdas[2])) // Bad eigen!
                        {
                            std::cout<<"pi: "<<pi<<", lambdas: "<<lambdas.transpose()<<std::endl;
                            continue;
                        }
                        Eigen::Matrix3d eigenvectors = saes.eigenvectors();
                        local_plane_->points_size = N_i;
                        local_plane_->center = center;
                        local_plane_->covariance = covariance;
                        local_plane_->ready_to_pca = true;
                        local_plane_->normal = eigenvectors.col(0);
                        local_plane_->y_normal = eigenvectors.col(1);
                        local_plane_->x_normal = eigenvectors.col(2);
                        local_plane_->min_eigen_value = lambdas(0);
                        local_plane_->mid_eigen_value = lambdas(1);
                        local_plane_->max_eigen_value = lambdas(2);
                        local_plane_->d = - local_plane_->normal.dot(center);
                        if(local_plane_->d<0)
                        {
                            local_plane_->d *= -1;
                            local_plane_->normal *= -1;
                        }
                        local_plane_->is_plane = true;
                        sub_map.planes.emplace_back(local_plane_);
						// 删除全局平面中包含的子平面
						int plane_ptrs_n = global_plane_->plane_ptrs.size();
						for(int ppn=0; ppn<plane_ptrs_n; ppn++) delete global_plane_->plane_ptrs[ppn];
                        delete kf_global_planes4submap[pi];
                        kf_global_planes4submap[pi] = nullptr;
                    }
                    kf_global_planes4submap.clear();
                    kf_global_planes4submap.shrink_to_fit();
                    delete octree_kf_plane_center4submap_ptr;
                    octree_kf_plane_center4submap_ptr = new thuni::Octree();
                    octree_kf_plane_center4submap_ptr->set_min_extent(0.1);
                    octree_kf_plane_center4submap_ptr->set_bucket_size(1);
				}
				sub_maps.push_back(sub_map);
			}
			std::vector<Plane *> proj_plane_list;
			std::vector<Plane *> merge_plane_list;
			get_project_plane(voxel_map, proj_plane_list); // 根据法线相似性和中心到平面的距离判断是否属于同一平面，然后进行平面融合，融合之后的平面保存在 project_plane_list
			debug_file << "proj_plane_list.size(): " << proj_plane_list.size() << std::endl;
			if (proj_plane_list.size() == 0)
			{
				Plane *single_plane = new Plane;
				single_plane->normal_ << 0, 0, 1;
				single_plane->center_ << current_sub_map->points[0].x, current_sub_map->points[0].y, current_sub_map->points[0].z;
				merge_plane_list.push_back(single_plane);
			} 
			else
			{
				std::sort(proj_plane_list.begin(), proj_plane_list.end(), [](Plane* plane1, Plane* plane2){return plane1->sub_plane_num_ > plane2->sub_plane_num_;}); // 融合子平面的数量越多，越靠前
				merge_plane(proj_plane_list, merge_plane_list);           // 和get_project_plane一样，二次融合
				std::sort(merge_plane_list.begin(), merge_plane_list.end(), [](Plane* plane1, Plane* plane2){return plane1->sub_plane_num_ > plane2->sub_plane_num_;});
			}
			int merge_plane_list_n = merge_plane_list.size();
			// 仅仅为了显示提取的平面
			// std::vector<Plane> & merged_planes_ = sub_maps.back().merged_planes;
			// merged_planes_.resize(merge_plane_list_n);
			// for(int i=0; i<merge_plane_list_n; i++) merged_planes_[i] = *(merge_plane_list[i]);
			auto merge_plane_end = std::chrono::system_clock::now();
			auto merged_planes_ms1 = std::chrono::duration<double, std::milli>(merge_plane_end - start1);
			debug_file << "elapsed_ms aft generate merge_plane: " << merged_planes_ms1.count() <<"ms" << std::endl;
			debug_file << "merge_plane_list.size(): " << merge_plane_list_n << std::endl;
			std::vector<BinaryDescriptor> binary_list;
			// std::vector<bool> occupy_array_; 记录在哪些分段上存在点
			// unsigned char summary_; 存在点的分段的数量
			// Eigen::Vector3d location_; 这些点在平面上的重心
			// 1. 将0.5*0.5*(5-0.2)的垂直于平面的立柱沿着法线划分为多段，记录在哪些分段上存在点 occupy_array_、存在点的分段的数量 summary_、这些点在平面上的重心 location_
			// 2. 搜索半径3米内的点，再次进行最大值抑制
			// 3. 排序，保留值最大的，最多保留30个关键点
			binary_extractor(merge_plane_list, current_sub_map, binary_list);
			std::vector<STD> STD_list;
			// 通过关键点构建多个三角形，保存三角形的三个顶点的二值描述子 binary_A_、中心center_和边长triangle_、关键帧id frame_number_
			generate_std(binary_list, STD_list);
			cur_STD_list = STD_list; // 仅仅用于显示
			auto end1 = std::chrono::system_clock::now();
			auto elapsed_ms1 = std::chrono::duration<double, std::milli>(end1 - start1);
			debug_file << "elapsed_ms aft generate std: " << elapsed_ms1.count() <<"ms"<< std::endl;
			if(associate_consecutive_frame) // 对于Mulran还是有用的。。。对于NCLT好像没什么用
			{
				if (!corners_curr_->empty()) corners_curr_->clear();
				PointType a_pt;
				for (auto a_bin : binary_list)
				{
					a_pt.x = a_bin.location_[0]; a_pt.y = a_bin.location_[1]; a_pt.z = a_bin.location_[2];
					corners_curr_->push_back(a_pt);
				}
				std::vector<std::pair<PointType,PointType>> corners_pairs;
				// 构建相邻两帧之间的顶点匹配
				associate_consecutive_frames(corners_curr_, corners_last_, corners_pairs);
				debug_file << "[interT] corners sizes: " << corners_curr_->size() << " " << corners_last_->size()<<", corners_pairs.size(): "<<corners_pairs.size() << std::endl;
				// 将匹配的顶点转换到各自的局部坐标系再加入因子图中，便于后续用因子图中的位姿建立约束
				insertKPPairConstraint(corners_pairs, sub_map_id-1, sub_map_id, 1);
				corners_pairs.clear();
				*corners_last_ = *corners_curr_;
			}
			addOdomFactor();
			start1 = std::chrono::system_clock::now();
			history_binary_list.push_back(binary_list);
			debug_file << "[Corner] corner size:" << binary_list.size() << "  descriptor size:" << STD_list.size() << std::endl;
			end1 = std::chrono::system_clock::now();
			elapsed_ms1 += std::chrono::duration<double, std::milli>(end1 - start1);
			time_lcd_file << elapsed_ms1.count() << " " << 3 << std::endl;
			// candidate search
			detect_loopclosure(STD_list, frame_plane_cloud, STD_map, history_plane_list);
			gtsam_optimize();
			is_build_descriptor = false;
			// debug_file<<"add_STD"<<std::endl;
			add_STD(STD_map, STD_list);
			sub_map_id++;
			// debug_file<<"sub_map_id: "<<sub_map_id<<std::endl;
			for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) delete (iter->second);
			// debug_file<<"voxel_map delete "<<std::endl;
		}
	}

};
};

int main( int argc, char **argv )
{
	Eigen::initParallel();
    ros::init( argc, argv, "loop closure" );
	loopdetection::LoopClosureDetection loop_detection;
	loop_detection.loop_run();
	return 0;
}