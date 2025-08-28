
#pragma once
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <common_lib.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/UInt64.h>
#include <std_msgs/Float64MultiArray.h>
#include <opencv2/core/eigen.hpp>
#include <visualization_msgs/Marker.h>

#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <geometry_msgs/Vector3.h>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "lib_sophus/so3.hpp"
#include "lib_sophus/se3.hpp"

#include "tools_logger.hpp"
#include "tools_color_printf.hpp"
#include "tools_eigen.hpp"
#include "tools_data_io.hpp"
#include "tools_timer.hpp"
#include "tools_thread_pool.hpp"
#include "tools_ros.hpp"
#include "tools_mem_used.h"

#include "loam/IMU_Processing.hpp"
#include "octree/Octree.hpp"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <malloc.h>
#include "isdor_filter.hpp"

class SPLIN
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::mutex  m_mutex_lio_process;
    std::shared_ptr<ImuProcess> m_imu_process;
    double m_maximum_pt_kdtree_dis = 0.5;
    float tree_ds_size = 0.05;
    double m_maximum_res_dis = 1.0;
    double m_planar_check_dis = 0.05;
    double m_lidar_imu_time_delay = 0;
    double m_long_rang_pt_dis = 500.0;
    int NUM_MAX_ITERATIONS = 0;
    /// IMU relative variables
    std::mutex mtx_buffer;
    std::mutex mtx_path_buffer;
    std::mutex mtx_submap_info_buffer;
    std::condition_variable sig_buffer;
    bool lidar_pushed = false;
    bool flg_reset = false;
    // Buffers for measurements
    double lidar_end_time = 0.0;
    double last_timestamp_lidar = -1.0;
    double last_timestamp_imu = -1;
    std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_lio;
    ros::Publisher pubLaserCloudFullRes;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubLaserColorCloud;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubOdomKeyFrame;
    ros::Publisher pubPlaneKeyFrame;
    ros::Publisher pubLaserCloudUndistorted;
    ros::Publisher pubPath;
    ros::Publisher pubTimeCorrection; // 发布拥有正确位姿的雷达点云对应的时间戳
    ros::Publisher pubPoseRelatAftGPPO; // 发布全局平面&点优化之后的位姿关联关系，用于构建类回环约束
    ros::Publisher pubFilterPC; // 
    ros::Publisher pubUndistortedPC; // 
    ros::Publisher pubplane4extractPC;
    
    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;
    ros::Subscriber sub_pgo_path;
    ros::Subscriber sub_pgo_kf_path;
    ros::Subscriber sub_submap_info;
    bool dense_map_en;
    double m_voxel_downsample_size_surf;
    ros::NodeHandle m_ros_node_handle;
    std::string m_map_output_dir;
    const int recent_ten_frames_N = 10;
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>> recent_ten_frames = std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>>(recent_ten_frames_N); // 仅仅保留最近10帧
    int recent_ten_frames_i = 0;
    std::shared_ptr<Common_tools::ThreadPool> m_thread_pool_ptr;
    MeasureGroup Measures;
    StatesGroup g_lio_state;
    Eigen::Matrix<double, 3, 3> m_lidar_to_imu_R;
    Eigen::Matrix<double, 3, 1> m_lidar_to_imu_t;
    int num_match_pts;
    double lidar_pt_cov = 0.00015; // (0.00015==>0.001)
    double lidar_cov_p = 1.02;
    // unibn::Octree<pcl::PointXYZINormal, pcl::PointCloud<pcl::PointXYZINormal>> octree;
    thuni::Octree octree; // 用于纯点匹配
    thuni::Octree * octree_feature_delay_ptr = nullptr; // 用于点面混合匹配
    thuni::Octree * octree_feature_ptr = nullptr; // 用于点面混合匹配
    thuni::Octree * octree_feature_replace_ptr = nullptr; // 用于点面混合匹配
    thuni::Octree * octree_pose_ptr = nullptr; // 用于搜索附近子图索引
    std::deque<std::vector<Eigen::Vector3d>> tree_pts_delay;
    std::deque<std::vector<std::vector<float>>> tree_pts_attrs_delay;
    std::mutex mtx_tree_pts_delay_buffer;
    cv::RNG g_rng = cv::RNG(0);
    int g_LiDAR_frame_index = 0;
    bool debug_plot = false, plane_debug = false;
    // std::vector<GlobalPlane> global_planes;
    std::vector<LidarFrame*> all_lidar_frames;
    std::vector<std::pair<int, int>> submap_loop_ids; // 回环id 第一个为目标子图id，第二个为当前子图id
    std::vector<std::pair<int, int>> loop_ids; // 回环id 第一个为参考帧，第二个为变动帧
    std::vector<std::pair<int, int>> loop_ids_used; // 记录已经使用过的回环id 第一个为参考帧，第二个为变动帧
    std::vector<int> loop_opti_frame_ids; // 回环序列优化帧
    Eigen::Matrix4d delta_T_loop; // 回环帧的位姿变化
    int last_loop_frame_id = 0;
    bool loop_closure_optimized = false;
    bool process_end = false;
    bool process_end2 = false;
    bool wait_for_save_data = false;
    bool use_backend = true;
    double lio_pts_num = 1000;
    double plane_merge_angle=8, plane_merge_dist=0.02, max_eigen_dist_th=0.05;
    int max_merge_plane_n=20, min_plane_pts_n=50, pi_split_n=45; // 最大融合小平面数量 最小用于拟合平面的点数量 偏航角划分的份数，也就是水平方向划分为多少份
    bool use_pl_lio = false;
    bool process_debug = false;
    bool use_non_plane_pt = false;
    bool large_scale_env = false;
    int large_scale_env_tree_state = -1; // -1 无操作 0 达到开始某个阈值，比如1e6，开始准备替换 1 替换树准备完毕，开始替换
    double P_r_value;
    double error_scale = 0.01;
    std::deque<nav_msgs::Path::ConstPtr> pgo_submaps_path_buffer;
    std::deque<nav_msgs::Path::ConstPtr> pgo_KF_path_buffer;
    std::deque<nav_msgs::OdometryConstPtr> submap_info_buffer;
    std::vector<SubmapInfo> submap_infos;
    double update_dis_interval = 200;
    std::unordered_map<int, int> key_frame_id_to_submap_id;
    int curr_tree_start_frame_id = 0, next_tree_start_frame_id = 0;
    int lc_updated_front_end_frame_id = -1; // 完成回环更新的前端帧id
    std::deque<std::vector<Eigen::Vector3d>> historical_tree_pts_to_add;
    std::deque<std::vector<std::vector<float>>> historical_tree_attrs_to_add;
    std::mutex mtx_historical_tree_pts_to_add;
    std::set<int> historical_tree_attrs_added_submap_ids, historical_tree_attrs_added_replace;
    std::vector<LidarFrame*> all_key_frames; // 全部关键帧
    double lidar_blind = 1.0;
    std::vector<Eigen::Vector3d> non_act_pts;
    DynamicPointsFilter dynamic_pts_filter;
    bool frame_poses_updating = false; // 记录是否正在更新帧位姿，在更新的时候前端不能使用！
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    bool use_sphere_filter = false;
    bool use_voxel_filter = true;
    ISDOR_Simple vdf;
    bool show_mem_use = false;
    bool use_key_frame_planes = false;
    bool rematch_less = true;
    bool decouple_front = false; // 解耦前后端，前端的运行不受后端的影响，也就是回环检测之后不更新前端的搜索树和位姿
    ISDOR isdor;
    bool use_isdor = false;
    bool isdor_robust_flag = false;
    double plane2orig_dist_th = 0.2; // 点到面的距离太近的话可能是同一条扫描线上的

    SPLIN()
    {
        // 发布器
        {
            pubLaserColorCloud = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/color_plane", 100);
            pubLaserCloudFullRes = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
            pubLaserCloudEffect = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
            pubLaserCloudMap = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
            pubOdomAftMapped = m_ros_node_handle.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
            pubPath = m_ros_node_handle.advertise<nav_msgs::Path>("/path", 10);
            pubLaserCloudUndistorted = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/key_frame_undistorted", 100);
            pubOdomKeyFrame = m_ros_node_handle.advertise<nav_msgs::Odometry>("/key_frame_pose", 10);
            pubPlaneKeyFrame = m_ros_node_handle.advertise<std_msgs::Float64MultiArray>("/key_frame_planes", 10);
            pubTimeCorrection = m_ros_node_handle.advertise<std_msgs::UInt64>("/time_correction", 10);
            pubPoseRelatAftGPPO = m_ros_node_handle.advertise<nav_msgs::Path>("/aft_gppo_poses_relat", 100);
            pubFilterPC = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/filtered_pc", 100);
            pubUndistortedPC = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/undistorted_pc", 100);
            pubplane4extractPC = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/frame4plane_extract", 100);
        }
        // 订阅器
        {
            std::string LiDAR_pointcloud_topic, IMU_topic;
            get_ros_parameter<std::string>(m_ros_node_handle, "/IMU_topic", IMU_topic, std::string("/livox/imu") );
            get_ros_parameter<std::string>(m_ros_node_handle, "/LiDAR_pointcloud_topic", LiDAR_pointcloud_topic, std::string("/laser_cloud_flat") );
            scope_color(ANSI_COLOR_BLUE_BOLD);
            cout << "======= Summary of subscribed topics =======" << endl;
            cout << "LiDAR pointcloud topic: " << LiDAR_pointcloud_topic << endl;
            cout << "IMU topic: " << IMU_topic << endl;
            cout << "=======        -End-                =======" << endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            sub_imu = m_ros_node_handle.subscribe(IMU_topic.c_str(), 2000000, &SPLIN::imu_cbk, this, ros::TransportHints().tcpNoDelay());
            sub_pcl = m_ros_node_handle.subscribe(LiDAR_pointcloud_topic.c_str(), 2000000, &SPLIN::feat_points_cbk, this, ros::TransportHints().tcpNoDelay());
            sub_pgo_path = m_ros_node_handle.subscribe<nav_msgs::Path>("/aft_pgo_submaps_path", 2000000, &SPLIN::pgo_submaps_path_cbk, this, ros::TransportHints().tcpNoDelay());
            sub_pgo_kf_path = m_ros_node_handle.subscribe<nav_msgs::Path>("/aft_pgo_keyframes_path", 2000000, &SPLIN::pgo_kf_path_cbk, this, ros::TransportHints().tcpNoDelay());
            sub_submap_info = m_ros_node_handle.subscribe<nav_msgs::Odometry>("/submap_info", 2000000, &SPLIN::submap_info_cbk, this, ros::TransportHints().tcpNoDelay());
        }
        // 参数
        {
            // get_ros_parameter( m_ros_node_handle, "poslam_common/map_output_dir", m_map_output_dir, ros::package::getPath("splin").append( "/../output" ) );
            get_ros_parameter( m_ros_node_handle, "SaveDir", m_map_output_dir, std::string("/media/zhujun/0DFD06D20DFD06D2/ws_PPSLAM/src/SPLIN/output/tmp") );
            if(!Common_tools::if_file_exist(m_map_output_dir))
            {
                cout << ANSI_COLOR_BLUE_BOLD << "Create splin output dir: " << m_map_output_dir << ANSI_COLOR_RESET << endl;
                Common_tools::create_dir(m_map_output_dir);
            }
            // scope_color( ANSI_COLOR_GREEN );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/dense_map_enable", dense_map_en, true );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/lidar_time_delay", m_lidar_imu_time_delay, 0.0 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/max_iteration", NUM_MAX_ITERATIONS, 4 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/voxel_downsample_size_surf", m_voxel_downsample_size_surf, 0.3 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/maximum_pt_kdtree_dis", m_maximum_pt_kdtree_dis, 0.5 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/maximum_res_dis", m_maximum_res_dis, 0.3 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/planar_check_dis", m_planar_check_dis, 0.10 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/long_rang_pt_dis", m_long_rang_pt_dis, 500.0 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/num_match_pts", num_match_pts, 5 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/lidar_cov_p", lidar_cov_p, 1.02 );
            get_ros_parameter( m_ros_node_handle, "poslam_lio/use_backend", use_backend, true);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/lio_pts_num", lio_pts_num, 1000.0);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/plane_merge_angle", plane_merge_angle, 8.0);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/plane_merge_dist", plane_merge_dist, 0.02);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/max_merge_plane_n", max_merge_plane_n, 20);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/min_plane_pts_n", min_plane_pts_n, 50);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/pi_split_n", pi_split_n, 45);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/use_pl_lio", use_pl_lio, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/process_debug", process_debug, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/max_eigen_dist_th", max_eigen_dist_th, 0.05);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/use_non_plane_pt", use_non_plane_pt, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/P_r_value", P_r_value, 1.0);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/error_scale", error_scale, 0.01);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/tree_ds_size", tree_ds_size, 0.05f);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/large_scale_env", large_scale_env, false);
            get_ros_parameter( m_ros_node_handle, "Lidar_front_end/blind", lidar_blind, 1.0);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/use_sphere_filter", use_sphere_filter, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/use_voxel_filter", use_voxel_filter, true);
            get_ros_parameter( m_ros_node_handle, "loop_detection/use_key_frame_planes", use_key_frame_planes, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/rematch_less", rematch_less, true);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/decouple_front", decouple_front, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/use_isdor", use_isdor, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/sparse_scan", isdor.sparse_scan, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/isdor_robust_flag", isdor_robust_flag, false);
            get_ros_parameter( m_ros_node_handle, "poslam_lio/plane2orig_dist_th", plane2orig_dist_th, 0.2);

            std::vector< double > lidar_to_imu_R_data, lidar_to_imu_t_data;
            // std::cout<<"get data:"<<std::endl;
            m_ros_node_handle.getParam( "poslam_lio/lidar_to_imu_R", lidar_to_imu_R_data );
            m_ros_node_handle.getParam( "poslam_lio/lidar_to_imu_t", lidar_to_imu_t_data );
            // std::cout<<"lidar_to_imu_R_data.size(): "<<lidar_to_imu_R_data.size()<<std::endl;
            // std::cout<<"lidar_to_imu_t_data.size(): "<<lidar_to_imu_t_data.size()<<std::endl;
            m_lidar_to_imu_R = Eigen::Matrix< double, 3, 3 >::Identity();
            m_lidar_to_imu_t = Eigen::Matrix< double, 3, 1 >::Zero();
            if(lidar_to_imu_R_data.size()==9 && lidar_to_imu_t_data.size()==3)
            {
                m_lidar_to_imu_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( lidar_to_imu_R_data.data() );
                m_lidar_to_imu_t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( lidar_to_imu_t_data.data() );
            }
        }
        // 类实例化
        {
            octree_feature_ptr = new thuni::Octree();
            octree_feature_delay_ptr = new thuni::Octree();
            octree_feature_replace_ptr = new thuni::Octree();
            octree_pose_ptr = new thuni::Octree();
            m_thread_pool_ptr = std::make_shared<Common_tools::ThreadPool>(2, true, false); // At least 5 threads are needs, here we allocate 6 threads.
            m_imu_process = std::make_shared<ImuProcess>(m_lidar_to_imu_R, m_lidar_to_imu_t); // At least 5 threads are needs, here we allocate 6 threads.
            // IMU参数设置
            {
                m_ros_node_handle.param< double >( "poslam_lio/cov_acc", m_imu_process->cov_acc, 0.4 );
                m_ros_node_handle.param< double >( "poslam_lio/cov_gyr", m_imu_process->cov_gyr, 0.2 );
                m_ros_node_handle.param< double >( "poslam_lio/cov_bias_acc", m_imu_process->cov_bias_acc, 0.05 );
                m_ros_node_handle.param< double >( "poslam_lio/cov_bias_gyro", m_imu_process->cov_bias_gyro, 0.1 );
                m_ros_node_handle.param< double >( "poslam_lio/cov_omega", m_imu_process->cov_omega, 0.1 );
            }
            downSizeFilterSurf.setLeafSize(m_voxel_downsample_size_surf, m_voxel_downsample_size_surf, m_voxel_downsample_size_surf);
            if(use_backend)
                m_thread_pool_ptr->commit_task(&SPLIN::data_association_thread, this);
        }
        std::cout<<"Init done!"<<std::endl;
    }
    ~SPLIN()
    {
        if(octree_feature_ptr != nullptr) delete octree_feature_ptr;
        if(octree_feature_delay_ptr != nullptr) delete octree_feature_delay_ptr;
        if(octree_feature_replace_ptr != nullptr) delete octree_feature_replace_ptr;
        if(octree_pose_ptr != nullptr) delete octree_pose_ptr;
    };
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
    bool sync_packages(MeasureGroup &meas);
    void feat_points_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg_in);
    bool get_pointcloud_data_from_ros_message(sensor_msgs::PointCloud2::ConstPtr & msg, pcl::PointCloud<pcl::PointXYZINormal> & pcl_pc);
    int service_LIO_update_plane();
    void set_initial_state_cov( StatesGroup &state );
    void data_association_thread();

    void publish_keyframe(LidarFrame * lidar_frame)
    {
        // 发布位姿
        nav_msgs::Odometry odomAftMapped;
        odomAftMapped.header.frame_id = "world";
        odomAftMapped.child_frame_id = "/aft_mapped";
        // odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
        odomAftMapped.header.stamp.fromSec( lidar_frame->lidar_end_time );
        Eigen::Quaterniond geoQuat(lidar_frame->R_w_l);
        odomAftMapped.pose.pose.orientation.x = geoQuat.x();
        odomAftMapped.pose.pose.orientation.y = geoQuat.y();
        odomAftMapped.pose.pose.orientation.z = geoQuat.z();
        odomAftMapped.pose.pose.orientation.w = geoQuat.w();
        odomAftMapped.pose.pose.position.x = lidar_frame->t_w_l( 0 );
        odomAftMapped.pose.pose.position.y = lidar_frame->t_w_l( 1 );
        odomAftMapped.pose.pose.position.z = lidar_frame->t_w_l( 2 );
        // 增量差异带来的协方差变化 左乘==>右乘
        Eigen::Matrix<double, 6,6> W_r = Eigen::Matrix<double, 6,6>::Identity();
        W_r.block<3,3>(0,0) = lidar_frame->R_w_l.transpose();
        W_r.block<3,3>(3,0) = -lidar_frame->R_w_l.transpose()*vec_to_hat(lidar_frame->t_w_l);
        W_r.block<3,3>(3,3) = lidar_frame->R_w_l.transpose();
        Eigen::Matrix<double, 6,6> cov1 = W_r*lidar_frame->cov*W_r.transpose();
        // float64[36] covariance 6*6
        for(int i=0; i<6; i++)
        for(int j=0; j<6; j++)
            odomAftMapped.pose.covariance[i*6+j] = cov1(i,j);
        odomAftMapped.twist.covariance[0] = double(lidar_frame->key_frame_id);
        pubOdomKeyFrame.publish(odomAftMapped);
        // 发布平面数据
        if(use_key_frame_planes)
        {
            int plane_n_ = lidar_frame->merged_planes.size()+lidar_frame->common_planes.size();
            int merge_plane_n_ = lidar_frame->merged_planes.size();
            // 更新平面中心搜索树
            std::vector<Eigen::Vector3d> plane_grid_centers_ = lidar_frame->plane_grid_centers;
            const std::vector<std::vector<float>> & plane_grid_attrs_ = lidar_frame->plane_grid_attrs;
            const int plane_grid_centers_n = plane_grid_centers_.size();
            std_msgs::Float64MultiArray msg;
            msg.layout.dim.resize(4);
            msg.layout.dim[0].label = "num_matrices";
            msg.layout.dim[0].size = plane_n_;
            msg.layout.dim[0].stride = plane_n_ * 16;
            msg.layout.dim[1].label = "flattened_matrix";
            msg.layout.dim[1].size = 16;
            msg.layout.dim[1].stride = 16;
            msg.layout.dim[2].label = "center_0";
            msg.layout.dim[2].size = merge_plane_n_;
            msg.layout.dim[2].stride = 3;
            msg.layout.dim[3].label = "grid_centers";
            msg.layout.dim[3].size = plane_grid_centers_n;
            msg.layout.dim[3].stride = 4;
            msg.layout.data_offset = lidar_frame->key_frame_id;
            int C_ij_float_num = plane_n_*16;
            int center_0_float_num = merge_plane_n_*3;
            int grid_centers_float_num = plane_grid_centers_n*4;
            int total_float_n = C_ij_float_num + center_0_float_num + grid_centers_float_num; // C_ij + center_0 + grid_centers
            msg.data.resize(total_float_n);
            std::set<int> not_good_plane_ids;
            double max_mid_eigen_ratio_ = 6.0;
            for(int pi=0; pi<plane_n_; pi++)
            {
                Plane * local_plane_ = lidar_frame->get_plane(pi);
                bool is_good_plane = true;
                // if(local_plane_->get_max_eigen_dist()>max_mid_eigen_ratio_*local_plane_->get_mid_eigen_dist() || abs(local_plane_->d)<0.2) is_good_plane = false;  // 这种极有可能是一条扫描线上的点！
                if(abs(local_plane_->d)<0.2) is_good_plane = false;  // 这种极有可能是一条扫描线上的点！
                const double & norm_scale_ = local_plane_->norm_scale;
                local_plane_->cal_cov_and_center();
                if(is_good_plane && local_plane_->center[2]>0.0 && local_plane_->norm_ints.size()>1 && local_plane_->max_norm-local_plane_->min_norm>2.0)
                {
                    int center_norm_int = local_plane_->center.norm()/norm_scale_;
                    std::vector<int> norm_ints_vec;
                    for(const int & ni: local_plane_->norm_ints) norm_ints_vec.emplace_back(ni);
                    std::sort(norm_ints_vec.begin(), norm_ints_vec.end(), [](int & a1, int & a2){return a1<a2;});
                    // int min_gap_int_l = 100, min_gap_int_r = 100;
                    int last_ni = norm_ints_vec[0];
                    for(const int & ni: norm_ints_vec)
                    {
                        // if(ni<center_norm_int) min_gap_int_l = center_norm_int - ni;
                        // else
                        // {
                        //     min_gap_int_r = ni - center_norm_int;
                        //     break;
                        // }
                        if((ni-last_ni)*norm_scale_>=3)
                        {
                            is_good_plane = false;
                            break;
                        }
                        last_ni = ni;
                        if(ni>center_norm_int) break;
                    }
                    // if(min_gap_int_l*norm_scale_>1 || min_gap_int_r*norm_scale_>1) is_good_plane = false;
                }
                if(!is_good_plane)
                {
                    not_good_plane_ids.insert(pi);
                    for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        msg.data[pi * 16 + r * 4 + c] = 0.0;
                    if(pi<merge_plane_n_) // 防止融合平面中心处于非平面块区域
                    {
                        for (int c = 0; c < 3; ++c)
                            msg.data[C_ij_float_num + pi * 3 + c] = 0.0;
                    }
                    continue;
                }
                local_plane_->recover_for_insert();
                Eigen::Matrix<double, 4,4> C_ij;
                C_ij << local_plane_->covariance, local_plane_->center, local_plane_->center.transpose(), local_plane_->points_size;
                for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    msg.data[pi * 16 + r * 4 + c] = C_ij(r, c);
                if(pi<merge_plane_n_) // 防止融合平面中心处于非平面块区域
                {
                    for (int c = 0; c < 3; ++c)
                        msg.data[C_ij_float_num + pi * 3 + c] = local_plane_->center_0[c];
                }
                local_plane_->cal_cov_and_center();
            }
            for(int j=0; j<plane_grid_centers_n; j++)
            {
                for (int c = 0; c < 3; ++c)
                    msg.data[C_ij_float_num + center_0_float_num + j * 4 + c] = plane_grid_centers_[j][c];
                msg.data[C_ij_float_num +center_0_float_num + j * 4 + 3] = plane_grid_attrs_[j][0];
                int local_plane_id_ = plane_grid_attrs_[j][0];
                if(not_good_plane_ids.count(local_plane_id_)>0) msg.data[C_ij_float_num +center_0_float_num + j * 4 + 3] = -1.0;
            }
            pubPlaneKeyFrame.publish(msg);
        }
        // 发布点云数据
        PointCloudXYZINormal::Ptr LaserCloudUndistorted( new PointCloudXYZINormal() );
        LaserCloudUndistorted->clear();
        // *LaserCloudUndistorted = dense_map_en ? ( *feats_undistort ) : ( *feats_down );
        if(lidar_frame->dynamic_filter_remained_pts.size()>0)
        {
            for(const Eigen::Vector3d & pt_ : lidar_frame->dynamic_filter_remained_pts )
            {
                PointType p;
                p.x = pt_[0];
                p.y = pt_[1];
                p.z = pt_[2];
                LaserCloudUndistorted->push_back(p);
            }
            for(const Eigen::Vector3d & pt_ : lidar_frame->dynamic_grd_pts )
            {
                PointType p;
                p.x = pt_[0];
                p.y = pt_[1];
                p.z = pt_[2];
                LaserCloudUndistorted->push_back(p);
            }
        }
        else
            *LaserCloudUndistorted = *(lidar_frame->point_cloud_ptr);
        sensor_msgs::PointCloud2 LaserCloudUndistorted2;
        pcl::toROSMsg( *LaserCloudUndistorted, LaserCloudUndistorted2 );
        // LaserCloudUndistorted2.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        LaserCloudUndistorted2.header.stamp.fromSec( lidar_frame->lidar_end_time );
        LaserCloudUndistorted2.header.frame_id = "world"; // world; camera_init
        pubLaserCloudUndistorted.publish( LaserCloudUndistorted2 );
    }

    void pgo_submaps_path_cbk(const nav_msgs::Path::ConstPtr &path_in)
    {
        mtx_path_buffer.lock();
        pgo_submaps_path_buffer.push_back(path_in);
        mtx_path_buffer.unlock();
    }

    void pgo_kf_path_cbk(const nav_msgs::Path::ConstPtr &path_in)
    {
        mtx_path_buffer.lock();
        pgo_KF_path_buffer.push_back(path_in);
        mtx_path_buffer.unlock();
    }

    void submap_info_cbk(const nav_msgs::Odometry::ConstPtr &odom_msg)
    {
        mtx_submap_info_buffer.lock();
        submap_info_buffer.push_back(odom_msg);
        mtx_submap_info_buffer.unlock();
    }

    char cv_keyboard_callback(int ms=1)
    {
        char c =  cv::waitKey(ms);
        // std::cout<<"capture "<<std::string(&c)<<std::endl;
        if ( c == 'l' || c == 'L')
        {
            scope_color( ANSI_COLOR_GREEN_BOLD );
            cout << "I capture the keyboard input!!!" << endl;
            m_ros_node_handle.setParam("/save_map", true);
            cout << "I capture the keyboard input!!! end" << endl;
        }
        if(c=='p' || c=='P')
        {
            scope_color( ANSI_COLOR_GREEN_BOLD );
            cout << "I capture the keyboard input!!!" << endl;
            debug_plot = !debug_plot;
            std::cout<<"debug_plot: "<<debug_plot<<std::endl;
        }
        if(c=='d' || c=='D')
        {
            scope_color( ANSI_COLOR_GREEN_BOLD );
            cout << "I capture the keyboard input!!!" << endl;
            plane_debug = !plane_debug;
            std::cout<<"plane_debug: "<<plane_debug<<std::endl;
        }
        if(c=='g' || c=='G')
        {
            scope_color( ANSI_COLOR_GREEN_BOLD );
            cout << "I capture the keyboard input!!!" << endl;
            cout << "I capture the keyboard input!!! end" << endl;
        }
        if(c=='e' || c=='E')
        {
            scope_color( ANSI_COLOR_GREEN_BOLD );
            cout << "I capture the keyboard input!!!" << endl;
            process_end2 = true;
        }
        if(c=='o' || c=='O')
        {
            scope_color( ANSI_COLOR_GREEN_BOLD );
            cout << "I capture the keyboard input!!!" << endl;
            cout << "I capture the keyboard input!!! end" << endl;
        }
        return c;
    }

    cv::Mat generate_control_panel_img()
    {
        int line_y = 40;
        int padding_x = 10;
        int padding_y = line_y * 0.7;
        std::vector<std::string> to_show_strs;
        to_show_strs.push_back("Click this windows to enable the keyboard controls.");
        to_show_strs.push_back("Press 'P' or 'p' to save isdor filter results.");
        to_show_strs.push_back("Press 'L' or 'l' to save recent frames and octee.");
        to_show_strs.push_back("Press 'D' or 'd' to save plane extract results.");
        to_show_strs.push_back("Press 'A' or 'a' to save all frames.");
        to_show_strs.push_back("Press 'E' or 'e' to end the process.");
        to_show_strs.push_back("Press 'O' or 'o' to save octree pts and current frame match.");
        // to_show_strs.push_back("Click");
        int num_lines = to_show_strs.size();
        cv::Mat res_image = cv::Mat(line_y * num_lines + 1 * padding_y, 1250, CV_8UC3, cv::Scalar::all(0));
        for(int i=0; i<num_lines; i++)
        {
            uint8_t r = i==0? 0 : 255;
            cv::putText(res_image, to_show_strs[i], cv::Point(padding_x, line_y * i + padding_y), cv::FONT_HERSHEY_COMPLEX, 1,cv::Scalar(r, 255, 255), 2, 8, 0);
        }
        return res_image;
    }

    template <typename PCType = pcl::PointCloud<pcl::PointXYZINormal>>
    void pc_voxel_filter(const PCType & pc_in, PCType & pc_out, float voxel_size = 1.0, bool no_sort = false)
    {
        std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
        for (int i=0; i<pc_in.points.size(); i++)
        {
            int64_t x = std::round(pc_in.points[i].x/voxel_size);
            int64_t y = std::round(pc_in.points[i].y/voxel_size);
            int64_t z = std::round(pc_in.points[i].z/voxel_size);
            VOXEL_LOC position(x, y, z);
            feat_map_tmp[position].push_back(i);
        }
        pc_out.clear();
        // int min_ptn = 1e5;
        for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter)
        {
            if(no_sort)
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
            for(int i=0; i<pt_n; i++)
            {
                int id = iter->second[i];
                float dist = (center - Eigen::Vector3f(pc_in.points[id].x, pc_in.points[id].y, pc_in.points[id].z)).norm();
                if(dist<min_dist)
                {
                    min_dist = dist;
                    best_id = id;
                }
            }
            pc_out.push_back(pc_in.points[best_id]);
        }
        // std::cout<<"min_ptn: "<<min_ptn<<std::endl;
    }

    template <typename PCType = pcl::PointCloud<pcl::PointXYZINormal>>
    void pc_voxel_filter_eigen(const PCType & pc_in, std::vector<Eigen::Vector3d> & pc_out, float voxel_size = 1.0, bool no_sort = false, std::vector<VOXEL_LOC> * orig_pts_down_eigen_keys_ = nullptr)
    {
        std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
        for (int i=0; i<pc_in.points.size(); i++)
        {
            int64_t x = std::round(pc_in.points[i].x/voxel_size);
            int64_t y = std::round(pc_in.points[i].y/voxel_size);
            int64_t z = std::round(pc_in.points[i].z/voxel_size);
            VOXEL_LOC position(x, y, z);
            feat_map_tmp[position].push_back(i);
        }
        pc_out.clear();
        // int min_ptn = 1e5;
        for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter)
        {
            if(orig_pts_down_eigen_keys_ != nullptr) orig_pts_down_eigen_keys_->push_back(iter->first);
            Eigen::Vector3d pt_ = Eigen::Vector3d(pc_in.points[iter->second[0]].x, pc_in.points[iter->second[0]].y, pc_in.points[iter->second[0]].z);
            if(no_sort)
            {
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
            for(int i=0; i<pt_n; i++)
            {
                int id = iter->second[i];
                Eigen::Vector3d pt_1 =  Eigen::Vector3d(pc_in.points[id].x, pc_in.points[id].y, pc_in.points[id].z);
                float dist = (center - pt_1).norm();
                if(dist<min_dist)
                {
                    min_dist = dist;
                    best_pt = pt_1;
                }
            }
            pc_out.push_back(best_pt);
        }
        // std::cout<<"min_ptn: "<<min_ptn<<std::endl;
    }

    template <typename EiPts>
    void eigen_pts_voxel_filter(const EiPts & pc_in, EiPts & pc_out, float voxel_size = 0.05, bool no_sort = false)
    {
        std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
        for (int i=0; i<pc_in.size(); i++)
        {
            int64_t x = std::round(pc_in[i][0]/voxel_size);
            int64_t y = std::round(pc_in[i][1]/voxel_size);
            int64_t z = std::round(pc_in[i][2]/voxel_size);
            VOXEL_LOC position(x, y, z);
            feat_map_tmp[position].push_back(i);
        }
        pc_out.clear();
        // pc_out.reserve(pc_in.size());
        // int min_ptn = 1e5;
        for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter)
        {
            if(iter->second.size()<2) continue;
            if(no_sort)
            {
                int median_id = iter->second.size()/2;
                pc_out.push_back(pc_in[iter->second[median_id]]);
                continue;
            }
            int best_id = 0;
            float min_dist = 1e8;
            int pt_n = iter->second.size();
            Eigen::Vector3d center(iter->first.x, iter->first.y, iter->first.z);
            center *= voxel_size;
            for(int i=0; i<pt_n; i++)
            {
                int id = iter->second[i];
                float dist = (center - pc_in[id]).norm();
                if(dist<min_dist)
                {
                    min_dist = dist;
                    best_id = id;
                }
            }
            pc_out.push_back(pc_in[best_id]);
        }
    }

    void down_sampling_voxel_unmap(std::unordered_map<VOXEL_LOC, std::vector<float>> & feat_map_tmp, const std::vector<Eigen::Vector3d> & tree_pts_, const std::vector<std::vector<float>> & tree_pts_attrs_, float voxel_size_ = 0.05)
    {
        int tree_pts_n = tree_pts_.size();
        for (int i=0; i<tree_pts_n; i++)
        {
            int64_t x = std::round(tree_pts_[i][0]/voxel_size_);
            int64_t y = std::round(tree_pts_[i][1]/voxel_size_);
            int64_t z = std::round(tree_pts_[i][2]/voxel_size_);
            VOXEL_LOC position(x, y, z);
            if (feat_map_tmp.find(position) != feat_map_tmp.end()) continue;
            std::vector<float> pt_ = {tree_pts_[i][0], tree_pts_[i][1], tree_pts_[i][2], tree_pts_attrs_[i][0], tree_pts_attrs_[i][1]};
            feat_map_tmp[position] = pt_;
        }
    }

    // 对于回环检测优化之后的子图，如果这些子图和最新关键帧较近，则将这些子图中的树点加入搜索树（如果octree_feature_ptr_不为空则直接加入，否则先保存，前端更新搜索树的时候再加入）
    void add_nearby_historical_tree_pts(std::set<int> & tree_pts_added_submap_ids, thuni::Octree * octree_feature_ptr_ = nullptr, int lc_prev_submap_id = -1, float search_radius = 50.0, double angle_th=120)
    {
        if(octree_pose_ptr->size()<1) return; // 输出子图点云
        std::vector<std::vector<float>> points_near;
        std::vector<float> pointSearchSqDis_surf;
        int all_kf_lidar_frames_n = all_key_frames.size();
        LidarFrame * lidar_frame = all_key_frames[all_kf_lidar_frames_n-1];
        LidarFrame * last_lidar_frame = all_key_frames[all_kf_lidar_frames_n-2];
        Eigen::Vector3d move_dir = lidar_frame->t_w_l_loop - last_lidar_frame->t_w_l_loop;
        octree_pose_ptr->radiusNeighbors_eigen(lidar_frame->t_w_l_loop, search_radius, points_near, pointSearchSqDis_surf);
        std::vector<Eigen::Vector3d> historical_tree_pts;
        std::vector<std::vector<float>> historical_tree_attrs;
        if(lc_prev_submap_id>-1) // 保证起码有一个子图点云
        {
            tree_pts_added_submap_ids.insert(lc_prev_submap_id);
            Eigen::Matrix4d delta_T = submap_infos[lc_prev_submap_id].corr_pose*submap_infos[lc_prev_submap_id].orig_pose.inverse();
            int pt_n = submap_infos[lc_prev_submap_id].tree_pts.size();
            std::vector<Eigen::Vector3d> lc_tree_pts(pt_n);
            for (int j=0;  j<pt_n; j++)
                lc_tree_pts[j] = delta_T.block<3,3>(0,0)*submap_infos[lc_prev_submap_id].tree_pts[j] + delta_T.block<3,1>(0,3);
            historical_tree_pts.insert(historical_tree_pts.end(), lc_tree_pts.begin(), lc_tree_pts.end());
            historical_tree_attrs.insert(historical_tree_attrs.end(), submap_infos[lc_prev_submap_id].tree_pts_attrs.begin(), submap_infos[lc_prev_submap_id].tree_pts_attrs.end());
        }
        int points_near_n = points_near.size();
        for(int i=0; i<points_near_n; i++)
        {
            int submap_id_ = int(points_near[i][3]);
            if(!submap_infos[submap_id_].corPoseFlag) continue;
            if(tree_pts_added_submap_ids.count(submap_id_)>0) continue;
            int submap_start_frame_id = all_key_frames[submap_infos[submap_id_].start_key_frame_id]->frame_id;
            int submap_end_frame_id   = all_key_frames[submap_infos[submap_id_].end_key_frame_id]->frame_id + 1;
            if(submap_start_frame_id>curr_tree_start_frame_id) continue;
            Eigen::Vector3d dir_ = submap_infos[submap_id_].corr_pose.block<3,1>(0,3)-lidar_frame->t_w_l_loop;
            double cos_ = move_dir.dot(dir_)/(move_dir.norm()*dir_.norm());
            double angle_ = acos(cos_)/3.1415926*180;
            // std::cout<<"angle_: "<<angle_<<std::endl;
            if(angle_>angle_th) continue;
            tree_pts_added_submap_ids.insert(submap_id_);
            Eigen::Matrix4d delta_T = submap_infos[submap_id_].corr_pose*submap_infos[submap_id_].orig_pose.inverse();
            int pt_n = submap_infos[submap_id_].tree_pts.size();
            std::vector<Eigen::Vector3d> lc_tree_pts(pt_n);
            for (int j=0;  j<pt_n; j++)
                lc_tree_pts[j] = delta_T.block<3,3>(0,0)*submap_infos[submap_id_].tree_pts[j] + delta_T.block<3,1>(0,3);
            historical_tree_pts.insert(historical_tree_pts.end(), lc_tree_pts.begin(), lc_tree_pts.end());
            historical_tree_attrs.insert(historical_tree_attrs.end(), submap_infos[submap_id_].tree_pts_attrs.begin(), submap_infos[submap_id_].tree_pts_attrs.end());
        }
        std::cout<<"historical_tree_pts.size(): "<<historical_tree_pts.size()<<std::endl;
        if(octree_feature_ptr_==nullptr) // 搜索树为空，等待前端更新的时候加入
        {
            mtx_historical_tree_pts_to_add.lock();
            historical_tree_pts_to_add.emplace_back(historical_tree_pts);
            historical_tree_attrs_to_add.emplace_back(historical_tree_attrs);
            mtx_historical_tree_pts_to_add.unlock();
            return;
        }
        std::cout<<"octree_feature_ptr_->size(): "<<octree_feature_ptr_->size()<<std::endl;
        double historical_tree_pts_t = omp_get_wtime();
        octree_feature_ptr_->update_with_attr(historical_tree_pts, historical_tree_attrs, true);
        double historical_tree_pts_end_t = omp_get_wtime();
        std::cout<<"add_nearby_historical_tree_pts octree_feature_ptr_->size(): "<<octree_feature_ptr_->size()<<", update_with_attr time: "<<historical_tree_pts_end_t-historical_tree_pts_t<<std::endl;
    }

    // 仅仅检索平面帧
    static bool avoid_non_plane_pts_func(const float * p_,  const int &  dim_)
    {
        if(dim_>=6 && p_[5]<0.0) return false; // 目前 p_[5] 为平面在所在帧中的 local_plane_id ，这个值小于0的为非平面点
        return true;
    }

    // 仅仅检索 frame_id 小于等于 param 的帧
    static bool search_remote_past_frame_pts_func(const float * p_, const int & dim_, const int & param)
    {
        if(dim_>=5 && p_[4]>param) return false; // 目前 p_[4] 为点在所在帧中的 frame_id, 仅仅检索 frame_id 小于等于 param 的帧
        return true;
    }

    // 检索时间上相距较远的帧
    static bool search_remote_past_frame_pts_avoid_nonplane_func(const float * p_, const int & dim_, const int & param)
    {
        if( dim_>=6 && (p_[4]>param || p_[5]<0.0) ) return false; // 目前 p_[4] 为点在所在帧中的 frame_id, 仅仅检索 frame_id 小于等于 param 的帧
        return true;
    }

};



