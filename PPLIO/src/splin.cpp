
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
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/eigen.hpp>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>

#include "common_lib.h"

#include "splin.hpp"

#include "loam/IMU_Processing.hpp"
#include "tools_logger.hpp"
#include "tools_color_printf.hpp"
#include "tools_eigen.hpp"
#include "tools_data_io.hpp"
#include "tools_timer.hpp"
#include "tools_openCV_3_to_4.hpp"

int main(int argc, char **argv)
{
    printf_program("PPLIO");
    Common_tools::printf_software_version();
    Eigen::initParallel();
    ros::init(argc, argv, "PPLIO");
    // PlaneMap plane_map_;
    // plane_map_.test_debug();
    // SPLIN * fast_lio_instance = new SPLIN();
    auto fast_lio_instance = std::make_unique<SPLIN>();
    // 启动一个线程运行 data_association_thread 函数
    // std::thread thread_backend;
    // if(fast_lio_instance->use_backend)
    // {
    //     thread_backend= std::thread(&SPLIN::data_association_thread, fast_lio_instance.get());
    // }
    if(fast_lio_instance->use_pl_lio)
        // fast_lio_instance->service_LIO_update_plane2();
        fast_lio_instance->service_LIO_update_plane();
    else
        fast_lio_instance->service_LIO_update();
    // 等待子线程结束（如果service_LIO_update_plane返回的话）
    // std::cout<<"getchar to end"<<std::endl;
    // getchar();
    // if(fast_lio_instance->use_backend && thread_backend.joinable())
    // {
    //     thread_backend.join();
    // }
    // ros::Rate rate(5000);
    // bool status = ros::ok();
    // ros::spin();
    return 0;
}
