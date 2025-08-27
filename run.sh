cd /media/zhujun/0DFD06D20DFD06D2/ws_PPSLAM/
# catkin_make clean -DCATKIN_WHITELIST_PACKAGES="package1;package2"
# catkin_make clean -DCATKIN_WHITELIST_PACKAGES="lidar_localization"
# rm -r build devel install #* 主要是为了删除build中的main函数导致的重复定义

catkin_make -DCMAKE_BUILD_TYPE=Release -DCATKIN_WHITELIST_PACKAGES="splin" -j16 -DCMAKE_CXX_FLAGS="" # livox_ros_driver  splin  poslam_camera_model
# catkin_make -DCMAKE_BUILD_TYPE=Debug -DCATKIN_WHITELIST_PACKAGES="splin" -j16 -DCMAKE_CXX_FLAGS="" # livox_ros_driver  splin  poslam_camera_model

# catkin_make -j16 -DCMAKE_BUILD_TYPE=Release
# catkin_make -j16 -DCMAKE_BUILD_TYPE=Debug

# # 清理旧编译
# catkin clean -y
# # 使用 scan-build 运行编译
# scan-build -o ./clang_analysis_report catkin_make -DCMAKE_BUILD_TYPE=Debug
# # roslaunch splin octree_test.launch

#* 程序运行
# roscore
# cd /media/zhujun/0DFD06D20DFD06D2/ws_PPSLAM/src/SPLIN
# source devel/setup.bash
# export LD_LIBRARY_PATH=/media/zhujun/0DFD06D20DFD06D2/ws_LTAOM_orig/src/LTAOM/thirdparty/lib:$LD_LIBRARY_PATH

# roslaunch splin poslam_run.launch
# rosparam set /save_map true 
# roslaunch file_player file_player.launch
# roslaunch file_player_nclt file_player_nclt.launch
# roslaunch fast_lio mapping_hilti.launch
# roslaunch fast_lio mapping_ouster64.launch
# roslaunch fast_lio mapping_ouster64mulran.launch
# 机械硬盘读取太慢了
# rosbag play /home/zhujun/SLAM_DATA/20230225/camera1-half.bag  /home/zhujun/SLAM_DATA/20230225/lidar-1.bag
# rosbag play  /home/zhujun/SLAM_DATA/20230225/lidar-1.bag

#* 测试pplio_sc
# cd /media/zhujun/0DFD06D20DFD06D2/benchmark_ws
# source devel/setup.bash
# roslaunch aloam_velodyne  pplio_sc_tset.launch

#* NewerCollege
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection1_newer_college/quad_easy_003/*.bag #* 196
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection1_newer_college/quad_medium_004/*.bag #* 187
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection1_newer_college/quad_hard_001/*.bag #* 184
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection1_newer_college/stairs_002/*.bag #* 116
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection2_newer_college/*.bag #* 278
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection3_maths_institute/math_easy/*math-easy.bag #* 215
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection3_maths_institute/math_medium/*math-medium.bag #* 177
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection3_maths_institute/math_hard/*math-hard.bag #* 51
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection4_underground_mine/easy/*.bag #* 141
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection4_underground_mine/medium/*.bag #* 148
# rosbag play /media/zhujun/0DFD06D20DFD06D2/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/collection4_underground_mine/hard/*.bag #* 190

#* 读取数据  Hilti
# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp04_construction_upper_level/*.bag #* 125
# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp_05_construction_upper_level_2/*.bag #* 124
# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp06_construction_upper_level_3/*.bag #* 150
# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp14_basement_2/*.bag #* 74
# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp16_attic_to_upper_gallery_2/*.bag #* 200
# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp18_corridor_lower_gallery_2/*.bag #* gt 有问题 才86秒

# rosbag play /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp21_outside_building/exp21_outside_building.bag #* gt 有问题

# * 轨迹误差计算
# cd /home/zhujun/WS/UbuntuData/SLAM/Hilti/exp04_construction_upper_level && actpy py39
# python plot_time.py


























#* 距离误差显示
# cd /home/zhujun/WS/catkin_ws/src/PPLIO/output/single_frame/frame_to_octree && actpy py39
# python test.py



