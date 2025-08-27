cd /media/zhujun/0DFD06D20DFD06D2/ws_PPSLAM/

catkin_make -DCMAKE_BUILD_TYPE=Release -DCATKIN_WHITELIST_PACKAGES="splin" -j16 -DCMAKE_CXX_FLAGS="" # livox_ros_driver  splin  poslam_camera_model

#* 程序运行
# roscore
# cd /media/zhujun/0DFD06D20DFD06D2/ws_PPSLAM/src/SPLIN
# source devel/setup.bash
# export LD_LIBRARY_PATH=/media/zhujun/0DFD06D20DFD06D2/ws_LTAOM_orig/src/LTAOM/thirdparty/lib:$LD_LIBRARY_PATH

# roslaunch splin poslam_run.launch
# rosparam set /save_map true 
# roslaunch file_player file_player.launch
# roslaunch file_player_nclt file_player_nclt.launch


