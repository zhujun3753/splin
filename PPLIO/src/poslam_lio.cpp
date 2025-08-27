#include "splin.hpp"


void SPLIN::imu_cbk( const sensor_msgs::Imu::ConstPtr &msg_in )
{
    sensor_msgs::Imu::Ptr msg( new sensor_msgs::Imu( *msg_in ) );
    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();
    if ( timestamp < last_timestamp_imu )
    {
        ROS_ERROR( "imu loop back, clear buffer" );
        imu_buffer_lio.clear();
        flg_reset = true;
    }
    last_timestamp_imu = timestamp;
    // std::cout<<"get imu"<<std::endl;
    imu_buffer_lio.push_back( msg );
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void SPLIN::feat_points_cbk( const sensor_msgs::PointCloud2::ConstPtr &msg_in )
{
    sensor_msgs::PointCloud2::Ptr msg( new sensor_msgs::PointCloud2( *msg_in ) );
    msg->header.stamp = ros::Time( msg_in->header.stamp.toSec() - m_lidar_imu_time_delay );
    mtx_buffer.lock();
    // std::cout<<"got feature"<<std::endl;
    if ( msg->header.stamp.toSec() < last_timestamp_lidar && 1)
    {
        ROS_ERROR( "lidar loop back, clear buffer" );
        // std::cout<<"lidar_buffer.size(): "<<lidar_buffer.size()<<std::endl;
        lidar_buffer.clear();
    }
    lidar_buffer.push_back( msg );
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void SPLIN::set_initial_state_cov( StatesGroup &state )
{
    // std::cout<<"scope_color num_match_pts:"<<num_match_pts<< std::endl;
    // Set cov
    scope_color( ANSI_COLOR_RED_BOLD );
    // std::cout<<"ANSI_COLOR_RED_BOLD num_match_pts:"<<num_match_pts<< std::endl;
    state.cov = state.cov.setIdentity() * INIT_COV;
    // std::cout<<"INIT_COV num_match_pts:"<<num_match_pts<< std::endl;
    // state.cov.block(18, 18, 6 , 6 ) = state.cov.block(18, 18, 6 , 6 ) .setIdentity() * 0.1;
    // state.cov.block(24, 24, 5 , 5 ) = state.cov.block(24, 24, 5 , 5 ).setIdentity() * 0.001;
    state.cov.block( 0, 0, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // R
    state.cov.block( 3, 3, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // T
    state.cov.block( 6, 6, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // vel
    state.cov.block( 9, 9, 3, 3 ) = mat_3_3::Identity() * 1e-3;   // bias_g
    state.cov.block( 12, 12, 3, 3 ) = mat_3_3::Identity() * 1e-1; // bias_a
    state.cov.block( 15, 15, 3, 3 ) = mat_3_3::Identity() * 1e-5; // Gravity
    // state.cov( 24, 24 ) = 0.00001;
    // std::cout<<"00001 num_match_pts:"<<num_match_pts<< std::endl; // 越界而导致num_match_pts值出错！！！！
    // state.cov.block( 18, 18, 6, 6 ) = state.cov.block( 18, 18, 6, 6 ).setIdentity() *  1e-3; // Extrinsic between camera and IMU.
    // state.cov.block( 25, 25, 4, 4 ) = state.cov.block( 25, 25, 4, 4 ).setIdentity() *  1e-3; // Camera intrinsic.
    // std::cout<<"setIdentity num_match_pts:"<<num_match_pts<< std::endl;
}

void printf_field_name( sensor_msgs::PointCloud2::ConstPtr &msg )
{
    cout << "Input pointcloud field names: [" << msg->fields.size() << "]: ";
    for ( size_t i = 0; i < msg->fields.size(); i++ )
    {
        cout << msg->fields[ i ].name << ", ";
    }
    cout << endl;
}

bool SPLIN::get_pointcloud_data_from_ros_message( sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud< pcl::PointXYZINormal > &pcl_pc )
{

    pcl::PointCloud< pcl::PointXYZI > res_pc;
    scope_color( ANSI_COLOR_YELLOW_BOLD );
    // printf_field_name(msg);
    if ( msg->fields.size() < 3 )
    {
        cout << "Get pointcloud data from ros messages fail!!!\n" << endl;
        scope_color( ANSI_COLOR_RED_BOLD );
        printf_field_name( msg );
        return false;
    }
    else
    {
        if ( ( msg->fields.size() == 8 ) && ( msg->fields[ 3 ].name == "intensity" ) &&
             ( msg->fields[ 4 ].name == "normal_x" ) ) // Input message type is pcl::PointXYZINormal
        {
            pcl::fromROSMsg( *msg, pcl_pc );
            return true;
        }
        else if ( ( msg->fields.size() == 4 ) && ( msg->fields[ 3 ].name == "rgb" ) )
        {
            double maximum_range = 5;
            get_ros_parameter< double >( m_ros_node_handle, "iros_range", maximum_range, 5 );
            pcl::PointCloud< pcl::PointXYZRGB > pcl_rgb_pc;
            pcl::fromROSMsg( *msg, pcl_rgb_pc );
            double lidar_point_time = msg->header.stamp.toSec();
            int pt_count = 0;
            pcl_pc.resize( pcl_rgb_pc.points.size() );
            for ( int i = 0; i < pcl_rgb_pc.size(); i++ )
            {
                pcl::PointXYZINormal temp_pt;
                temp_pt.x = pcl_rgb_pc.points[ i ].x;
                temp_pt.y = pcl_rgb_pc.points[ i ].y;
                temp_pt.z = pcl_rgb_pc.points[ i ].z;
                double frame_dis = sqrt( temp_pt.x * temp_pt.x + temp_pt.y * temp_pt.y + temp_pt.z * temp_pt.z );
                if ( frame_dis > maximum_range )
                {
                    continue;
                }
                temp_pt.intensity = ( pcl_rgb_pc.points[ i ].r + pcl_rgb_pc.points[ i ].g + pcl_rgb_pc.points[ i ].b ) / 3.0;
                temp_pt.curvature = 0;
                pcl_pc.points[ pt_count ] = temp_pt;
                pt_count++;
            }
            pcl_pc.points.resize( pt_count );
            return true;
        }
        else // TODO, can add by yourself
        {
            cout << "Get pointcloud data from ros messages fail!!! ";
            scope_color( ANSI_COLOR_RED_BOLD );
            printf_field_name( msg );
            exit(1);
            return false;
        }
    }
}

bool SPLIN::sync_packages( MeasureGroup &meas )
{
    if ( lidar_buffer.empty() || imu_buffer_lio.empty())
    {
        return false;
    }
    if ( !lidar_pushed )
    {
        meas.lidar.reset( new PointCloudXYZINormal() );
        if ( get_pointcloud_data_from_ros_message( lidar_buffer.front(), *( meas.lidar ) ) == false )
        {
            return false;
        }
        meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
        meas.lidar_end_time = lidar_end_time;
        meas.lidar_type = lidar_buffer.front()->header.frame_id;
        lidar_pushed = true;
    }
    if ( last_timestamp_imu < lidar_end_time ) // 等待imu数据覆盖点云时间段
    {
        return false;
    }
    double imu_time = imu_buffer_lio.front()->header.stamp.toSec();
    meas.imu.clear();
    while ( ( !imu_buffer_lio.empty() ) && ( imu_time < lidar_end_time ) )
    {
        imu_time = imu_buffer_lio.front()->header.stamp.toSec();
        meas.imu.push_back( imu_buffer_lio.front() );
        if ( imu_time > lidar_end_time ) // 可以适当大于雷达时间，但是不必弹出最后一个IMU数据
            break;
        imu_buffer_lio.pop_front();
    }
    lidar_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int SPLIN::service_LIO_update()
{
    // std::cout<<"service_LIO_update num_match_pts:"<<num_match_pts<< std::endl;
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "/world";
    /*** variables definition ***/
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE, Jh_inv, Jh, H_T_R_inv_H, P_inv;
    G.setZero();
    H_T_H.setZero();
    H_T_R_inv_H.setZero(); // H^T * R^{-1} * H
    I_STATE.setIdentity();
    Jh_inv.setIdentity();
    Jh.setIdentity();
    P_inv.setIdentity();

    cv::Mat matA1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matD1( 1, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matV1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matP( 6, 6, CV_32F, cv::Scalar::all( 0 ) );

    PointCloudXYZINormal::Ptr feats_undistort( new PointCloudXYZINormal() );
    PointCloudXYZINormal::Ptr feats_down( new PointCloudXYZINormal() );
    PointCloudXYZINormal::Ptr laserCloudOri( new PointCloudXYZINormal() );
    PointCloudXYZINormal::Ptr coeffSel( new PointCloudXYZINormal() );
    /*** variables initialize ***/
    //------------------------------------------------------------------------------------------------------
    ros::Rate rate( 5000 );
    bool status = ros::ok();
    set_initial_state_cov( g_lio_state );
    double first_lidar_time = -1.0;
    bool calib_flag = true;
    std::string alg_name="d-po-lio";
    // std::string alg_name="poslam_ikd_skip4";
    {
        std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+".txt", std::ios::out);
        log_file<<"# timestamp x y z q.x q.y q.z q.w"<<std::endl;
        log_file.close();
    }
    // {
    //     std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+"_imu.txt", std::ios::out);
    //     log_file<<"# timestamp x y z q.x q.y q.z q.w"<<std::endl;
    //     log_file.close();
    // }
    // {
    //     std::ofstream log_file(m_map_output_dir + "/bias_"+alg_name+".txt", std::ios::out);
    //     log_file<<"# timestamp bax bay baz bgx bgy bgz"<<std::endl;
    //     log_file.close();
    // }
    {
        std::ofstream log_file(m_map_output_dir + "/exc_time_"+alg_name+".txt", std::ios::out);
        log_file<<"# timestamp delta_t_per_frame"<<std::endl;
        log_file.close();
    }
    {
        std::ofstream log_file(m_map_output_dir + "/feats_down_size_"+alg_name+".txt", std::ios::out);
        log_file<<"# timestamp feats_down_size"<<std::endl;
        log_file.close();
    }
    bool flg_EKF_converged = 0;
    int frame_num = 0;
    double frame_first_pt_time=0.0;
    cv::imshow( "Control panel", generate_control_panel_img().clone() );
    int last_key_frame_size = 0;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr octree_data(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    double last_feats_down_size = lio_pts_num;
    double lio_pts_num_th = lio_pts_num*0.2;
    double voxel_ds_size = m_voxel_downsample_size_surf;
    bool not_saved_octree = true;
    while( ros::ok() && !process_end)
    {
        cv_keyboard_callback();
        ros::spinOnce();
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        std::unique_lock< std::mutex > lock( m_mutex_lio_process );
        if(wait_for_save_data && !not_saved_octree) continue;
        // printf_line;
        double start_t,end_t;
        start_t = omp_get_wtime();
        // ====== 1. 寻找与雷达匹配的imu ======
        if( sync_packages( Measures ) == 0 ) continue;
        if( flg_reset )
        {
            ROS_WARN( "reset when rosbag play back" );
            m_imu_process->Reset();
            flg_reset = false;
            continue;
        }
        // continue;
        if(first_lidar_time<0) first_lidar_time = Measures.lidar_beg_time;
        g_LiDAR_frame_index++;
        // std::cout<<"g_LiDAR_frame_index: "<<g_LiDAR_frame_index<<std::endl;
        StatesGroup last_state(g_lio_state);
        // ====== 2. imu传播与将雷达点全部投影到最后一个点所在的局部坐标系 ======
        m_imu_process->Process( Measures, g_lio_state, feats_undistort );
        double imu_process_t = omp_get_wtime();
        g_lio_state.lidar_type = Measures.lidar_type;
        StatesGroup state_imu_propagate(g_lio_state);
        if( feats_undistort->empty() || ( feats_undistort == NULL ) )
        {
            frame_first_pt_time = Measures.lidar_beg_time;
            std::cout << "not ready for odometry" << std::endl;
            continue;
        }
        if( ( Measures.lidar_beg_time - frame_first_pt_time ) < 0 ) // 雷达时间异常
        {
            std::cout << "||||||||||LiDAR disorder||||||||||" << std::endl;
            frame_first_pt_time = Measures.lidar_beg_time;
            std::cout << "not ready for odometry" << std::endl;
            continue;
        }
        LidarFrame * lidar_frame = new LidarFrame(feats_undistort);
        lidar_frame->lidar_end_time = Measures.lidar_end_time;
        lidar_frame->set_plane_merge_param(plane_merge_angle, plane_merge_dist, max_merge_plane_n, min_plane_pts_n, max_eigen_dist_th);
        lidar_frame->frame_id = all_lidar_frames.size();
        std::vector<Eigen::Vector3d> orig_pts_down_eigen;
        // 动态调整下采样网格大小
        if(abs(last_feats_down_size-lio_pts_num)>lio_pts_num_th && 0)
        {
            double scale_ = last_feats_down_size/lio_pts_num-1;
            if(scale_>1) scale_=1;
            if(scale_<-1) scale_=-1;
            voxel_ds_size +=  0.1*scale_;
            if(voxel_ds_size>0.5) voxel_ds_size=0.5;
            if(voxel_ds_size<0.1) voxel_ds_size=0.1;
        }
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down);
        for (int i=0; i<feats_down->points.size(); i++)
        {
            orig_pts_down_eigen.emplace_back(Eigen::Vector3d(feats_down->points[i].x, feats_down->points[i].y, feats_down->points[i].z));
        }
        // pc_voxel_filter_eigen(*feats_undistort, orig_pts_down_eigen, voxel_ds_size, false); // 0.00152921
        double downSizeFilter_t = omp_get_wtime();
        int feats_down_size = orig_pts_down_eigen.size();
        lidar_frame->feats_down_size_ = feats_down_size;
        last_feats_down_size = (double)feats_down_size;
        // 初始化八叉树
        if(feats_down_size > 5 && octree.size()==0)
        {
            std::vector<Eigen::Vector3d> orig_pts_down_eigen_updated = orig_pts_down_eigen;
            for( int i = 0; i < feats_down_size; i++ )
            {
                orig_pts_down_eigen_updated[i] = m_lidar_to_imu_R*orig_pts_down_eigen[i] + m_lidar_to_imu_t;
            }
            std::vector<float> extra_attr;
            octree.set_min_extent(0.25); // 网格内接圆半径
            octree.set_bucket_size(1);
            octree.initialize_with_attr(orig_pts_down_eigen_updated, extra_attr);
            std::cout << "================initialize the map octree===================" << std::endl;
            std::cout<<"octree.size(): "<<octree.size()<<std::endl;
            std::cout<<"init g_lio_state.pos_end: "<<g_lio_state.pos_end.transpose()<<std::endl;
            if(octree.size()==0)
            {
                std::cout << "~~~~~~~ Initialize octree Failed! ~~~~~~~" << std::endl;
                continue;
            }
            lidar_frame->R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
            lidar_frame->t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
            lidar_frame->R_w_l_loop = lidar_frame->R_w_l;
            lidar_frame->t_w_l_loop = lidar_frame->t_w_l;
            // 由IMU位姿转到雷达的位姿带来的协方差变化 扰动方式：简单相减 \delta_theta = Log((\check{R}^w_l)^T*R^w_l)  \delta t = t^w_l - \check{t}^w_l
            Eigen::Matrix<double, 6,6> W_ = Eigen::Matrix<double, 6,6>::Identity();
            W_.block<3,3>(0,0) = m_lidar_to_imu_R.transpose();
            W_.block<3,3>(3,0) = -g_lio_state.rot_end * vec_to_hat(m_lidar_to_imu_t);
            // 扰动方式变化带来的协方差变化 简单相减==>左乘扰动
            Eigen::Matrix<double, 6,6> W_1 = Eigen::Matrix<double, 6,6>::Identity();
            W_1.block<3,3>(0,0) = lidar_frame->R_w_l;
            W_1.block<3,3>(3,0) = vec_to_hat(lidar_frame->t_w_l) * lidar_frame->R_w_l;
            Eigen::Matrix<double, 6,6> W_2 = W_1 * W_;
            lidar_frame->cov = W_2*g_lio_state.get_imu_pose_cov().block<6,6>(0,0)*W_2.transpose(); // IMU位姿协方差转为雷达位姿左乘扰动协方差
            all_lidar_frames.emplace_back(lidar_frame);
            continue;
        }
        // std::cout<<"feats_down_size: "<<feats_down_size<<std::endl;
        // std::cout<<"start g_lio_state.pos_end: "<<g_lio_state.pos_end.transpose()<<std::endl;
        double vec_init_t = omp_get_wtime();
        std::vector< bool > point_selected_surf( feats_down_size, false );
        std::vector< bool > plane_valid_flags( feats_down_size, false );
        std::vector<Eigen::Vector3d> plane_normals(feats_down_size); // 平面参数
        std::vector<double> plane_ds(feats_down_size); // 平面参数
        std::vector<double> pt2plane_dists(feats_down_size); // 平面参数
        std::vector<double> pts_weights(feats_down_size);
        int rematch_num = 0;
        bool rematch_en = 0;
        flg_EKF_converged = 0;
        double deltaT = 0.0, deltaR = 0.0;
        double ready_t = omp_get_wtime();
        double solve_time = 0, match_time=0.0, kd_tree_search_time=0.0;
        for( int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ )
        {
            // std::cout<<"iterCount: "<<iterCount<< std::endl;
            double match_start = omp_get_wtime();
            Eigen::Matrix3d R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
            Eigen::Vector3d t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
            int valid_lidar_pts_n = 0;
            // 计算每个点对应的平面法线和有符号距离
            // #ifdef MP_EN // 需要在主线程中启动，在子线程中速度特别慢！！！
            //     omp_set_num_threads(MP_PROC_NUM);
            // #pragma omp parallel for
            // #endif
            for( int i = 0; i < feats_down_size; i += 1 )
            {
                // std::cout<<"i:"<<i<< std::endl;
                // 转到世界坐标系
                Eigen::Vector3d p_global( R_w_l*orig_pts_down_eigen[i] + t_w_l);
                point_selected_surf[i] = false;
                // std::cout<<"num_match_pts:"<<num_match_pts<< std::endl;
                double kd_search_start = omp_get_wtime();
                // 通过树搜索，拟合拟合当前点最近的平面
                if( iterCount == 0 || rematch_en )
                {
                    std::vector< float > pointSearchSqDis_surf;
                    std::vector<std::vector<float>> points_near;
                    plane_valid_flags[ i ] = false;
                    // 寻找5个最近点
                    octree.knnNeighbors_eigen( p_global, num_match_pts, points_near, pointSearchSqDis_surf );
                    float max_distance = pointSearchSqDis_surf[ num_match_pts - 1 ];
                    if( max_distance > 5.0 ) continue; //  超过0.5就无效了
                    // 平面拟合
                    {
                        cv::Mat matA0( num_match_pts, 3, CV_32F, cv::Scalar::all( 0 ) );
                        cv::Mat matB0( num_match_pts, 1, CV_32F, cv::Scalar::all( -1 ) );
                        cv::Mat matX0( num_match_pts, 1, CV_32F, cv::Scalar::all( 0 ) );
                        for( int j = 0; j < num_match_pts; j++ )
                        {
                            matA0.at< float >( j, 0 ) = points_near[j][0];
                            matA0.at< float >( j, 1 ) = points_near[j][1];
                            matA0.at< float >( j, 2 ) = points_near[j][2];
                        }
                        cv::solve( matA0, matB0, matX0, cv::DECOMP_QR ); // TODO
                        Eigen::Vector3d normal_fit;
                        normal_fit << matX0.at< float >( 0, 0 ), matX0.at< float >( 1, 0 ), matX0.at< float >( 2, 0 );
                        double norm_ = normal_fit.norm()+1e-6;
                        float pd = 1/norm_;
                        normal_fit /= norm_;
                        bool planeValid = true;
                        // 检查拟合平面的好坏，距离超过0.1就无效
                        for( int j = 0; j < num_match_pts; j++ )
                        {
                            float dist = fabs( normal_fit[0] * points_near[ j ][0] + normal_fit[1] * points_near[ j ][1] + normal_fit[2] * points_near[ j ][2] + pd );
                            if ( dist > m_planar_check_dis ) // Raw 0.10
                            {
                                planeValid = false;
                                break;
                            }
                        }
                        if(!planeValid) continue;
                        plane_normals[i] = normal_fit;
                        plane_ds[i] = pd;
                        plane_valid_flags[ i ] = true;
                    }
                }
                kd_tree_search_time += omp_get_wtime() - kd_search_start;
                if(plane_valid_flags[i])
                {
                    // 当前点到拟合平面的距离
                    double pd2 = p_global.dot(plane_normals[i]) + plane_ds[i];
                    // std::cout<<"fit pd2: "<<pd2<<std::endl;
                    // if( fabs(pd2) < 0.3-0.3/(1+exp(-0.5*iterCount)) || all_lidar_frames.size()<3) // 考虑一个随着时间减小的值
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(orig_pts_down_eigen[i].norm());
                    if (s > 0.9)
                    {
                        point_selected_surf[i] = true;
                        pts_weights[i] = lidar_cov_p;
                        pt2plane_dists[i] = pd2;
                        valid_lidar_pts_n++;
                    }
                }
            }
            // std::cout<<"rematch_en: "<<rematch_en<<", iterCount: "<<iterCount<<", valid_lidar_pts_n: "<<valid_lidar_pts_n<<std::endl;
            match_time += omp_get_wtime() - match_start;
            double solve_start = omp_get_wtime();
            // 计算雅克比矩阵和测量向量
            Eigen::MatrixXd Hsub( valid_lidar_pts_n, 6 );
            Eigen::VectorXd meas_vec( valid_lidar_pts_n );
            Eigen::MatrixXd H_T_R_inv( 6, valid_lidar_pts_n ); // H^T* R^{-1}
            Hsub.setZero();
            H_T_R_inv.setZero();
            int laserCloudSelNum_i = 0;
            for( int i = 0; i < feats_down_size; i++ )
            {
                if(!point_selected_surf[i]) continue;
                // 将点从雷达坐标系转换到IMU坐标系，因为不存在旋转，向量测量向量就不转换了
                Eigen::Vector3d point_this = m_lidar_to_imu_R*orig_pts_down_eigen[i] + m_lidar_to_imu_t;
                Eigen::Matrix3d point_crossmat;
                point_crossmat << SKEW_SYM_MATRIX( point_this );
                //* 转置，而point_crossmat没转置，就是添加负号！！
                Eigen::Vector3d A = point_crossmat * g_lio_state.rot_end.transpose() * plane_normals[i];
                Hsub.block<1, 6>(laserCloudSelNum_i, 0)  << A[0], A[1], A[2], plane_normals[i][0], plane_normals[i][1], plane_normals[i][2];
                meas_vec( laserCloudSelNum_i ) = -pt2plane_dists[i];
                double lidar_dist_cov = lidar_pt_cov/pts_weights[i]; // 在这里改变每个距离误差的协方差
                H_T_R_inv.block<6, 1>(0, laserCloudSelNum_i) = Hsub.block<1, 6>(laserCloudSelNum_i, 0).transpose()/lidar_dist_cov;
                laserCloudSelNum_i++;
            }
            // 迭代拓展卡尔曼滤波
            Eigen::MatrixXd K( DIM_OF_STATES, valid_lidar_pts_n );
            H_T_R_inv_H.block<6,6>(0,0) = H_T_R_inv * Hsub;
            K = ( g_lio_state.cov.inverse() + H_T_R_inv_H ).inverse().block< DIM_OF_STATES, 6 >( 0, 0 ) * H_T_R_inv;
            auto vec = state_imu_propagate - g_lio_state;
            Eigen::Matrix< double, DIM_OF_STATES, 1 > solution = K * ( meas_vec - Hsub * vec.block< 6, 1 >( 0, 0 ) );
            g_lio_state = state_imu_propagate + solution;
            Eigen::Vector3d rot_add = solution.block< 3, 1 >( 0, 0 );
            Eigen::Vector3d t_add = solution.block< 3, 1 >( 3, 0 );
            flg_EKF_converged = false;
            if( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) )
                flg_EKF_converged = true;
            deltaR = rot_add.norm() * 57.3;
            deltaT = t_add.norm() * 100;
            g_lio_state.last_update_time = Measures.lidar_end_time;
            rematch_en = false;
            if( flg_EKF_converged || ( ( rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) )
            {
                rematch_en = true;
                rematch_num++;
            }
            if( rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) // Fast lio ori version.
            {
                G.block< DIM_OF_STATES, 6 >( 0, 0 ) = K * Hsub;
                g_lio_state.cov = ( I_STATE - G ) * g_lio_state.cov;
                break;
            }
            // std::cout<<"solve_time"<<std::endl;
            solve_time += omp_get_wtime() - solve_start;
        }
        double t1 = omp_get_wtime();
        // 树更新
        {
            Eigen::Matrix3d R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
            Eigen::Vector3d t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
            std::vector<Eigen::Vector3d> orig_pts_down_eigen_updated = orig_pts_down_eigen;
            std::vector<float> extra_attr;
            for ( int i = 0; i < feats_down_size; i++ )
            {
                orig_pts_down_eigen_updated[i] = R_w_l*orig_pts_down_eigen[i] + t_w_l;
            }
            octree.update_with_attr( orig_pts_down_eigen_updated, extra_attr, true);
        }
        lidar_frame->R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
        lidar_frame->t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
        lidar_frame->R_w_l_loop = lidar_frame->R_w_l;
        lidar_frame->t_w_l_loop = lidar_frame->t_w_l;
        // 由IMU位姿转到雷达的位姿带来的协方差变化 扰动方式：简单相减 \delta_theta = Log((\check{R}^w_l)^T*R^w_l)  \delta t = t^w_l - \check{t}^w_l
        Eigen::Matrix<double, 6,6> W_ = Eigen::Matrix<double, 6,6>::Identity();
        W_.block<3,3>(0,0) = m_lidar_to_imu_R.transpose();
        W_.block<3,3>(3,0) = -g_lio_state.rot_end * vec_to_hat(m_lidar_to_imu_t);
        // 扰动方式变化带来的协方差变化 简单相减==>左乘扰动
        Eigen::Matrix<double, 6,6> W_1 = Eigen::Matrix<double, 6,6>::Identity();
        W_1.block<3,3>(0,0) = lidar_frame->R_w_l;
        W_1.block<3,3>(3,0) = vec_to_hat(lidar_frame->t_w_l) * lidar_frame->R_w_l;
        Eigen::Matrix<double, 6,6> W_2 = W_1 * W_;
        lidar_frame->cov = W_2*g_lio_state.get_imu_pose_cov().block<6,6>(0,0)*W_2.transpose(); // IMU位姿协方差转为雷达位姿左乘扰动协方差
        all_lidar_frames.emplace_back(lidar_frame);
		if(frame_num%10==0) cout << "Lidar time " << Measures.lidar_end_time - first_lidar_time <<", feats_down_size: "<<feats_down_size << endl;
        // if(all_lidar_frames.size()>=100) all_lidar_frames[all_lidar_frames.size()-100]->clear_unnecessary_data();
        end_t = omp_get_wtime();
        // std::cout<<"octree.size(): "<<octree.size()<<std::endl; // 0.0130674
        if(0)
        {
            std::cout<< std::fixed << std::setprecision(7);
            std::cout<<"\ng_LiDAR_frame_index: "<<g_LiDAR_frame_index<<", feats_down_size: "<<feats_down_size<<std::endl;
            std::cout<<"imu_process_time: "<<imu_process_t - start_t<<std::endl; // 0.000894163
            std::cout<<"downSizeFilter_time: "<<downSizeFilter_t - imu_process_t<<std::endl; // 0.000967148
            std::cout<<"vec_init: "<<ready_t - vec_init_t<<std::endl; // 0.000967148
            std::cout<<"ready_time: "<<ready_t-start_t<<std::endl; // 0.0019098
            std::cout<<"solve_time: "<<solve_time<<std::endl; // 0.000392439
            std::cout<<"match_time: "<<match_time<<std::endl; // 0.00243819
            std::cout<<"kd_tree_search_time: "<<kd_tree_search_time<<std::endl; // 0.00243819
            std::cout<<"kd tree update time: "<<end_t-t1<<std::endl; // 0.000344489
            std::cout<<"total time: "<<end_t-start_t<<std::endl<<std::endl; // 0.00530818
        }
        // exit(1);
        // continue;
        // std::cout<<"end g_lio_state.pos_end: "<<g_lio_state.pos_end.transpose()<<std::endl;
        
        {
            std::ofstream log_file(m_map_output_dir + "/feats_down_size_"+alg_name+".txt", std::ios::app);
            log_file    <<std::fixed << std::setprecision(7)
                        <<Measures.lidar_end_time<<" " // lidar_beg_time
                        <<feats_down_size
                        <<std::endl;
            log_file.close();
        }
        {
            std::ofstream log_file(m_map_output_dir + "/exc_time_"+alg_name+".txt", std::ios::app);
            log_file    <<std::fixed << std::setprecision(7)
                        <<Measures.lidar_end_time<<" " // lidar_beg_time
                        <<end_t-start_t
                        <<std::endl;
            log_file.close();
        }
        // {
        //     std::ofstream log_file(m_map_output_dir + "/bias_"+alg_name+".txt", std::ios::app);
        //     log_file    << std::fixed << std::setprecision(7)
        //                 <<Measures.lidar_end_time<<" " // lidar_beg_time
        //                 <<g_lio_state.bias_a[0]<<" "<<g_lio_state.bias_a[1]<<" "<<g_lio_state.bias_a[2]<<" "
        //                 <<g_lio_state.bias_g[0]<<" "<<g_lio_state.bias_g[1]<<" "<<g_lio_state.bias_g[2]
        //                 <<std::endl;
        //     log_file.close();
        // }
        {
            Eigen::Quaterniond qr(g_lio_state.rot_end);
            Eigen::Vector3d position = g_lio_state.pos_end;
            std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+".txt", std::ios::app);
            log_file    << std::fixed << std::setprecision(7)
                        <<Measures.lidar_end_time<<" " // lidar_beg_time
                        <<position[0]<<" "<<position[1]<<" "<<position[2]<<" "
                        <<qr.x()<<" "<<qr.y()<<" "<<qr.z()<<" "<<qr.w()
                        <<std::endl;
            log_file.close();
        }
        // {
        //     Eigen::Quaterniond qr(state_imu_propagate.rot_end);
        //     Eigen::Vector3d position = state_imu_propagate.pos_end;
        //     std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+"_imu.txt", std::ios::app);
        //     log_file    << std::fixed << std::setprecision(7)
        //                 <<Measures.lidar_end_time<<" " // lidar_beg_time
        //                 <<position[0]<<" "<<position[1]<<" "<<position[2]<<" "
        //                 <<qr.x()<<" "<<qr.y()<<" "<<qr.z()<<" "<<qr.w()
        //                 <<std::endl;
        //     log_file.close();
        // }
        /******* Publish current frame points in world coordinates:  *******/
        PointCloudXYZINormal::Ptr laserCloudFullRes2( new PointCloudXYZINormal() );
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor( new pcl::PointCloud<pcl::PointXYZI>() );
        laserCloudFullRes2->clear();
        // *laserCloudFullRes2 = dense_map_en ? ( *feats_undistort ) : ( *feats_down );
        *laserCloudFullRes2 = *feats_undistort;
        int laserCloudFullResNum = laserCloudFullRes2->points.size();
        pcl::PointXYZI temp_point;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_color(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        laserCloudFullResColor->clear();
        {
            Eigen::Matrix3d R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
            Eigen::Vector3d t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
            std::vector<unsigned int> colors;
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            for ( int i = 0; i < laserCloudFullResNum; i++ )
            {
                Eigen::Vector3d p_body( laserCloudFullRes2->points[i].x, laserCloudFullRes2->points[i].y, laserCloudFullRes2->points[i].z );
                // Eigen::Vector3d p_global( g_lio_state.rot_end * ( m_lidar_to_imu_R*p_body + m_lidar_to_imu_t ) + g_lio_state.pos_end );
                Eigen::Vector3d p_global( R_w_l*p_body + t_w_l);
                temp_point.x = p_global( 0 );
                temp_point.y = p_global( 1 );
                temp_point.z = p_global( 2 );
                temp_point.intensity = laserCloudFullRes2->points[i].intensity;
                laserCloudFullRes2->points[i].x = p_global( 0 );
                laserCloudFullRes2->points[i].y = p_global( 1 );
                laserCloudFullRes2->points[i].z = p_global( 2 );
                if(laserCloudFullRes2->points[ i ].curvature == -2)
                {
                    Eigen::Vector3d normal_body( laserCloudFullRes2->points[ i ].normal_x, laserCloudFullRes2->points[ i ].normal_y, laserCloudFullRes2->points[ i ].normal_z );
                    Eigen::Vector3d normal_global = R_w_l*normal_body;
                    normal_global /= (1-normal_global.transpose()*t_w_l);
                    laserCloudFullRes2->points[ i ].normal_x = normal_global( 0 );
                    laserCloudFullRes2->points[ i ].normal_y = normal_global( 1 );
                    laserCloudFullRes2->points[ i ].normal_z = normal_global( 2 );
                }
                laserCloudFullResColor->push_back( temp_point );
                if(1)
                {
                    pcl::PointXYZRGBNormal pt;
                    pt.x = laserCloudFullRes2->points[i].x;
                    pt.y = laserCloudFullRes2->points[i].y;
                    pt.z = laserCloudFullRes2->points[i].z;
                    Eigen::Vector3f normal(laserCloudFullRes2->points[i].normal_x,laserCloudFullRes2->points[i].normal_y,laserCloudFullRes2->points[i].normal_z);
                    normal = normal.norm()>0 ? normal/normal.norm() : normal;
                    pt.normal_x = normal(0);
                    pt.normal_y = normal(1);
                    pt.normal_z = normal(2);
                    pt.curvature = laserCloudFullRes2->points[i].curvature;
                    pt.r = colors[0];
                    pt.g = colors[1];
                    pt.b = colors[2];
                    cloud_color->push_back(pt);
                }
            }
            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
            // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
            laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time );
            laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
            pubLaserCloudFullRes.publish( laserCloudFullRes3 );
        }

        recent_ten_frames[(recent_ten_frames_i++)%recent_ten_frames_N] = *cloud_color;
        /******* Publish Odometry ******/
        {
            Eigen::Vector3d euler_cur = RotMtoEuler( g_lio_state.rot_end );
            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw( euler_cur( 0 ), euler_cur( 1 ), euler_cur( 2 ) );
            nav_msgs::Odometry odomAftMapped;
            odomAftMapped.header.frame_id = "world";
            odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
            odomAftMapped.pose.pose.orientation.x = geoQuat.x;
            odomAftMapped.pose.pose.orientation.y = geoQuat.y;
            odomAftMapped.pose.pose.orientation.z = geoQuat.z;
            odomAftMapped.pose.pose.orientation.w = geoQuat.w;
            odomAftMapped.pose.pose.position.x = g_lio_state.pos_end( 0 );
            odomAftMapped.pose.pose.position.y = g_lio_state.pos_end( 1 );
            odomAftMapped.pose.pose.position.z = g_lio_state.pos_end( 2 );
            pubOdomAftMapped.publish( odomAftMapped );
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3( odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z ) );
            q.setW( odomAftMapped.pose.pose.orientation.w );
            q.setX( odomAftMapped.pose.pose.orientation.x );
            q.setY( odomAftMapped.pose.pose.orientation.y );
            q.setZ( odomAftMapped.pose.pose.orientation.z );
            transform.setRotation( q );
            br.sendTransform( tf::StampedTransform( transform, ros::Time().fromSec( Measures.lidar_end_time ), "world", "/aft_mapped" ) );
            geometry_msgs::PoseStamped msg_body_pose;
            msg_body_pose.header.stamp = ros::Time::now();
            msg_body_pose.header.frame_id = "/camera_odom_frame";
            msg_body_pose.pose.position.x = g_lio_state.pos_end( 0 );
            msg_body_pose.pose.position.y = g_lio_state.pos_end( 1 );
            msg_body_pose.pose.position.z = g_lio_state.pos_end( 2 );
            msg_body_pose.pose.orientation.x = geoQuat.x;
            msg_body_pose.pose.orientation.y = geoQuat.y;
            msg_body_pose.pose.orientation.z = geoQuat.z;
            msg_body_pose.pose.orientation.w = geoQuat.w;
            /******* Publish Path ********/
            msg_body_pose.header.frame_id = "world";
            if ( frame_num > 10 )
            {
                path.poses.push_back( msg_body_pose );
            }
            pubPath.publish( path );
            frame_num++;
        }
        // std::cout<<"post time: "<<omp_get_wtime()-start_t<<std::endl<<std::endl; // 0.00530818
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}

int SPLIN::service_LIO_update_plane()
{
    // std::cout<<"service_LIO_update num_match_pts:"<<num_match_pts<< std::endl;
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "/world";
    /*** variables definition ***/
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE, Jh_inv, Jh, H_T_R_inv_H, P_inv;
    G.setZero();
    H_T_H.setZero();
    H_T_R_inv_H.setZero(); // H^T * R^{-1} * H
    I_STATE.setIdentity();
    Jh_inv.setIdentity();
    Jh.setIdentity();
    P_inv.setIdentity();
    cv::Mat matA1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matD1( 1, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matV1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matP( 6, 6, CV_32F, cv::Scalar::all( 0 ) );
    PointCloudXYZINormal::Ptr feats_undistort( new PointCloudXYZINormal() );
    /*** variables initialize ***/
    //------------------------------------------------------------------------------------------------------
    ros::Rate rate( 5000 );
    bool status = ros::ok();
    // std::cout<<"set_initial_state_cov num_match_pts:"<<num_match_pts<< std::endl;
    set_initial_state_cov( g_lio_state );
    double first_lidar_time = -1.0;
    bool calib_flag = true;
    std::string alg_name="d-pl-lio";
    // std::string alg_name="poslam_ikd_skip4";
    // std::cout<<"calib_flag num_match_pts:"<<num_match_pts<< std::endl;
    {
        std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+".txt", std::ios::out);
        log_file<<"# timestamp x y z q.x q.y q.z q.w"<<std::endl;
        log_file.close();
    }
    if(0)
    {
        std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+"_imu.txt", std::ios::out);
        log_file<<"# timestamp x y z q.x q.y q.z q.w"<<std::endl;
        log_file.close();
    }
    if(0)
    {
        std::ofstream log_file(m_map_output_dir + "/bias_"+alg_name+".txt", std::ios::out);
        log_file<<"# timestamp bax bay baz bgx bgy bgz"<<std::endl;
        log_file.close();
    }
    {
        std::ofstream log_file(m_map_output_dir + "/exc_time_ppsam.txt", std::ios::out);
        log_file<<"# timestamp delta_t_per_frame"<<std::endl;
        log_file.close();
    }
    {
        std::ofstream log_file(m_map_output_dir + "/feats_down_size_"+alg_name+".txt", std::ios::out);
        log_file<<"# timestamp plane_pts_ds_n"<<std::endl;
        log_file.close();
    }
    bool flg_EKF_converged = 0;
    int frame_num = 0;
    double frame_first_pt_time=0.0;
    cv::imshow( "Control panel", generate_control_panel_img().clone() );
    octree_feature_ptr->set_min_extent(tree_ds_size);
    octree_feature_ptr->set_bucket_size(1);
    octree_feature_replace_ptr->set_min_extent(tree_ds_size);
    octree_feature_replace_ptr->set_bucket_size(1);
    // std::string alg_name="poslam_ikd_skip4";
    // std::cout<<"calib_flag num_match_pts:"<<num_match_pts<< std::endl;
    int last_key_frame_size = 0;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr octree_data(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    double last_plane_pts_ds_n = lio_pts_num;
    double last_non_plane_pts_ds_n = lio_pts_num;
    double lio_pts_num_th = lio_pts_num*0.2;
    double plane_voxel_ds_size = m_voxel_downsample_size_surf;
    double non_plane_voxel_ds_size = m_voxel_downsample_size_surf;
    bool not_saved_octree = true;
    bool use_octree_delay = false;
    bool first_loop = false;
    double path_length = 0.0;
    double last_start_time = 0;
    bool pca_filter_flag = false; // 是否在pca的时候就滤波
    if(decouple_front) pca_filter_flag = true;
    while ( ros::ok() && !process_end)
    {
        cv_keyboard_callback();
        ros::spinOnce();
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        std::unique_lock< std::mutex > lock( m_mutex_lio_process );
        if(wait_for_save_data && !not_saved_octree) continue;
        // printf_line;
        double start_t,end_t;
        start_t = omp_get_wtime();
        Common_tools::Timer tim;
        // ====== 1. 寻找与雷达匹配的imu ======
        if ( sync_packages( Measures ) == 0 ) continue;
        if ( flg_reset )
        {
            ROS_WARN( "reset when rosbag play back" );
            m_imu_process->Reset();
            flg_reset = false;
            continue;
        }
        // std::cout<<"lidar_buffer.size(): "<<lidar_buffer.size()<<", imu_buffer_lio.size(): "<<imu_buffer_lio.size()<<std::endl;
        // continue;
        last_start_time = start_t;
        if(first_lidar_time<0) first_lidar_time = Measures.lidar_beg_time;
        g_LiDAR_frame_index++;
        // std::cout<<"\ng_LiDAR_frame_index: "<<g_LiDAR_frame_index<<std::endl;
        // if(g_LiDAR_frame_index%10==0)
        // std::cout<<"Lidar time: "<<Measures.lidar_beg_time-first_lidar_time<<", octree_feature_ptr: "<<octree_feature_ptr->size()<<", all_key_frames.size(): "<<all_key_frames.size()<<std::endl;
        tim.tic( "Preprocess" );
        if(large_scale_env_tree_state == 1) // 替换搜索树已经准备好，开始替换
        {
            // 替换搜索树
            std::cout<<"替换搜索树: "<<"octree_feature_ptr: "<<octree_feature_ptr->size()<<", octree_feature_replace_ptr: "<<octree_feature_replace_ptr->size()<<std::endl;
            thuni::Octree * octree_feature_ptr_tmp = octree_feature_ptr;
            octree_feature_ptr = octree_feature_replace_ptr;
            octree_feature_replace_ptr = octree_feature_ptr_tmp;
            large_scale_env_tree_state = -1;
        }
        // 回环检测优化之后，更新状态
        if(!decouple_front && loop_closure_optimized)
        {
            const int all_lidar_frames_n = all_lidar_frames.size(); // 在获取搜索树点之后再获取全部帧数量，避免新加入的帧没有被考虑
            std::cout<<"all_lidar_frames_n: "<<all_lidar_frames_n<<std::endl;
            std::cout<<"g_lio_state.pos_end: "<<(g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end).transpose()<<std::endl;
            std::cout<<"all_lidar_frames[all_lidar_frames_n-2]->t_w_l: "<<all_lidar_frames[all_lidar_frames_n-2]->t_w_l.transpose()<<std::endl;
            std::cout<<"all_lidar_frames[all_lidar_frames_n-1]->t_w_l: "<<all_lidar_frames[all_lidar_frames_n-1]->t_w_l.transpose()<<std::endl;
            std::cout<<"all_lidar_frames[all_lidar_frames_n-2]->t_w_l_loop: "<<all_lidar_frames[all_lidar_frames_n-2]->t_w_l_loop.transpose()<<std::endl;
            std::cout<<"all_lidar_frames[all_lidar_frames_n-1]->t_w_l_loop: "<<all_lidar_frames[all_lidar_frames_n-1]->t_w_l_loop.transpose()<<std::endl;
            // 更新 g_lio_state
            std::cout<<"更新 g_lio_state"<<std::endl;
            Eigen::Matrix4d T_w_j = Eigen::Matrix4d::Identity();
            T_w_j.block<3,3>(0,0) = g_lio_state.rot_end;
            T_w_j.block<3,1>(0,3) = g_lio_state.pos_end;
            Eigen::Matrix4d T_w_j_update = delta_T_loop*T_w_j;
            if(m_imu_process->imu_poses.size()>0)
            {
                IMUPose & last_imu_pose = m_imu_process->imu_poses.back();
                last_imu_pose.acc_imu = delta_T_loop.block<3,3>(0,0)*last_imu_pose.acc_imu;
                last_imu_pose.pos_imu = delta_T_loop.block<3,3>(0,0)*last_imu_pose.pos_imu + delta_T_loop.block<3,1>(0,3);
                last_imu_pose.vel_imu = delta_T_loop.block<3,3>(0,0)*last_imu_pose.vel_imu;
                last_imu_pose.R_imu = delta_T_loop.block<3,3>(0,0)*last_imu_pose.R_imu;
            }
            g_lio_state.rot_end = T_w_j_update.block<3,3>(0,0);
            g_lio_state.pos_end = T_w_j_update.block<3,1>(0,3);
            g_lio_state.vel_end = delta_T_loop.block<3,3>(0,0)*g_lio_state.vel_end;
            // 交换搜索树
            std::cout<<"交换搜索树"<<std::endl;
            thuni::Octree * octree_feature_ptr_tmp = octree_feature_ptr;
            octree_feature_ptr = octree_feature_replace_ptr;
            octree_feature_replace_ptr = octree_feature_ptr_tmp;
            // 更新可能漏掉的帧位姿
            std::cout<<"更新可能漏掉的帧位姿: ";
            int used_loop_id = loop_ids_used.size()-1;
            int loop_end_id = loop_ids_used[used_loop_id].second;
            for(int j=all_lidar_frames_n-1; j>loop_end_id; j--)
            {
                if(all_lidar_frames[j]->updated_loop_ids.size()>0) break; // 判断是否已经更新
                std::cout<<j<<", ";
                Eigen::Matrix4d T_w_j = Eigen::Matrix4d::Identity();
                T_w_j.block<3,3>(0,0) = all_lidar_frames[j]->R_w_l;
                T_w_j.block<3,1>(0,3) = all_lidar_frames[j]->t_w_l;
                Eigen::Matrix4d T_w_j_update = delta_T_loop*T_w_j;
                all_lidar_frames[j]->R_w_l_loop = T_w_j_update.block<3,3>(0,0);
                all_lidar_frames[j]->t_w_l_loop = T_w_j_update.block<3,1>(0,3);
                all_lidar_frames[j]->updated_loop_ids.push_back(used_loop_id);
            }
            std::cout<<std::endl;
            std::cout<<"g_lio_state.pos_end: "<<(g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end).transpose()<<std::endl;
            std::cout<<"all_lidar_frames[all_lidar_frames_n-1]->t_w_l_loop: "<<all_lidar_frames[all_lidar_frames_n-1]->t_w_l_loop.transpose()<<std::endl;
            std::cout<<"更新结束"<<std::endl;
            // first_loop = true;
            loop_closure_optimized = false;
            lc_updated_front_end_frame_id = all_lidar_frames.size();
            if(1) // 发布回环更新之后的时间戳
            {
                std_msgs::UInt64 jump_time_msg;
                jump_time_msg.data = ros::Time().fromSec(Measures.lidar_end_time).toNSec();
                pubTimeCorrection.publish(jump_time_msg);
            }
            // wait_for_save_data = true;
            frame_poses_updating = false;
            vdf.clear_curr_data();
        }
        if(process_debug) std::cout<<"m_imu_process"<< std::endl;
        // ====== 2. imu传播与将雷达点全部投影到最后一个点所在的局部坐标系 ======
        m_imu_process->Process( Measures, g_lio_state, feats_undistort );
        double imu_process_t = omp_get_wtime();
        g_lio_state.lidar_type = Measures.lidar_type;
        StatesGroup state_imu_propagate(g_lio_state);
        if ( feats_undistort->empty() || ( feats_undistort == NULL ) )
        {
            frame_first_pt_time = Measures.lidar_beg_time;
            std::cout << "not ready for odometry" << std::endl;
            continue;
        }
        if ( ( Measures.lidar_beg_time - frame_first_pt_time ) < 0 ) // 雷达时间异常
        {
            std::cout << "||||||||||LiDAR disorder||||||||||" << std::endl;
            frame_first_pt_time = Measures.lidar_beg_time;
            std::cout << "not ready for odometry" << std::endl;
            continue;
        }
        double plane_fit_start_t = omp_get_wtime();
        LidarFrame * lidar_frame = new LidarFrame(feats_undistort);
        lidar_frame->frame_id = all_lidar_frames.size();
        lidar_frame->lidar_end_time = Measures.lidar_end_time;
        lidar_frame->R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
        lidar_frame->t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
        lidar_frame->set_plane_merge_param(plane_merge_angle, plane_merge_dist, max_merge_plane_n, min_plane_pts_n, max_eigen_dist_th, plane2orig_dist_th);
        LidarFrameFilterUse * lidar_frame_for_filter = new LidarFrameFilterUse();
        if(use_isdor)
        {
            double isdor_filter_start_t = omp_get_wtime();
            lidar_frame_for_filter->frame_id = lidar_frame->frame_id;
            lidar_frame_for_filter->point_cloud_ptr = *feats_undistort;
            lidar_frame_for_filter->R_w_l = lidar_frame->R_w_l;
            lidar_frame_for_filter->t_w_l = lidar_frame->t_w_l;
            isdor.pts_filter(lidar_frame_for_filter, isdor_robust_flag, 8);
            lidar_frame->use_filtered_pts = true;
            lidar_frame->dynamic_filter_remained_pts = lidar_frame_for_filter->dynamic_filter_remained_pts;
            double isdor_filter_end_t = omp_get_wtime();
            std::cout<<"isdor_filter: "<<isdor_filter_end_t-isdor_filter_start_t<<std::endl;
        }
        if(process_debug) std::cout<<"angle_split_and_pca_plane_fit_fast"<< std::endl;
        lidar_frame->angle_split_and_pca_plane_fit_fast(pi_split_n, process_debug, pca_filter_flag);
        // 动态调整下采样网格大小
        if(!large_scale_env && abs(last_plane_pts_ds_n-lio_pts_num)>lio_pts_num_th)
        {
            double scale_ = 0.1*(last_plane_pts_ds_n/lio_pts_num-1);
            if(scale_>0.1) scale_=0.1;
            if(scale_<-0.1) scale_=-0.1;
            plane_voxel_ds_size +=  scale_;
            if(plane_voxel_ds_size>0.5) plane_voxel_ds_size=0.5;
            if(plane_voxel_ds_size<0.01) plane_voxel_ds_size=0.01;
        }
        if(!large_scale_env && abs(last_non_plane_pts_ds_n-lio_pts_num)>lio_pts_num_th)
        {
            double scale_ = 0.1*(last_non_plane_pts_ds_n/lio_pts_num-1);
            if(scale_>0.2) scale_=0.2;
            if(scale_<-0.2) scale_=-0.2;
            non_plane_voxel_ds_size +=  scale_;
            if(non_plane_voxel_ds_size>1.5) non_plane_voxel_ds_size=2.0;
            if(non_plane_voxel_ds_size<0.01) non_plane_voxel_ds_size=0.01;
        }
        if(process_debug) std::cout<<"use_non_plane_pt"<< std::endl;
        double voxel_filter_start_t = omp_get_wtime();
        double plane_fit_time = voxel_filter_start_t-plane_fit_start_t;
        if(use_non_plane_pt) lidar_frame->use_non_plane = true;
        if(process_debug) std::cout<<"voxel_filter"<< std::endl;
        int total_lp_n = lidar_frame->planes_num();
        #ifdef MP_EN // 需要在主线程中启动，在子线程中速度特别慢！！！
            omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
        #endif
        for( int i = 0; i < total_lp_n; i++ )
        {
            Plane * local_plane_ = lidar_frame->get_plane(i);
            double voxel_ds_size = local_plane_->type>-1 ? plane_voxel_ds_size : non_plane_voxel_ds_size;
            local_plane_->voxel_filter(voxel_ds_size, true, false, pca_filter_flag); // pca_filter_flag 有的话对于点数太少的网格不提取点
        }
        double downSizeFilter_t = omp_get_wtime(); // downSizeFilter_t-voxel_filter_start_t
        double update_sphere_start_t = omp_get_wtime();
        if(use_sphere_filter) dynamic_pts_filter.update_sphere2(lidar_frame);
        double update_sphere_end_t = omp_get_wtime();
        if(use_voxel_filter)
        {
            // double pts_filter_start_t = omp_get_wtime();
            vdf.pts_filter(lidar_frame);
            // double pts_filter_end_t = omp_get_wtime();
            // if(g_LiDAR_frame_index%10==0) std::cout<<"pts_filter: "<<pts_filter_end_t-pts_filter_start_t<<std::endl;
        }
        std::vector<Eigen::Vector3d> local_pts_ds; // 局部坐标系下的点
        std::vector<int> pts_lp_ids; // 点id
        int lp_n = lidar_frame->planes_num(true);
        std::vector<Eigen::Vector3d> lp_normals(lp_n);
        for(int lp_i=0; lp_i<lp_n; lp_i++) // 逐个考虑当前关键帧中的平面
        {
            const Plane * local_plane_ = lidar_frame->get_plane(lp_i);
            lp_normals[lp_i] = local_plane_->normal;
            std::vector<int> pt_plane_ids_(local_plane_->pts_lio.size(), lp_i);
            pts_lp_ids.insert(pts_lp_ids.end(), pt_plane_ids_.begin(), pt_plane_ids_.end());
            local_pts_ds.insert(local_pts_ds.end(), local_plane_->pts_lio.begin(), local_plane_->pts_lio.end());
        }
        int plane_pts_ds_n = local_pts_ds.size(); // 平面点数量
        int non_planes_n = lidar_frame->non_planes.size();
        for(int lp_i=0; lp_i<non_planes_n; lp_i++) // 逐个考虑当前关键帧中的平面
        {
            const Plane * local_plane_ = lidar_frame->non_planes[lp_i];
            local_pts_ds.insert(local_pts_ds.end(), local_plane_->pts_lio.begin(), local_plane_->pts_lio.end());
        }
        int total_pts_ds_n = local_pts_ds.size();
        int non_plane_pts_ds_n = total_pts_ds_n - plane_pts_ds_n; // 非平面点数量
        lidar_frame->feats_down_size_ = total_pts_ds_n;
        last_plane_pts_ds_n = plane_pts_ds_n;
        last_non_plane_pts_ds_n = non_plane_pts_ds_n;
        std::vector<int> pt_plane_ids_(non_plane_pts_ds_n, -1);
        pts_lp_ids.insert(pts_lp_ids.end(), pt_plane_ids_.begin(), pt_plane_ids_.end());
        // 平面点云下采样之后的点数量
        // 非平面点云下采样之后的点数量
        double after_update_sphere = omp_get_wtime()-update_sphere_end_t;
        // std::cout<<"plane_pts_ds_n: "<<plane_pts_ds_n<<", plane_fit_time: "<<plane_fit_time<<std::endl;
        // if(!use_backend) std::cout<<"plane_fit_time: "<<plane_fit_time<<std::endl;
        if(octree_feature_ptr->size()==0 && total_pts_ds_n>5)
        {
            if(process_debug) std::cout<<"octree_feature_init"<< std::endl;
            std::vector<Eigen::Vector3d> gl_pts_add(total_pts_ds_n);
            std::vector<float> frame_plane_id = {static_cast<float>(lidar_frame->frame_id), -1.0};
            std::vector<std::vector<float>> gl_attrs_add(total_pts_ds_n, frame_plane_id);
            // std::vector<Plane> local_planes; std::vector<Eigen::Vector3d> pts;  global_planes
            Eigen::Matrix3d R_w_l = lidar_frame->R_w_l;
            Eigen::Vector3d t_w_l = lidar_frame->t_w_l;
            for(int pt_i=0; pt_i<total_pts_ds_n; pt_i++)
            {
                gl_pts_add[pt_i] = R_w_l*local_pts_ds[pt_i] + t_w_l;
                gl_attrs_add[pt_i][1] = pts_lp_ids[pt_i];
            }
            octree_feature_ptr->update_with_attr(gl_pts_add, gl_attrs_add, true);
            if(use_backend)
            {
                mtx_tree_pts_delay_buffer.lock();
                tree_pts_delay.emplace_back(gl_pts_add);
                tree_pts_attrs_delay.emplace_back(gl_attrs_add);
                mtx_tree_pts_delay_buffer.unlock();
            }
            // lidar_frame->clear_lio_data();
            lidar_frame->R_w_l_loop = lidar_frame->R_w_l;
            lidar_frame->t_w_l_loop = lidar_frame->t_w_l;
            // 由IMU位姿转到雷达的位姿带来的协方差变化 扰动方式：简单相减 \delta_theta = Log((\check{R}^w_l)^T*R^w_l)  \delta t = t^w_l - \check{t}^w_l
            Eigen::Matrix<double, 6,6> W_ = Eigen::Matrix<double, 6,6>::Identity();
            W_.block<3,3>(0,0) = m_lidar_to_imu_R.transpose();
            W_.block<3,3>(3,0) = -g_lio_state.rot_end * vec_to_hat(m_lidar_to_imu_t);
            // 扰动方式变化带来的协方差变化 简单相减==>左乘扰动
            Eigen::Matrix<double, 6,6> W_1 = Eigen::Matrix<double, 6,6>::Identity();
            W_1.block<3,3>(0,0) = lidar_frame->R_w_l;
            W_1.block<3,3>(3,0) = vec_to_hat(lidar_frame->t_w_l) * lidar_frame->R_w_l;
            Eigen::Matrix<double, 6,6> W_2 = W_1 * W_;
            lidar_frame->cov = W_2*g_lio_state.get_imu_pose_cov().block<6,6>(0,0)*W_2.transpose(); // IMU位姿协方差转为雷达位姿左乘扰动协方差
            lidar_frame->calau_global_normal_and_d();
            all_lidar_frames.emplace_back(lidar_frame);
            if(!use_backend) lidar_frame->clear_unnecessary_data();
            if(process_debug) std::cout<<"octree_feature_init_end"<< std::endl;
            continue;
        }
        // std::cout<<"迭代优化"<< std::endl;
        StatesGroup state_init(g_lio_state);
        StatesGroup state_update(g_lio_state);
        std::vector<Eigen::Vector3d> gnormal_to_glp(total_pts_ds_n); // 全局坐标系下每个局部平面点在对应的全局平面法线  glp 全局坐标系下的局部平面点
        std::vector<Eigen::Vector3d> gl_pts_lio(total_pts_ds_n);
        std::vector<Eigen::Matrix3d> point_crossmats(total_pts_ds_n);
        std::vector<float> gd_to_glp(total_pts_ds_n); // 全局坐标系下每个局部平面点在对应的全局平面参数d  glp 全局坐标系下的局部平面点
        std::vector<float> dist_to_glp(total_pts_ds_n); //全局坐标系下每个局部平面的每个点到对应全局平面的距离
        std::vector<float> pts_weights(total_pts_ds_n); //全局坐标系下每个局部平面的每个点到对应全局平面的距离的权重
        std::vector<int> valid_lidar_pt_ids(total_pts_ds_n, -1);
        for(int pt_i=0; pt_i<total_pts_ds_n; pt_i++)
        {
            Eigen::Vector3d point_this = m_lidar_to_imu_R*local_pts_ds[pt_i] + m_lidar_to_imu_t;
            Eigen::Matrix3d point_crossmat;
            point_crossmat << SKEW_SYM_MATRIX( point_this );
            point_crossmats[pt_i] = point_crossmat;
        }
        double deltaT = 0.0, deltaR = 0.0;
        int rematch_num = 0;
        bool rematch_en = 0;
        int valid_lidar_pts_n = 0;
        P_inv = state_update.cov.inverse(); // 迭代过程中不会更新协方差，减少不必要的计算
        // state_update = state_init + last_solution;
        // state_init = state_update;
        if(process_debug) std::cout<<"iterCount"<< std::endl;
        double ready_time = omp_get_wtime()-start_t;
        double solve_time = 0, match_time=0.0, kd_tree_search_time=0.0;
        std::vector<Eigen::Matrix3d> R_w_ls;
        std::vector<Eigen::Vector3d> t_w_ls;
        // ======4. 迭代优化======
        for ( int iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ )
        {
            double match_start = omp_get_wtime();
            valid_lidar_pts_n = 0;
            // std::cout<<"iterCount: "<<iterCount<< std::endl;
            Eigen::Matrix3d R_w_l = state_update.rot_end*m_lidar_to_imu_R;
            Eigen::Vector3d t_w_l = state_update.rot_end*m_lidar_to_imu_t + state_update.pos_end;
            R_w_ls.push_back(R_w_l);
            t_w_ls.push_back(t_w_l);
            lidar_frame->R_w_l = R_w_l;
            lidar_frame->t_w_l = t_w_l;
            // if(process_debug) std::cout<<"plane_match"<< std::endl;
            // 计算每个点对应的平面法线和有符号距离
            #ifdef MP_EN // 需要在主线程中启动，在子线程中速度特别慢！！！
                omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
            #endif
            for(int pt_i=0; pt_i<total_pts_ds_n; pt_i++)
            {
                // double kd_search_start = omp_get_wtime();
                // 2. 找到每个平面对应的全局平面，这样就可以简单建立点到面的距离关系
                Eigen::Vector3d glp_normal;
                bool is_lp_pt = pts_lp_ids[pt_i]>=0;
                if(is_lp_pt) glp_normal = R_w_l*lp_normals[pts_lp_ids[pt_i]]; // 当前平面法线在全局坐标系下坐标
                std::vector<std::vector<float>> points_near;
                std::vector<float> pointSearchSqDis_surf;
                gl_pts_lio[pt_i] = R_w_l*local_pts_ds[pt_i] + t_w_l;
                if(iterCount == 0 || rematch_en)
                {
                    pts_weights[pt_i] = 0.0;
                    octree_feature_ptr->knnNeighbors_eigen(gl_pts_lio[pt_i], num_match_pts, points_near, pointSearchSqDis_surf);
                    // avoid_non_plane_pts_func 避开搜索树中非平面点
                    float max_distance = pointSearchSqDis_surf[ num_match_pts - 1 ];
                    if(max_distance > m_maximum_pt_kdtree_dis && !large_scale_env || max_distance > 5.0 && large_scale_env) continue; //  超过0.5就无效了
                    if(is_lp_pt && int(points_near[0][4])>=0 && !frame_poses_updating) // 只要回环检测一开始更新位姿，这里就不能使用，要等位姿和搜索树更新完毕之后才能继续使用
                    {
                        if(pointSearchSqDis_surf[0]>=0.04 ) continue; // 平方距离
                        const int frame_id_ = int(points_near[0][3]);
                        const int local_plane_id_ = int(points_near[0][4]);
                        Eigen::Vector3d normal_;
                        float d_;
                        all_lidar_frames[frame_id_]->get_plane_normal_d(local_plane_id_, normal_, d_);
                        double cos_ = glp_normal.transpose()*normal_;
                        if(cos_<0.9659) continue; // 15度
                        gnormal_to_glp[pt_i] = normal_; // 相邻平面法线
                        gd_to_glp[pt_i] = d_; // 相邻平面参数d
                        pts_weights[pt_i] = cos_;
                    }
                    else
                    {
                        // 平面拟合
                        Eigen::Vector3d normal_fit;
                        float pd;
                        double pt_weight = lidar_cov_p;
                        bool planeValid = true;
                        if(1)
                        {
                            cv::Mat matA0( num_match_pts, 3, CV_32F, cv::Scalar::all( 0 ) );
                            cv::Mat matB0( num_match_pts, 1, CV_32F, cv::Scalar::all( -1 ) );
                            cv::Mat matX0( num_match_pts, 1, CV_32F, cv::Scalar::all( 0 ) );
                            for( int j = 0; j < num_match_pts; j++ )
                            {
                                matA0.at< float >( j, 0 ) = points_near[j][0];
                                matA0.at< float >( j, 1 ) = points_near[j][1];
                                matA0.at< float >( j, 2 ) = points_near[j][2];
                                const int frame_id_ = int(points_near[j][3]);
                            }
                            cv::solve( matA0, matB0, matX0, cv::DECOMP_QR ); // TODO
                            normal_fit << matX0.at< float >( 0, 0 ), matX0.at< float >( 1, 0 ), matX0.at< float >( 2, 0 );
                            double norm_ = normal_fit.norm()+1e-6;
                            pd = 1/norm_;
                            normal_fit /= norm_;
                            pt_weight=0;
                            // 检查拟合平面的好坏，距离超过0.1就无效
                            for( int j = 0; j < num_match_pts; j++)
                            {
                                float dist = fabs( normal_fit[0] * points_near[ j ][0] + normal_fit[1] * points_near[ j ][1] + normal_fit[2] * points_near[ j ][2] + pd );
                                if(dist > m_planar_check_dis) // Raw 0.10
                                {
                                    planeValid = false;
                                    break;
                                }
                                pt_weight+=dist;
                            }
                            double pt_weight_avr = pt_weight/num_match_pts;
                            if(pt_weight_avr>0.2) pt_weight_avr=0.2;
                            pt_weight = cos(pt_weight_avr*3.1415926*2.5);
                        }
                        if(!planeValid) continue;
                        if(is_lp_pt) // 对于平面点
                        {
                            double cos_ = glp_normal.transpose()*normal_fit;
                            if(cos_<0.9659) continue; // 15度
                            pts_weights[pt_i] = cos_;
                        }
                        else
                            pts_weights[pt_i] = pt_weight;
                        gnormal_to_glp[pt_i] = normal_fit;
                        gd_to_glp[pt_i] = pd;
                    }
                    // valid_lidar_pts_n++;
                }
                if(pts_weights[pt_i]<1e-9) continue;
                double pd2 = gnormal_to_glp[pt_i].dot(gl_pts_lio[pt_i]) + gd_to_glp[pt_i];
                if(is_lp_pt)
                {
                    double cos_ = abs(glp_normal.transpose()*gnormal_to_glp[pt_i]);
                    pts_weights[pt_i] = cos_;
                    dist_to_glp[pt_i] = pd2;
                }
                else
                {
                    // if( fabs(pd2) >= 0.3-0.3/(1+exp(-0.5*iterCount)) && all_lidar_frames.size()>=3 && !large_scale_env) continue; // 考虑一个随着时间减小的值
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(local_pts_ds[pt_i].norm());
                    // if(large_scale_env && s<=0.9) // 当初始位置偏差较大的时候这种方式很难矫正
                    if(s<=0.9) // 当初始位置偏差较大的时候这种方式很难矫正
                    {
                        pts_weights[pt_i] = 0.0;
                        continue;
                    }
                    dist_to_glp[pt_i] = pd2;
                }
                // kd_tree_search_time += omp_get_wtime() - kd_search_start;
            }
            // std::cout<<"计算雅克比矩阵和测量向量: "<<iterCount<< std::endl;
            match_time += omp_get_wtime() - match_start;
            double solve_start = omp_get_wtime();
            // if(process_debug) std::cout<<"EKF_solve_start"<<", valid_lidar_pts_n: "<<valid_lidar_pts_n<< std::endl;
            // 3. 计算雅克比矩阵和测量向量
            // 统计有效点数，并记录每个点在有效向量中的索引id
            valid_lidar_pts_n = 0;
            for(int pt_i=0; pt_i<total_pts_ds_n; pt_i++)
            {
                valid_lidar_pt_ids[pt_i] = -1;
                if(pts_weights[pt_i]<1e-9) continue;
                valid_lidar_pt_ids[pt_i] = valid_lidar_pts_n;
                valid_lidar_pts_n++;
            }
            Eigen::MatrixXd Hsub( valid_lidar_pts_n, 6 ); // 本来应该是 DIM_OF_STATES 维度的，但是除了前6维度都是0，这里使用缩减版
            Eigen::VectorXd meas_vec( valid_lidar_pts_n );
            Eigen::MatrixXd H_T_R_inv( 6, valid_lidar_pts_n ); // H^T* R^{-1}
            Hsub.setZero();
            H_T_R_inv.setZero();
            auto vec = state_init - state_update;
            Eigen::Vector3d delta_theta =  -vec.block<3,1>(0,0);
            // Eigen::Matrix3d J_r_vec_inv = inverse_right_jacobian_of_rotation_matrix(delta_theta); // ==> 对应公式中的 J_r^{-1}
            Eigen::Matrix3d J_r_vec = right_jacobian_of_rotation_matrix(delta_theta); // ==> 对应公式中的 J_r(\check{\theta}_{op,k} -\check{\theta}_{k} )
            #ifdef MP_EN // 需要在主线程中启动，在子线程中速度特别慢！！！
                omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
            #endif
            for(int pt_i=0; pt_i<total_pts_ds_n; pt_i++)
            {
                if(pts_weights[pt_i]<1e-9) continue;
                const int & Num_i = valid_lidar_pt_ids[pt_i];
                // if(Num_i>=valid_lidar_pts_n) std::cout<<"lp_i: "<<lp_i<<", "<<pt_i<<", "<<iterCount<< std::endl;
                const Eigen::Vector3d & normal_ = gnormal_to_glp[pt_i];
                //* 转置，而point_crossmat没转置，就是添加负号！！
                Eigen::Vector3d A = point_crossmats[pt_i] * state_update.rot_end.transpose() * normal_;
                A = J_r_vec.transpose()*A; // H*J = H*J_h^{-1} = H*(J_r^{-1})^{-1} = H*J_r
                // std::cout<<"Num_i: "<<Num_i<<", "<<pt_i<<", "<<iterCount<< std::endl;
                Hsub.block<1, 6>(Num_i, 0)  << A[0], A[1], A[2], normal_[0], normal_[1], normal_[2];
                meas_vec( Num_i ) = -dist_to_glp[pt_i];
                H_T_R_inv.block<6, 1>(0, Num_i) = Hsub.block<1, 6>(Num_i, 0).transpose()/lidar_pt_cov*pts_weights[pt_i];
            }
            // std::cout<<"key_frame.plane_pts_num(): "<<key_frame.plane_pts_num()<< std::endl;
            // if(process_debug) std::cout<<"EKF_solve_K: "<<iterCount<<", DIM_OF_STATES: "<<DIM_OF_STATES<<", valid_lidar_pts_n: "<<valid_lidar_pts_n<< std::endl;
            // std::cout<<"valid_lidar_pts_n: "<<valid_lidar_pts_n<<", DIM_OF_STATES "<<DIM_OF_STATES<< std::endl;
            // 4. 迭代拓展卡尔曼滤波
            Eigen::MatrixXd K( DIM_OF_STATES, valid_lidar_pts_n );
            H_T_R_inv_H.block<6,6>(0,0) = H_T_R_inv * Hsub;
            K = ( P_inv + H_T_R_inv_H ).inverse().block< DIM_OF_STATES, 6 >( 0, 0 ) * H_T_R_inv;
            // vec.block< 3, 1 >( 0, 0 ) = J_r_vec * vec.block< 3, 1 >( 0, 0 ); // J^{-1}*vec
            Eigen::Matrix< double, DIM_OF_STATES, 1 > solution = K * ( meas_vec - Hsub * vec.block< 6, 1 >( 0, 0 ) );
            state_update = state_init + solution;
            Eigen::Vector3d rot_add = solution.block< 3, 1 >( 0, 0 );
            Eigen::Vector3d t_add = solution.block< 3, 1 >( 3, 0 );
            bool flg_EKF_converged = false;
            if( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) )
                flg_EKF_converged = true;
            deltaR = rot_add.norm() * 57.3;
            deltaT = t_add.norm() * 100;
            state_update.last_update_time = Measures.lidar_end_time;
            rematch_en = false;
            if(!rematch_less || flg_EKF_converged || rematch_num == 0 && iterCount == NUM_MAX_ITERATIONS - 2 )
            {
                rematch_en = true;
                rematch_num++;
            }
            if(rematch_less && rematch_num >= 2 || iterCount == NUM_MAX_ITERATIONS - 1 ) // Fast lio ori version.
            {
                G.block< DIM_OF_STATES, 6 >( 0, 0 ) = K * Hsub;
                state_update.cov = ( I_STATE - G ) * state_update.cov;
                lidar_frame->H_T_R_inv_H = H_T_R_inv_H.block<6,6>(0,0);
                break;
            }
            double solve_end = omp_get_wtime();
            solve_time += solve_end-solve_start;
            // if(process_debug) std::cout<<"EKF_solve_end"<< std::endl;
        }
        g_lio_state = state_update;
        lidar_frame->R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
        lidar_frame->t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
        lidar_frame->R_w_l_loop = lidar_frame->R_w_l;
        lidar_frame->t_w_l_loop = lidar_frame->t_w_l;
        if(all_lidar_frames.size()>0) path_length += (all_lidar_frames[all_lidar_frames.size()-1]->t_w_l-lidar_frame->t_w_l).norm();
        // 由IMU位姿转到雷达的位姿带来的协方差变化 扰动方式：简单相减 \delta_theta = Log((\check{R}^w_l)^T*R^w_l)  \delta t = t^w_l - \check{t}^w_l
        Eigen::Matrix<double, 6,6> W_ = Eigen::Matrix<double, 6,6>::Identity();
        W_.block<3,3>(0,0) = m_lidar_to_imu_R.transpose();
        W_.block<3,3>(3,0) = -g_lio_state.rot_end * vec_to_hat(m_lidar_to_imu_t);
        // 扰动方式变化带来的协方差变化 简单相减==>左乘扰动
        Eigen::Matrix<double, 6,6> W_1 = Eigen::Matrix<double, 6,6>::Identity();
        W_1.block<3,3>(0,0) = lidar_frame->R_w_l;
        W_1.block<3,3>(3,0) = vec_to_hat(lidar_frame->t_w_l) * lidar_frame->R_w_l;
        Eigen::Matrix<double, 6,6> W_2 = W_1 * W_;
        lidar_frame->cov = W_2*g_lio_state.get_imu_pose_cov().block<6,6>(0,0)*W_2.transpose(); // IMU位姿协方差转为雷达位姿左乘扰动协方差
        if(process_debug) std::cout<<"tree_update_start"<< std::endl;
        double tree_update_start_t = omp_get_wtime();
        // std::cout<<"八叉树更新 start"<<std::endl;
        // ======5. 八叉树更新======
        if(1)
        {
            Eigen::Matrix3d R_w_l = lidar_frame->R_w_l;
            Eigen::Vector3d t_w_l = lidar_frame->t_w_l;
            // std::cout<<"front_lio t_w_l: "<<t_w_l.transpose()<<", frame_id: "<<lidar_frame->frame_id<<std::endl;
            std::vector<Eigen::Vector3d> gl_pts_add(total_pts_ds_n);
            std::vector<float> frame_plane_id = {static_cast<float>(lidar_frame->frame_id), -1.0};
            std::vector<std::vector<float>> gl_attrs_add(total_pts_ds_n, frame_plane_id);
            #ifdef MP_EN // 需要在主线程中启动，在子线程中速度特别慢！！！
                omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
            #endif
            for(int pt_i=0; pt_i<total_pts_ds_n; pt_i++)
            {
                gl_pts_add[pt_i] = R_w_l*local_pts_ds[pt_i] + t_w_l;
                gl_attrs_add[pt_i][1] = pts_lp_ids[pt_i];
            }
            octree_feature_ptr->update_with_attr(gl_pts_add, gl_attrs_add, true);
            if(use_backend)
            {
                mtx_tree_pts_delay_buffer.lock();
                tree_pts_delay.emplace_back(gl_pts_add);
                tree_pts_attrs_delay.emplace_back(gl_attrs_add);
                mtx_tree_pts_delay_buffer.unlock();
                if(historical_tree_pts_to_add.size()>0)
                {
                    mtx_historical_tree_pts_to_add.lock();
                    std::cout<<"octree_feature_ptr->size(): "<<octree_feature_ptr->size()<<std::endl;
                    double historical_tree_pts_t = omp_get_wtime();
                    octree_feature_ptr->update_with_attr(historical_tree_pts_to_add.front(), historical_tree_attrs_to_add.front(), true);
                    historical_tree_pts_to_add.pop_front();
                    historical_tree_attrs_to_add.pop_front();
                    double historical_tree_pts_end_t = omp_get_wtime();
                    std::cout<<"octree_feature_ptr->size(): "<<octree_feature_ptr->size()<<", time: "<<historical_tree_pts_end_t-historical_tree_pts_t<<std::endl;
                    mtx_historical_tree_pts_to_add.unlock();
                }
            }
            if(all_lidar_frames.size()%5==0 && 0) // 非激活网格测试
            {
                std::cout<<"添加非激活网格"<<std::endl;
                double update_non_act_start_t = omp_get_wtime();
                int lp_n_ = lidar_frame->planes_num();
                std::vector<Eigen::Vector3d> non_act_pts_;
                double scale_ = 0.1;
                int start_iter_i = 0;
                if(lidar_blind>scale_) start_iter_i = int(lidar_blind/scale_);
                int sphere_grid2planes_n = lidar_frame->sphere_id2planes.size();
                for(int sphere_id=0; sphere_id<sphere_grid2planes_n; sphere_id++)
                {
                    if(lidar_frame->sphere_id2planes[sphere_id]==nullptr) continue;
                    Plane * cube_plane = lidar_frame->sphere_id2planes[sphere_id];
                    // if(cube_plane->points_size<min_plane_pts_n) continue;
                    if(cube_plane->get_center()[2]>0) continue;
                    Eigen::Vector3d center_g_ = lidar_frame->R_w_l*cube_plane->get_center() + lidar_frame->t_w_l;
                    Eigen::Vector3d dir_ = center_g_-lidar_frame->t_w_l;
                    int end_iter_i = (cube_plane->min_norm*0.9/scale_);
                    double len = dir_.norm();
                    dir_ /= len;
                    int pn = len/scale_;
                    end_iter_i = std::min(pn, end_iter_i);
                    for(int k=start_iter_i; k<end_iter_i; k++)
                    {
                        Eigen::Vector3d pt_non_act = lidar_frame->t_w_l + dir_*scale_*k;
                        non_act_pts_.push_back(pt_non_act);
                    }
                }
                for(int lp_i=0; lp_i<lp_n_ && 0; lp_i++) // 逐个考虑当前关键帧中的平面
                {
                    const Plane * local_plane_ = lidar_frame->get_plane(lp_i);
                    Eigen::Vector3d center_g_ = lidar_frame->R_w_l*local_plane_->get_center() + lidar_frame->t_w_l;
                    Eigen::Vector3d dir_ = center_g_-lidar_frame->t_w_l;
                    int end_iter_i = (local_plane_->min_norm*0.9/scale_);
                    double len = dir_.norm();
                    dir_ /= len;
                    int pn = len/scale_;
                    end_iter_i = std::min(pn, end_iter_i);
                    for(int k=start_iter_i; k<end_iter_i; k++)
                    {
                        Eigen::Vector3d pt_non_act = lidar_frame->t_w_l + dir_*scale_*k;
                        non_act_pts_.push_back(pt_non_act);
                    }
                }
                octree_feature_ptr->update_non_act(non_act_pts_);
                non_act_pts.insert(non_act_pts.end(), non_act_pts_.begin(), non_act_pts_.end());
                double update_non_act_end_t = omp_get_wtime();
                std::cout<<"添加非激活网格: "<<update_non_act_end_t-update_non_act_start_t<<std::endl;
                // non_act_pts.clear();
            }
        }
        lidar_frame->calau_global_normal_and_d();
        if(use_isdor)
        {
            double isdor_update_voxel_start_t = omp_get_wtime();
            lidar_frame_for_filter->R_w_l = lidar_frame->R_w_l;
            lidar_frame_for_filter->t_w_l = lidar_frame->t_w_l;
            isdor.kfs_buffer.push_back(lidar_frame_for_filter);
            isdor.update_voxel(lidar_frame->frame_id, 0); // 0 for 07.bag  5 for corridor.bag
            double isdor_update_voxel_end_t = omp_get_wtime();
            std::cout<<"isdor_update_voxel: "<<isdor_update_voxel_end_t-isdor_update_voxel_start_t<<std::endl;
        }
        else
        {
            delete lidar_frame_for_filter;
            lidar_frame_for_filter = nullptr;
        }
        if(use_voxel_filter)
        {
            double update_voxel_start_t = omp_get_wtime();
            int updated_kf_id = vdf.update_voxel_directly2(int(all_key_frames.size())-1);
            double update_voxel_end_t = omp_get_wtime();
            // if(updated_kf_id>=0) std::cout<<"updated_kf_id: "<<updated_kf_id<<", cur_kf_id: "<<int(all_key_frames.size())-1<<", update_voxel_directly2: "<<update_voxel_end_t-update_voxel_start_t<<std::endl;
        }
        all_lidar_frames.emplace_back(lidar_frame);
        if(!use_backend) lidar_frame->clear_unnecessary_data();
        if(process_debug) std::cout<<"lio_solve_end"<< std::endl;
        if(g_LiDAR_frame_index%10==0) 
        std::cout<<"Lidar time: "<<Measures.lidar_beg_time-first_lidar_time<<", octree_feature_ptr: "<<octree_feature_ptr->size()<<", all_key_frames.size(): "<<all_key_frames.size()<< ", memory used: (" <<  Common_tools::get_RSS_Gb() << " Gb)" <<", total_pts_ds_n: "<<total_pts_ds_n<<std::endl;
        end_t = omp_get_wtime();
        // 显示各种运算时间
        if(!use_backend && all_lidar_frames.size()%10==0)
        {
            std::cout<<"\ng_LiDAR_frame_index: "<<g_LiDAR_frame_index<<std::endl;
            std::cout<< std::fixed << std::setprecision(7);
            std::cout<<"undistort: "<<feats_undistort->size()<<", valid_split_n: "<<lidar_frame->valid_split_n<<", plane_voxel_ds: "<<plane_voxel_ds_size<<", nonplane_voxel_ds: "<<non_plane_voxel_ds_size<<std::endl;
            std::cout<<"planes_num: "<<lidar_frame->planes_num()<<", non_planes: "<<lidar_frame->non_planes.size()<<", plane_pts_ds_n: "<<plane_pts_ds_n<<", non_plane_pts_ds_n: "<<non_plane_pts_ds_n<<std::endl;
            std::cout<<"imu_process_time: "<<imu_process_t - start_t<<std::endl; // 0.000894163
            std::cout<<"plane_fit_time: "<<plane_fit_time<<std::endl; // 0.0019098
            std::cout<<"downSizeFilter_time: "<<downSizeFilter_t-voxel_filter_start_t<<std::endl; // 0.000967148
            std::cout<<"update_sphere: "<<update_sphere_end_t - update_sphere_start_t<<std::endl;
            std::cout<<"after_update_sphere: "<<after_update_sphere<<std::endl; // 0.0019098
            std::cout<<"ready_time: "<<ready_time<<std::endl; // 0.0019098
            std::cout<<"match_time: "<<match_time<<std::endl; // 0.00243819
            std::cout<<"solve_time: "<<solve_time<<std::endl; // 0.000392439
            std::cout<<"cov: "<<state_update.get_imu_pose_cov().diagonal().transpose()<<std::endl;
            std::cout<<"tree search_time: "<<kd_tree_search_time<<std::endl; // 0.00243819
            std::cout<<"tree update time: "<<end_t-tree_update_start_t<<std::endl; // 0.000344489
            std::cout<<"lio total time: "<<end_t-start_t<<std::endl; // 0.00530818
        }
        // std::cout<<"lio total time: "<<end_t-start_t<<std::endl; // 0.00530818
        // exit(1);
        // continue;
        // std::cout<<"end g_lio_state.pos_end: "<<g_lio_state.pos_end.transpose()<<std::endl;
        // 记录下采样点
        {
            std::ofstream log_file(m_map_output_dir + "/feats_down_size_"+alg_name+".txt", std::ios::app);
            log_file    <<std::fixed << std::setprecision(7)
                        <<Measures.lidar_end_time<<" " // lidar_beg_time
                        <<lidar_frame->feats_down_size_
                        // <<plane_pts_ds_n
                        <<std::endl;
            log_file.close();
        }
        {
            std::ofstream log_file(m_map_output_dir + "/exc_time_ppsam.txt", std::ios::app);
            log_file    <<std::fixed << std::setprecision(7)
                        <<Measures.lidar_end_time<<" " // lidar_beg_time
                        <<end_t-start_t
                        <<std::endl;
            log_file.close();
        }
        {
            Eigen::Quaterniond qr(g_lio_state.rot_end);
            Eigen::Vector3d position = g_lio_state.pos_end;
            std::ofstream log_file(m_map_output_dir + "/poses_"+alg_name+".txt", std::ios::app);
            log_file    << std::fixed << std::setprecision(7)
                        <<Measures.lidar_end_time<<" " // lidar_beg_time
                        <<position[0]<<" "<<position[1]<<" "<<position[2]<<" "
                        <<qr.x()<<" "<<qr.y()<<" "<<qr.z()<<" "<<qr.w()
                        // <<" "<<lidar_frame->frame_id
                        <<std::endl;
            log_file.close();
        }
        if(use_isdor && lidar_frame_for_filter!=nullptr && pubFilterPC.getNumSubscribers() > 0)
        {
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pts_lio_filtered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            std::vector<Eigen::Vector3d> & dynamic_object_pts_ = lidar_frame_for_filter->dynamic_object_pts;
            std::vector<Eigen::Vector3d> & filter_remained_pts_ = lidar_frame_for_filter->dynamic_filter_remained_pts;
            std::vector<int> color_;
            const int dynamic_object_pts_n = dynamic_object_pts_.size();
            const int filter_remained_pts_n = filter_remained_pts_.size();
            color_= std::vector<int>{255, 0, 0};
            for(int pt_i=0; pt_i<dynamic_object_pts_n; pt_i++)
            {
                const Eigen::Vector3d & pt_ = dynamic_object_pts_[pt_i];
                Eigen::Vector3d global_p = lidar_frame_for_filter->R_w_l*pt_ + lidar_frame_for_filter->t_w_l;
                pcl::PointXYZRGBNormal pt;
                pt.x = global_p(0);
                pt.y = global_p(1);
                pt.z = global_p(2);
                pt.r = color_[0];
                pt.g = color_[1];
                pt.b = color_[2];
                pts_lio_filtered->push_back(pt);
            }
            color_= std::vector<int>{125, 125, 125};
            for(int pt_i=0; pt_i<filter_remained_pts_n; pt_i++)
            {
                const Eigen::Vector3d & pt_ = filter_remained_pts_[pt_i];
                // if(abs(pt_[0])>max_x || abs(pt_[1])>max_y || pt_[2]>max_z) continue;
                Eigen::Vector3d global_p = lidar_frame_for_filter->R_w_l*pt_ + lidar_frame_for_filter->t_w_l;
                pcl::PointXYZRGBNormal pt;
                pt.x = global_p(0);
                pt.y = global_p(1);
                pt.z = global_p(2);
                pt.r = color_[0];
                pt.g = color_[1];
                pt.b = color_[2];
                pts_lio_filtered->push_back(pt);
            }
            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg( *pts_lio_filtered, laserCloudFullRes3 );
            laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time );
            laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
            pubFilterPC.publish( laserCloudFullRes3 );
        }
        /******* Publish Odometry ******/
        if(1)
        {
            Eigen::Vector3d euler_cur = RotMtoEuler( g_lio_state.rot_end );
            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw( euler_cur(0), euler_cur(1), euler_cur(2) );
            nav_msgs::Odometry odomAftMapped;
            odomAftMapped.header.frame_id = "world";
            odomAftMapped.child_frame_id = "/aft_mapped";
            // odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
            odomAftMapped.header.stamp.fromSec( Measures.lidar_end_time );
            odomAftMapped.pose.pose.orientation.x = geoQuat.x;
            odomAftMapped.pose.pose.orientation.y = geoQuat.y;
            odomAftMapped.pose.pose.orientation.z = geoQuat.z;
            odomAftMapped.pose.pose.orientation.w = geoQuat.w;
            odomAftMapped.pose.pose.position.x = g_lio_state.pos_end( 0 );
            odomAftMapped.pose.pose.position.y = g_lio_state.pos_end( 1 );
            odomAftMapped.pose.pose.position.z = g_lio_state.pos_end( 2 );
            pubOdomAftMapped.publish(odomAftMapped);
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3( odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z ) );
            q.setW( odomAftMapped.pose.pose.orientation.w );
            q.setX( odomAftMapped.pose.pose.orientation.x );
            q.setY( odomAftMapped.pose.pose.orientation.y );
            q.setZ( odomAftMapped.pose.pose.orientation.z );
            transform.setRotation( q );
            br.sendTransform( tf::StampedTransform( transform, ros::Time().fromSec( Measures.lidar_end_time ), "world", "/aft_mapped" ) );
            geometry_msgs::PoseStamped msg_body_pose;
            // msg_body_pose.header.stamp = ros::Time::now();
            msg_body_pose.header.stamp.fromSec( Measures.lidar_end_time );
            msg_body_pose.header.frame_id = "/camera_odom_frame";
            msg_body_pose.pose.position.x = g_lio_state.pos_end( 0 );
            msg_body_pose.pose.position.y = g_lio_state.pos_end( 1 );
            msg_body_pose.pose.position.z = g_lio_state.pos_end( 2 );
            msg_body_pose.pose.orientation.x = geoQuat.x;
            msg_body_pose.pose.orientation.y = geoQuat.y;
            msg_body_pose.pose.orientation.z = geoQuat.z;
            msg_body_pose.pose.orientation.w = geoQuat.w;
            /******* Publish Path ********/
            msg_body_pose.header.frame_id = "world";
            if ( frame_num > 10 )
            {
                path.poses.push_back( msg_body_pose );
            }
            pubPath.publish( path );
            frame_num++;
            if(1) // 发布去畸变之后的点云
            {
                int pc_size = feats_undistort->points.size();
                pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIMUBody(new pcl::PointCloud<pcl::PointXYZI>());
                laserCloudIMUBody->resize(pc_size);
                for (int i = 0; i < pc_size; i++)
                {
                    Eigen::Vector3d p_lidar( feats_undistort->points[i].x, feats_undistort->points[i].y, feats_undistort->points[i].z );
                    Eigen::Vector3d p_imu = m_lidar_to_imu_R*p_lidar + m_lidar_to_imu_t;
                    laserCloudIMUBody->points[i].x = p_imu[0];
                    laserCloudIMUBody->points[i].y = p_imu[1];
                    laserCloudIMUBody->points[i].z = p_imu[2];
                    laserCloudIMUBody->points[i].intensity = feats_undistort->points[i].intensity;
                }
                sensor_msgs::PointCloud2 laserCloudmsg;
                pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
                laserCloudmsg.header.stamp = ros::Time().fromSec(Measures.lidar_end_time);
                laserCloudmsg.header.frame_id = "body";
                pubUndistortedPC.publish(laserCloudmsg);
            }
        }
        // 发布全局坐标系下的点云
        if(1)
        {
            /******* Publish current frame points in world coordinates:  *******/
            PointCloudXYZINormal::Ptr laserCloudFullRes2( new PointCloudXYZINormal() );
            pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor( new pcl::PointCloud<pcl::PointXYZI>() );
            laserCloudFullRes2->clear();
            // *laserCloudFullRes2 = dense_map_en ? ( *feats_undistort ) : ( *feats_down );
            *laserCloudFullRes2 = *feats_undistort;
            int laserCloudFullResNum = laserCloudFullRes2->points.size();
            pcl::PointXYZI temp_point;
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_color(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            laserCloudFullResColor->clear();
            Eigen::Matrix3d R_w_l = g_lio_state.rot_end*m_lidar_to_imu_R;
            Eigen::Vector3d t_w_l = g_lio_state.rot_end*m_lidar_to_imu_t + g_lio_state.pos_end;
            std::vector<unsigned int> colors;
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            for ( int i = 0; i < laserCloudFullResNum; i++ )
            {
                Eigen::Vector3d p_body( laserCloudFullRes2->points[i].x, laserCloudFullRes2->points[i].y, laserCloudFullRes2->points[i].z );
                // Eigen::Vector3d p_global( g_lio_state.rot_end * ( m_lidar_to_imu_R*p_body + m_lidar_to_imu_t ) + g_lio_state.pos_end );
                Eigen::Vector3d p_global( R_w_l*p_body + t_w_l);
                temp_point.x = p_global( 0 );
                temp_point.y = p_global( 1 );
                temp_point.z = p_global( 2 );
                temp_point.intensity = laserCloudFullRes2->points[i].intensity;
                laserCloudFullResColor->push_back( temp_point );
                if(0)
                {
                    laserCloudFullRes2->points[i].x = p_global( 0 );
                    laserCloudFullRes2->points[i].y = p_global( 1 );
                    laserCloudFullRes2->points[i].z = p_global( 2 );
                    if(laserCloudFullRes2->points[ i ].curvature == -2)
                    {
                        Eigen::Vector3d normal_body( laserCloudFullRes2->points[ i ].normal_x, laserCloudFullRes2->points[ i ].normal_y, laserCloudFullRes2->points[ i ].normal_z );
                        Eigen::Vector3d normal_global = R_w_l*normal_body;
                        normal_global /= (1-normal_global.transpose()*t_w_l);
                        laserCloudFullRes2->points[ i ].normal_x = normal_global( 0 );
                        laserCloudFullRes2->points[ i ].normal_y = normal_global( 1 );
                        laserCloudFullRes2->points[ i ].normal_z = normal_global( 2 );
                    }
                    pcl::PointXYZRGBNormal pt;
                    pt.x = laserCloudFullRes2->points[i].x;
                    pt.y = laserCloudFullRes2->points[i].y;
                    pt.z = laserCloudFullRes2->points[i].z;
                    Eigen::Vector3f normal(laserCloudFullRes2->points[i].normal_x,laserCloudFullRes2->points[i].normal_y,laserCloudFullRes2->points[i].normal_z);
                    normal = normal.norm()>0 ? normal/normal.norm() : normal;
                    pt.normal_x = normal(0);
                    pt.normal_y = normal(1);
                    pt.normal_z = normal(2);
                    pt.curvature = laserCloudFullRes2->points[i].curvature;
                    pt.r = colors[0];
                    pt.g = colors[1];
                    pt.b = colors[2];
                    cloud_color->push_back(pt);
                }
            }
            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
            // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
            laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time );
            laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
            pubLaserCloudFullRes.publish( laserCloudFullRes3 );
        }
        // std::cout<<"post time: "<<omp_get_wtime()-start_t<<std::endl<<std::endl; // 0.00530818
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}

void SPLIN::data_association_thread()
{
    ros::Rate rate( 5000 );
    bool status = ros::ok();
    // 对关键帧进行平面特征提取
    int last_lidar_frame_id = 0;
    int last_key_frame_id = 0;
    LidarFrame * last_key_frame;
    int last_ds2map_frame_id = -1;
    Eigen::Vector3d last_update_pose;
    std::unordered_map<VOXEL_LOC, std::vector<float>> feat_cloud;
    bool feat_cloud_need_update = false;
    double kf_path_length = 0.0;
    while ( ros::ok() && !process_end)
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        if(last_lidar_frame_id>=all_lidar_frames.size() || loop_closure_optimized) continue;
        double start_t,end_t;
        start_t = omp_get_wtime();
        LidarFrame * lidar_frame = all_lidar_frames[last_lidar_frame_id];
        // 计算雷达位姿的协方差
        lidar_frame->R_w_l_updated = lidar_frame->R_w_l_loop; // R_w_l_loop 才是输入后端的数据，前端使用 R_w_l 仅仅用于记录里程计的数据
        lidar_frame->t_w_l_updated = lidar_frame->t_w_l_loop;
        lidar_frame->R_w_l_marg = lidar_frame->R_w_l_loop;
        lidar_frame->t_w_l_marg = lidar_frame->t_w_l_loop;
        if(1)
        {
            // 添加子图信息，并将带有回环位姿的子图信息添加到搜索树中(TODO)
            if(submap_info_buffer.size()>0 && 1) // 子图建立之后直接发送子图信息过来
            {
                mtx_submap_info_buffer.lock();
                const nav_msgs::Odometry::ConstPtr & odom_msg = submap_info_buffer.front();
                SubmapInfo submap_info;
                Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();
                Eigen::Quaterniond quaternion_(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
                pose_.block<3,3>(0,0) = quaternion_.toRotationMatrix();
                pose_.block<3,1>(0,3) = Eigen::Vector3d(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
                submap_info.msg_time = odom_msg->header.stamp.toSec();
                submap_info.submap_id = int(odom_msg->twist.covariance[0]);
                submap_info.end_key_frame_id = int(odom_msg->twist.covariance[1]);
                submap_info.start_key_frame_id = int(odom_msg->twist.covariance[2]);
                for(int i=0; i<6; i++)
				for(int j=0; j<6; j++)
					submap_info.pose_cov(i,j) = odom_msg->pose.covariance[i*6+j];
                submap_info_buffer.pop_front();
                mtx_submap_info_buffer.unlock();
                submap_info.orig_pose = pose_;
                submap_info.corr_pose = pose_; // 给一个初值
                submap_info.oriPoseFlag = true;
                if(!decouple_front)
                {
                    submap_info.get_data(feat_cloud);
                    feat_cloud.clear();
                }
                key_frame_id_to_submap_id[submap_info.end_key_frame_id] = submap_infos.size();
                if(submap_infos.size() != submap_info.submap_id)
                {
                    std::cout<<"Error id!!!!!\n";
                    std::cout<<"submap_id: "<<submap_info.submap_id<<", submap_infos.size(): "<<submap_infos.size()<<", tree_pts.size(): "<<submap_info.tree_pts.size()<<std::endl;
                    process_end = true;
                    break;
                }
                std::cout<<"submap_id: "<<submap_info.submap_id<<", submap_infos.size(): "<<submap_infos.size()<<", tree_pts.size(): "<<submap_info.tree_pts.size()<<std::endl;
                submap_infos.emplace_back(submap_info);
                if(!decouple_front) add_nearby_historical_tree_pts(historical_tree_attrs_added_submap_ids);
            }
            // 更新子图的回环位姿，并当上一次重构搜索树时候的回环位姿距离当前回环位姿超过 200米 的时候再次重构搜索树
            if(!decouple_front && pgo_KF_path_buffer.size()>0) // 回环检测成功之后才会发布关键帧路径信息
            {
                mtx_path_buffer.lock();
                nav_msgs::Path::ConstPtr pgo_path_nav = pgo_KF_path_buffer.back();
                pgo_KF_path_buffer.clear();
                mtx_path_buffer.unlock();
                // pgo_path_nav->poses[0].pose.position.x = lc_prev_idx; // 回环目标子图id
                // pgo_path_nav->poses[0].pose.position.y = lc_curr_idx; // 回环当前子图id
                // pgo_path_nav->poses[0].pose.position.z = lc_start; // 回环目标关键帧id
                // pgo_path_nav->poses[0].pose.orientation.x = lc_end; // 回环当前关键帧id
                int lc_prev_submap_idx = int(pgo_path_nav->poses[0].pose.position.x);
                int lc_target_kf_id = int(pgo_path_nav->poses[0].pose.position.z);
                int lc_curren_kf_id = int(pgo_path_nav->poses[0].pose.orientation.x);
                int lc_target_frame_id = all_key_frames[lc_target_kf_id]->frame_id;
                int lc_curren_frame_id = all_key_frames[lc_curren_kf_id]->frame_id;
                int used_loop_id = loop_ids.size();
                submap_loop_ids.push_back(std::pair<int, int>(int(pgo_path_nav->poses[0].pose.position.x), int(pgo_path_nav->poses[0].pose.position.y)));
                loop_ids.push_back(std::pair<int, int>(lc_target_frame_id, lc_curren_frame_id));
                Eigen::Vector3d curr_update_pose(pgo_path_nav->poses.back().pose.position.x, pgo_path_nav->poses.back().pose.position.y, pgo_path_nav->poses.back().pose.position.z);
                int curr_key_frame_id_ = pgo_path_nav->poses.back().header.seq;
                double pose_drift_ = (curr_update_pose-all_key_frames[curr_key_frame_id_]->t_w_l).norm();
                feat_cloud_need_update = true;
                std::cout<<"pose_drift_: "<<pose_drift_<<std::endl;
                // 1. 更新子图、关键帧、普通帧位姿信息
                last_update_pose = curr_update_pose;
                loop_ids_used.push_back(loop_ids.back()); // 记录使用过的回环id
                std::vector<Eigen::Vector3d> pose_pts;
                std::vector<std::vector<float>> pose_attrs;
                const int pgo_path_nav_pose_n = pgo_path_nav->poses.size();
                const int all_lidar_frames_n = all_lidar_frames.size(); // 在获取搜索树点之后再获取全部帧数量，避免新加入的帧没有被考虑
                frame_poses_updating = true; // 开始位姿更新，暂时不能用于前端，前端只能拟合建立平面
                for(int i=1; i<pgo_path_nav_pose_n; i++)
                {
                    const geometry_msgs::PoseStamped & cur_pose = pgo_path_nav->poses[i];
                    int key_frame_id_ = cur_pose.header.seq;
                    Eigen::Quaterniond current_quaternion(cur_pose.pose.orientation.w, cur_pose.pose.orientation.x, cur_pose.pose.orientation.y, cur_pose.pose.orientation.z);
                    Eigen::Vector3d current_trans = Eigen::Vector3d(cur_pose.pose.position.x, cur_pose.pose.position.y, cur_pose.pose.position.z);
                    Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();
                    current_pose.block<3,3>(0,0) = current_quaternion.toRotationMatrix();
                    current_pose.block<3,1>(0,3) = current_trans;
                    // 更新子图位姿信息
                    if(key_frame_id_to_submap_id.find(key_frame_id_) != key_frame_id_to_submap_id.end())
                    {
                        int submap_id_ = key_frame_id_to_submap_id[key_frame_id_];
                        // if(submap_infos[submap_id_].corPoseFlag) break; // 每次都得更新
                        submap_infos[submap_id_].corr_pose = current_pose;
                        submap_infos[submap_id_].corPoseFlag = true;
                        pose_pts.emplace_back(current_trans);
                        std::vector<float> pose_attr = {submap_id_};
                        pose_attrs.emplace_back(pose_attr);
                    }
                    // 更新关键帧以及普通帧位姿信息
                    int frame_id1 = all_key_frames[key_frame_id_]->frame_id;
                    Eigen::Matrix4d T_w_i = Eigen::Matrix4d::Identity();
                    T_w_i.block<3,3>(0,0) = all_lidar_frames[frame_id1]->R_w_l;
                    T_w_i.block<3,1>(0,3) = all_lidar_frames[frame_id1]->t_w_l;
                    Eigen::Matrix4d delta_T_j = current_pose*T_w_i.inverse(); // 左乘的方式看不出来增量变化
                    int frame_id2 = all_lidar_frames_n;
                    if(i+1<pgo_path_nav_pose_n)
                    {
                        int key_frame_id_2 = pgo_path_nav->poses[i+1].header.seq;
                        frame_id2 = all_key_frames[key_frame_id_2]->frame_id;
                    }
                    else delta_T_loop = delta_T_j; // delta_T_loop 用于更新没来及更新的帧位姿
                    for(int j=frame_id1; j<frame_id2; j++)
                    {
                        Eigen::Matrix4d T_w_j = Eigen::Matrix4d::Identity();
                        T_w_j.block<3,3>(0,0) = all_lidar_frames[j]->R_w_l;
                        T_w_j.block<3,1>(0,3) = all_lidar_frames[j]->t_w_l;
                        Eigen::Matrix4d T_w_j_update = delta_T_j*T_w_j;
                        all_lidar_frames[j]->R_w_l_loop_updated = T_w_j_update.block<3,3>(0,0);
                        all_lidar_frames[j]->t_w_l_loop_updated = T_w_j_update.block<3,1>(0,3);
                        all_lidar_frames[j]->R_w_l_loop = T_w_j_update.block<3,3>(0,0);
                        all_lidar_frames[j]->t_w_l_loop = T_w_j_update.block<3,1>(0,3);
                        all_lidar_frames[j]->updated_loop_ids.push_back(used_loop_id);
                    }
                }
                // 2. 更新子图位姿搜索树,用于搜索附近子图
                if(octree_pose_ptr) delete octree_pose_ptr;
                octree_pose_ptr = new thuni::Octree();
                octree_pose_ptr->update_with_attr(pose_pts, pose_attrs);
                // 3. 重构搜索树
                std::cout<<"重构搜索树\n";
                large_scale_env_tree_state = -1; // 这里到底要不要用之前累计的点？？先留着吧，后面需要再说
                std::cout<<"old octree_feature_replace_ptr->size(): "<<octree_feature_replace_ptr->size()<<std::endl;
                // 由树中的点重构树
                double get_tree_rebuild_data_t = omp_get_wtime();
                thuni::Octree * octree_feature_replace_ptr2del = octree_feature_replace_ptr; // 先保存，后面慢慢删除
                octree_feature_replace_ptr = new thuni::Octree();
                octree_feature_replace_ptr->set_min_extent(tree_ds_size);
                octree_feature_replace_ptr->set_bucket_size(1);
                // 删除树中的点
                double octree_feature_clear_t = omp_get_wtime();
                double pts_delta_t = omp_get_wtime();
                double update_with_attr_t = omp_get_wtime();
                std::set<int> tree_pts_add_tmp_;
                // 对于回环检测优化之后的子图，如果这些子图和最新关键帧较近，则将这些子图中的树点加入搜索树（如果octree_feature_ptr_不为空则直接加入，否则先保存，前端更新搜索树的时候再加入）
                add_nearby_historical_tree_pts(tree_pts_add_tmp_, octree_feature_replace_ptr, lc_prev_submap_idx); // 这里居然会出现没有点的情况！！！！！！！
                std::cout<<"delta_T_loop\n"<<delta_T_loop<<std::endl;
                std::cout<<"new octree_feature_replace_ptr->size(): "<<octree_feature_replace_ptr->size()<<std::endl;
                // wait_for_save_data = true;
                // getchar();
                loop_closure_optimized = true; // 等回环相关的所有状态更新之后再置为false，设置这个状态之后，回环检测在前端就开始生效了
                // wait_for_save_data = true;
                double update_with_attr_end_t = omp_get_wtime();
                // 后面慢慢删除
                if(octree_feature_replace_ptr2del != nullptr)
                {
                    delete octree_feature_replace_ptr2del;
                }
                std::cout<<"get_tree_rebuild_data: "<<octree_feature_clear_t-get_tree_rebuild_data_t<<std::endl; // 0.0947236
                std::cout<<"octree_feature_clear: "<<pts_delta_t-octree_feature_clear_t<<std::endl; // 
                std::cout<<"pts_delta: "<<update_with_attr_t-pts_delta_t<<std::endl; // 0.0027146
                std::cout<<"update_with_attr: "<<update_with_attr_end_t-update_with_attr_t<<std::endl; // 0.0639246
                // std::cout<<"loop_closure_time: "<<update_with_attr_end_t-loop_detect_t<<std::endl; // 0.1613629
                // get_tree_rebuild_data: 0.0901092
                // octree_feature_clear: 0.0000001
                // pts_delta: 0.0026765
                // update_with_attr: 0.0632728
                // total_time: 0.1560586
                // std::cout<<"getchar"<<std::endl;
                // getchar();
            }
            mtx_tree_pts_delay_buffer.lock();
            std::deque<std::vector<Eigen::Vector3d>> tree_pts_delay2 = tree_pts_delay;
            std::deque<std::vector<std::vector<float>>> tree_pts_attrs_delay2 = tree_pts_attrs_delay;
            tree_pts_delay.clear();
            tree_pts_attrs_delay.clear();
            mtx_tree_pts_delay_buffer.unlock();
            int tree_pts_delay_n = tree_pts_delay2.size();
            if(!decouple_front)
            {
                for(int i=0; i<tree_pts_delay_n; i++)
                {
                    if(tree_pts_attrs_delay2[i].size()<1) continue;
                    int frame_id_ = tree_pts_attrs_delay2[i][0][0];
                    if(frame_id_>last_ds2map_frame_id)
                    {
                        down_sampling_voxel_unmap(feat_cloud, tree_pts_delay2[i], tree_pts_attrs_delay2[i], 0.5);
                        last_ds2map_frame_id = frame_id_;
                    }
                }
                if(feat_cloud_need_update && !loop_closure_optimized) // 当前累计的子图点云需要更新且前端已经更新完毕
                {
                    feat_cloud_need_update = false;
                    double voxel_size_ = 0.5;
                    std::unordered_map<VOXEL_LOC, std::vector<float>> feat_cloud_tmp = std::move(feat_cloud);
                    feat_cloud.clear();
                    for (auto iter = feat_cloud_tmp.begin(); iter != feat_cloud_tmp.end(); ++iter)
                    {
                        const int frame_id_ = int(iter->second[3]);
                        Eigen::Vector3d pt_ = Eigen::Vector3d(iter->second[0], iter->second[1], iter->second[2]);
                        if(frame_id_<lc_updated_front_end_frame_id) // 回环更新后的帧
                            pt_ = delta_T_loop.block<3,3>(0,0)*pt_ + delta_T_loop.block<3,1>(0,3);
                        int64_t x = std::round(pt_[0]/voxel_size_);
                        int64_t y = std::round(pt_[1]/voxel_size_);
                        int64_t z = std::round(pt_[2]/voxel_size_);
                        VOXEL_LOC position(x, y, z);
                        if (feat_cloud.find(position) != feat_cloud.end()) continue;
                        std::vector<float> pt_2 = {pt_[0], pt_[1], pt_[2], iter->second[3], iter->second[4]};
                        feat_cloud[position] = pt_2;
                    }
                }
            }
            if(!loop_closure_optimized && large_scale_env_tree_state==-1 && octree_feature_ptr->size()>1e6 && tree_pts_delay_n>0) // 达到开始某个阈值，比如1e6，开始准备替换
            {
                large_scale_env_tree_state = 0;
                if(octree_feature_replace_ptr != nullptr) delete octree_feature_replace_ptr;
                octree_feature_replace_ptr = nullptr;
                octree_feature_replace_ptr = new thuni::Octree();
                octree_feature_replace_ptr->set_min_extent(tree_ds_size);
                octree_feature_replace_ptr->set_bucket_size(1);
                next_tree_start_frame_id = tree_pts_attrs_delay2[0][0][0]; // 下一个搜索树的起始帧id
                if(!decouple_front) historical_tree_attrs_added_replace.clear();
            }
            if(large_scale_env_tree_state == 0 && tree_pts_delay_n>0)
            {
                for(int i=0; i<tree_pts_delay_n; i++)
                {
                    if(tree_pts_delay2[i].size()<10 || tree_pts_delay2[i].size()!=tree_pts_attrs_delay2[i].size()) continue;
                    octree_feature_replace_ptr->update_with_attr(tree_pts_delay2[i], tree_pts_attrs_delay2[i], true);
                }
                if(tree_pts_attrs_delay2[tree_pts_delay_n-1][0][0]-next_tree_start_frame_id>100 && octree_feature_replace_ptr->size()>150000) // 150000
                {
                    if(!decouple_front) add_nearby_historical_tree_pts(historical_tree_attrs_added_replace, octree_feature_replace_ptr);
                    large_scale_env_tree_state = 1; // 替换树构建完毕，可以替换
                    curr_tree_start_frame_id = next_tree_start_frame_id; // 即将成为下一个搜索树的起始帧id
                    if(!decouple_front) 
                    {
                        historical_tree_attrs_added_submap_ids = historical_tree_attrs_added_replace;
                        historical_tree_attrs_added_replace.clear();
                    }
                }
            }
            tree_pts_delay2.clear();
            tree_pts_attrs_delay2.clear();
        }
        double tree_replace_t = omp_get_wtime();
        // 选择关键帧
        // std::cout<<"选择关键帧: "<<last_lidar_frame_id<<", all_lidar_frames.size(): "<<all_lidar_frames.size()<<std::endl;
        if(last_lidar_frame_id==0) lidar_frame->updated = true;
        else
        {
            // LidarFrame * last_key_frame = all_lidar_frames[last_key_frame_id];
            Eigen::Vector3d delta_t_ = lidar_frame->t_w_l_loop - last_key_frame->t_w_l_loop;
            Eigen::Matrix3d delta_R_ = last_key_frame->R_w_l_loop.transpose()*lidar_frame->R_w_l_loop;
            double delta_theta_ = SO3_LOG(delta_R_).norm()*57.3;
            if(delta_t_.norm()<0.2 && delta_theta_<20 && lidar_frame->frame_id-last_key_frame->frame_id<10)
            {
                last_lidar_frame_id++;
                lidar_frame->clear_unnecessary_data(); // 如果没有入选关键帧，则清除非必要数据
                // std::cout<<"last_lidar_frame_id: "<<last_lidar_frame_id<<std::endl;
                continue;
            }
            // std::cout<<"delta_t_.norm(): "<<delta_t_.norm()<<", delta_theta_: "<<delta_theta_<<", delta_t_: "<<delta_t_.transpose()<<std::endl;
            // std::cout<<"lidar_frame->t_w_l: "<<lidar_frame->t_w_l.transpose()<<std::endl;
            // std::cout<<"last_key_frame->t_w_l: "<<last_key_frame->t_w_l.transpose()<<std::endl;
        }
        lidar_frame->frame_type = 1; // 关键帧
        // std::cout<<"angle_split_and_pca_plane_fit"<<std::endl;
        lidar_frame->angle_split_and_pca_plane_fit(pi_split_n);
        double angle_split_t = omp_get_wtime();
        lidar_frame->key_frame_id = all_key_frames.size();
        if(use_voxel_filter)
        {
            vdf.kfs_buffer.push_back(lidar_frame);
            if(vdf.kfs_buffer.size()>100) vdf.kfs_buffer.pop_front();
        }
        all_key_frames.push_back(lidar_frame);
        double add_frame_t = omp_get_wtime();
        const int key_frames_n = all_key_frames.size();
        // std::cout<<"key_frame_id: "<<lidar_frame->key_frame_id<<", t_w_l: "<<lidar_frame->t_w_l.transpose()<<", frame_id: "<<lidar_frame->frame_id<<std::endl;
        publish_keyframe(lidar_frame);
        // std::cout<<"clear_unnecessary_data"<<std::endl;
        bool exclude_kf = false;
        int keep_kf_num = 50;
        // lidar_frame->clear_unnecessary_data(exclude_kf); // 对于关键帧，也清除部分数据
        if(all_key_frames.size()>keep_kf_num) all_key_frames[all_key_frames.size()-keep_kf_num]->clear_unnecessary_data(exclude_kf); // 暂时输出一下结果(TODO)
        if(lidar_frame->key_frame_id>0) kf_path_length += (lidar_frame->t_w_l_loop - last_key_frame->t_w_l_loop).norm();
        // std::cout<<"key_frame_id: "<<lidar_frame->key_frame_id<<", kf_path_length: "<<kf_path_length<<std::endl;
        // last_key_frame_id = last_lidar_frame_id;
        last_lidar_frame_id++;
        last_key_frame = lidar_frame;
        rate.sleep(); // 避免CPU满载
        // std::cout<<"sleep"<<std::endl;
    }
}

