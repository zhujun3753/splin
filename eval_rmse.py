import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial.transform import Rotation

def pose_pair(ref_poses, cur_poses, max_time_gap=0.01):
    poses_pair = []
    gt_i = 0
    cur_i = 0
    while(cur_i<len(cur_poses)) and cur_poses[cur_i,0]+max_time_gap*2<ref_poses[gt_i,0]:
        if cur_poses[cur_i,0]+max_time_gap*2<ref_poses[gt_i,0]:
            cur_i +=1
        else:
            break
    while(cur_i<len(cur_poses)):
        cur_time = cur_poses[cur_i,0]
        best_gt_i = gt_i
        best_gap_time = abs(ref_poses[gt_i,0]-cur_time)
        while(gt_i<len(ref_poses) and ref_poses[gt_i,0]-max_time_gap*2<cur_time):
            if abs(ref_poses[gt_i,0]-cur_time)<=best_gap_time:
                best_gt_i = gt_i
                best_gap_time = abs(ref_poses[gt_i,0]-cur_time)
                gt_i+=1
            else: break
        if best_gap_time<max_time_gap:
            poses_pair.append([cur_i, best_gt_i,best_gap_time])
        cur_i+=1
        gt_i -=1
    poses_pair_np = np.asarray(poses_pair)
    # import pdb;pdb.set_trace()
    return poses_pair_np

def pose_align(pos_ref, pos_cur):
    mu_pos_cur = pos_cur.mean(0)
    mu_pos_ref = pos_ref.mean(0)
    sigma_pos_cur = ((pos_cur-mu_pos_cur)**2).sum(1).mean()
    cov = (pos_cur-mu_pos_cur).T @ (pos_ref-mu_pos_ref) / len(pos_cur)
    U, S,Vh = np.linalg.svd(cov, full_matrices=False)
    D = np.eye(len(cov))
    if np.linalg.det(U)*np.linalg.det(Vh)<0:
        D[-1,-1] = -1
    R = Vh.T @ (D @ U.T)
    s = np.trace(np.diag(S)@D)/sigma_pos_cur
    T = mu_pos_ref - s*R@mu_pos_cur
    return R,s,T

def iterated_pose_align(ref_poses, cur_poses, max_time_gap=0.01, max_iter_num=10,converge_thd=0.001, show=False):
    #* 支持的位姿格式： timestamp, x, y, z, qx, qy, qz, qw
    if ref_poses.shape[1]<4 or ref_poses.shape[1]<4:
        print("至少含有timestamp, x, y, z, 格式为Nxm, N为元素个数, m为维数")
        return None, None, None
    R_final, s_final, T_final = np.eye(3), 1.0, np.zeros(3)
    poses_pair_np = pose_pair(ref_poses, cur_poses,max_time_gap)
    # import pdb;pdb.set_trace()
    pos_cur = cur_poses[poses_pair_np[:,0].astype(int).tolist(),1:4]
    pos_ref = ref_poses[poses_pair_np[:,1].astype(int).tolist(),1:4]
    cur_poses_new = pos_cur*1.0
    for i in range(max_iter_num):
        last_R, last_s, last_T = pose_align(pos_ref*1.0, cur_poses_new*1.0)
        R_final = last_R@R_final
        s_final = last_s*s_final
        T_final = last_s*last_R@T_final + last_T
        for j in range(len(pos_cur)):
            cur_poses_new[j] = s_final*R_final@pos_cur[j] + T_final
        if show:
            fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(12, 8))
            for i in range(3):
                axes[i].plot(pos_ref[:,0],pos_ref[:,i],label='ref')
                axes[i].plot(cur_poses_new[:,0],cur_poses_new[:,1+i],label='cur')
                axes[i].grid()
                # axes[i].set_ylabel(ylabel_names[i])
                if i==0:
                    axes[i].set_title('trajectory')
                    axes[i].legend(fontsize="14")
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.xlabel('t(s)')
            plt.legend()
            plt.show()
            print('iter num: ',i, 'norm(delta T): ', np.linalg.norm(last_T))
        # import pdb;pdb.set_trace()
        if np.linalg.norm(last_T)<converge_thd:
            break
    return R_final, s_final, T_final




data_dir = "eval_data/MulRan/DCC01" #* 8:51

show_plot = 1
show2d  = 1
share_time = 0 #* 是否统一时间
show_time = 0

#* 位姿转换
if "MulRan" in data_dir and not os.path.exists(os.path.join(data_dir,'poses_gt.txt')):
    lines = []
    with open(os.path.join(data_dir, 'global_pose.csv'), 'r') as f:
        lines = f.readlines()
    poses = []
    for i in range(len(lines)):
        pose_t_xyz_quat = np.zeros(8)
        data_str = lines[i].strip().split(',')
        timestamp = int(data_str[0])*1e-9
        pose = np.eye(4)
        pose3x4 = np.array([float(v) for v in data_str[1:]]).reshape(3,4)
        pose[:3,:4] = pose3x4
        quat = Rotation.from_matrix(pose[:3,:3]).as_quat()
        pose_t_xyz_quat[0] = timestamp
        pose_t_xyz_quat[1:4] = pose[:3,3]
        pose_t_xyz_quat[4:8] = quat
        poses.append(pose_t_xyz_quat)
    poses_np =np.asarray(poses)
    np.savetxt(os.path.join(data_dir,'poses_gt.txt'), poses_np, fmt = '%.09f', newline = '\n')
    # import pdb;pdb.set_trace()
    pass

if "NCLT" in data_dir and not os.path.exists(os.path.join(data_dir,'poses_gt.txt')):
    lines = []
    gt_files = [f for f in os.listdir(data_dir) if 'groundtruth' in f and '.csv' in f]
    # import pdb;pdb.set_trace()
    gt = np.loadtxt(os.path.join(data_dir, gt_files[0]), delimiter = ",")
    pose_gt = gt[~np.isnan(gt[:,1]),:]
    # NED (North, East Down)
    x = pose_gt[:, 1]
    y = pose_gt[:, 2]
    z = pose_gt[:, 3]
    r = pose_gt[:, 4]
    p = pose_gt[:, 5]
    h = pose_gt[:, 6]
    with open(os.path.join(data_dir,'poses_gt.txt'),'w') as f:
        f.write('# timestamp(s) x y z qx qy qz qw \n')
        for i in range(len(pose_gt)):
            pose = pose_gt[i]
            rot = Rotation.from_euler('xyz', [pose[4], pose[5], pose[6]], degrees=False)
            rot_quat = rot.as_quat()
            f.write(str(pose[0]/1e6)+' ')
            f.write(str(pose[1])+' ')
            f.write(str(pose[2])+' ')
            f.write(str(pose[3])+' ')
            f.write(str(rot_quat[0])+' ')
            f.write(str(rot_quat[1])+' ')
            f.write(str(rot_quat[2])+' ')
            f.write(str(rot_quat[3])+'\n')

print('data_dir: ', data_dir)
pose_files = [f for f in os.listdir(data_dir) if 'poses_' in f and 'slam' in f or 'gt' in f or 'loop' in f]
pose_files = [f for f in os.listdir(data_dir) if 'poses_' in f and  'gt' in f or 'loop' in f]
# pose_files = [f for f in os.listdir(data_dir) if 'poses_' in f and  'gt' in f or 'pl-lio' in f]
pose_files = [f for f in os.listdir(data_dir) if 'poses_' in f]
excu_time_files = [f for f in os.listdir(data_dir) if 'exc_time_' in f]
excu_times = [np.loadtxt(os.path.join(data_dir, f)) for f in excu_time_files]
alg_names = [f.split('exc_time_')[1].split(".")[0] for f in excu_time_files]
print(pose_files)
poses_align = [[] for v in range(len(pose_files))]
rmses = [0.0 for v in range(len(pose_files))]
poses_list = [np.loadtxt(os.path.join(data_dir, f)) for f in pose_files]
data_names = [f.split('poses_')[1].split(".")[0] for f in pose_files]
if show_time:
    # print("单帧执行时间")
    plt.figure(figsize=(12, 8))
    for i in range(len(excu_times)):
        data = excu_times[i]
        plt.plot(data[:,0],data[:,1]*1e3, label=alg_names[i])
    plt.xlabel('frame time')
    plt.ylabel('excute time')
    plt.title('excute time per frame')
    plt.legend(fontsize="14")
    plt.grid()
    plt.show()
ref_data_id = None
for i in range(len(pose_files)):
    if "gt" in pose_files[i]:
        ref_data_id = i
ref_poses = poses_list[ref_data_id]
ref_t_xyz_rpy = np.zeros((len(ref_poses),7))
ref_t_xyz_rpy[:,:4] = ref_poses[:,:4]
for i in range(len(ref_poses)):
    ref_t_xyz_rpy[i,4:7] = Rotation.from_quat(ref_poses[i,4:8]).as_euler('xyz')
# 统一时间
time_start = -1
time_end = -1

time_len = []
for i in range(len(poses_list)):
    time_len.append(poses_list[i][-1,0] - poses_list[i][0,0])
    if i is not ref_data_id or 1:
        if time_start<0:
            time_start = poses_list[i][0,0]
            time_end = poses_list[i][-1,0]
        else:
            if time_start < poses_list[i][0,0]:
                time_start = poses_list[i][0,0]
            if time_end > poses_list[i][-1,0]:
                time_end = poses_list[i][-1,0]
# 修改时间
# time_end = time_start + 3000
# time_start += 1000
# invalid_time_start = time_start + 20
# invalid_time_end = time_start + 100
# print('share time :', time_end-time_start)
print("time_len: ", time_len)
pose_files_t_cut = []
for i in range(len(poses_list)):
    if i is not ref_data_id:
        mask_ = (time_start <= poses_list[i][:,0]) * (time_end >= poses_list[i][:,0])
        # invalid_mask_ = (invalid_time_start <= poses_list[i][:,0]) * (invalid_time_end >= poses_list[i][:,0])
        # mask_ *= ~invalid_mask_
        poses_list[i] = poses_list[i][mask_]
        np.savetxt(os.path.join(data_dir, data_names[i]+'_t_cut.txt'), poses_list[i][:,:8],fmt='%.7f')
        pose_files_t_cut.append(os.path.join(data_dir, data_names[i]+'_t_cut.txt'))
        # import pdb;pdb.set_trace()
    else:
        mask_ = (time_start <= ref_t_xyz_rpy[:,0]) * (time_end >= ref_t_xyz_rpy[:,0])
        # invalid_mask_ = (invalid_time_start <= ref_t_xyz_rpy[:,0]) * (invalid_time_end >= ref_t_xyz_rpy[:,0])
        # mask_ *= ~invalid_mask_
        ref_t_xyz_rpy_cut = ref_t_xyz_rpy[mask_]
        poses_list[i] = poses_list[i][mask_]
        np.savetxt(os.path.join(data_dir, data_names[i]+'_t_cut.txt'), poses_list[i][:,:8],fmt='%.7f')
        pose_files_t_cut.append(os.path.join(data_dir, data_names[i]+'_t_cut.txt'))
        # pose_files_t_cut.append(pose_files[i])
if 1:
    import subprocess
    import re
    ref_file = [os.path.join(data_dir, f) for f in pose_files if 'sam.txt' in f]
    src_files = [os.path.join(data_dir, f) for f in pose_files if 'lio.txt' in f]
    # src_files = [os.path.join(data_dir, f) for f in pose_files]
    if 0:
        ref_file = ref_file[0]
        print("位姿绘制")
        # # ref_file = '/home/zhujun/WS/UbuntuData/SLAM/Hilti/exp04_construction_upper_level/poses_gt.txt'
        cmd = 'evo_traj tum  -a -p --ref '+ref_file
        for pf in src_files:
            cmd += ' '+pf+''
        os.system(cmd)
    ref_file = [os.path.join(data_dir, f) for f in pose_files if 'gt.txt' in f or 'gt.csv' in f][0]
    # src_files = [os.path.join(data_dir, f) for f in pose_files if 'gt.txt' not in f]
    src_files = [os.path.join(data_dir, f) for f in pose_files]
    if share_time:
        src_files = pose_files_t_cut
        ref_file = pose_files_t_cut[ref_data_id]
        ref_t_xyz_rpy = ref_t_xyz_rpy_cut
    # print("位姿评估")
    # ref_file = None
    poses_list = [np.loadtxt(f) for f in src_files]
    for i in range(len(src_files)):
        if poses_list[i].shape[1]==9:
            # import pdb;pdb.set_trace()
            np.savetxt(src_files[i], poses_list[i][:,:8],fmt='%.7f')
    # cmd = 'evo_ape tum  -a -p -va  '+ref_file
    cmd = 'evo_ape tum  -a   '+ref_file
    for i in range(len(src_files)):
        pf = src_files[i]
        if 'gt_t_cut.txt' in pf or "gt.txt" in pf:
            poses_align[i] = [np.eye(3), np.zeros(3), 0.0]
            continue
        # print(pf)
        # command = ['evo_ape', 'tum', ref_file, pf, '-a', '-va', '-p']
        command = ['evo_ape', 'tum', ref_file, pf, '-a', '-va']
        result = subprocess.run(command, capture_output=True, text=True)
        # 1. 提取 Rotation of alignment 矩阵
        rotation_pattern = r'Rotation of alignment:\n\[(.*)\]\nTranslation' 
        translation_pattern = r'Translation of alignment:\n\[(.*)\]\nScale' 
        ape_pattern = r'rmse\s*([\d.]+)'
        rotation_match = re.search(rotation_pattern, result.stdout, re.DOTALL)
        translation_match = re.search(translation_pattern, result.stdout)
        ape_match = re.search(ape_pattern, result.stdout)
        if rotation_match and translation_match and ape_match:
            rotation_matrix_str = rotation_match.group(1)
            rotation_matrix = np.array([list(map(float, row.replace('[','').replace(']','').split())) for row in rotation_matrix_str.split("\n")])
            # print("Rotation Matrix:\n", rotation_matrix)
            translation_array = np.array(list(map(float, translation_match.group(1).split())))
            # print("Translation Array:", translation_array)
            rmse = float(ape_match.group(1))
            # print("RMSE:", rmse)
            poses_align[i] = [rotation_matrix, translation_array, rmse]
            rmses[i] = rmse
        # import pdb;pdb.set_trace()
        # os.system(cmd + ' '+pf+'')
        pass
    # exit()
    print("rmses: ", rmses)
    print("alg_names: ", alg_names)
    print("excu_time: ", [v[:,1].mean()*1000 for v in excu_times])
    # import pdb;pdb.set_trace()
    if not show_plot : exit()


bias_files = [f for f in os.listdir(data_dir) if 'bias_' in f and '.txt' in f]
bias = [np.loadtxt(os.path.join(data_dir, f)) for f in bias_files]
bias_data_names = [f.split('bias_')[1].split(".")[0] for f in bias_files]
# import pdb;pdb.set_trace()


feats_down_size_files = [f for f in os.listdir(data_dir) if 'feats_down_size_' in f]
feats_down_size_list = [np.loadtxt(os.path.join(data_dir, f)) for f in feats_down_size_files]
feats_down_size_data_names = [f.split('feats_down_size_')[1].split(".")[0] for f in feats_down_size_files]
# frame_trans_certainty = np.loadtxt('/media/zhujun/0DFD06D20DFD06D2/catkin_ws/src/PO-LIO/frame_trans_certainty.txt')
# key_frame_trans_certainty = np.loadtxt('/media/zhujun/0DFD06D20DFD06D2/catkin_ws/src/PO-LIO/key_frame_trans_certainty.txt')
# print("位姿对齐")

start_t = ref_poses[0,0]
cur_t_xyz_rpy = []
similar_trans=[]

# import pdb;pdb.set_trace()
for i in range(len(poses_list)):
    t_xyz_rpy = []
    R, s, T = np.eye(3), 1.0, np.zeros(3)
    if i is not ref_data_id and 'imu' not in data_names[i]:
        cur_poses = poses_list[i]
        # print(data_names[i])
        if len(poses_align[i])==0 or 0:
            # import pdb;pdb.set_trace()
            R, s, T = iterated_pose_align(ref_poses, cur_poses)
        else:
            R, T = poses_align[i][:2]
            s = 1.0
        # print(data_names[i])
        # print('R: ',R)
        # print('s: ',s)
        # print('T: ',T)
        # import pdb;pdb.set_trace()
        t_xyz_rpy = np.zeros((len(cur_poses),7))
        t_xyz_rpy[:,0] = cur_poses[:,0]
        for j in range(len(cur_poses)):
            t_xyz_rpy[j,1:4] = s*R@cur_poses[j,1:4] + T
            cur_rot = Rotation.from_quat(cur_poses[j,4:8]).as_matrix()
            t_xyz_rpy[j,4:7] = Rotation.from_matrix(R@cur_rot).as_euler('xyz')
            # import pdb;pdb.set_trace()
            # break
    cur_t_xyz_rpy.append(t_xyz_rpy)
    similar_trans.append([R, s, T])
cur_t_xyz_rpy[ref_data_id] = ref_t_xyz_rpy
for i in range(len(poses_list)):
    if 'imu' in data_names[i]:
        cur_poses = poses_list[i]
        cor_i = 0
        cor_name = data_names[i].split('_')[0]
        for j in range(len(poses_list)):
            if cor_name in data_names[j] and 'imu' not in data_names[j]:
                cor_i = j
                break
        R, s, T = similar_trans[cor_i]
        # import pdb;pdb.set_trace()
        t_xyz_rpy = np.zeros((len(cur_poses),7))
        t_xyz_rpy[:,0] = cur_poses[:,0]
        for j in range(len(cur_poses)):
            t_xyz_rpy[j,1:4] = s*R@cur_poses[j,1:4] + T
            cur_rot = Rotation.from_quat(cur_poses[j,4:8]).as_matrix()
            t_xyz_rpy[j,4:7] = Rotation.from_matrix(R@cur_rot).as_euler('xyz')
        cur_t_xyz_rpy[i] = t_xyz_rpy
apes = []
for i in range(len(cur_t_xyz_rpy)):
    delta_Ts = []
    if i is not ref_data_id or 1:
        # import pdb;pdb.set_trace()
        poses_pair_np = pose_pair(ref_t_xyz_rpy, cur_t_xyz_rpy[i], 0.04)
        # import pdb;pdb.set_trace()
        pos_cur = cur_t_xyz_rpy[i][poses_pair_np[:,0].astype(int).tolist(),:]
        pos_ref = ref_t_xyz_rpy[poses_pair_np[:,1].astype(int).tolist(),:]
        delta_Ts = pos_cur*1.0
        for j in range(len(pos_cur)):
            cur_T = np.eye(4)
            ref_T = np.eye(4)
            cur_T[:3,:3] = Rotation.from_euler('xyz', pos_cur[j,4:7], degrees=False).as_matrix()
            ref_T[:3,:3] = Rotation.from_euler('xyz', pos_ref[j,4:7], degrees=False).as_matrix()
            cur_T[:3,3] = pos_cur[j,1:4]
            ref_T[:3,3] = pos_ref[j,1:4]
            delta_T = np.linalg.inv(ref_T)@cur_T
            delta_Ts[j,1:4] = delta_T[:3,3]
            delta_Ts[j,4:7] = Rotation.from_matrix(delta_T[:3,:3]).as_euler('xyz')
    apes.append(delta_Ts)
pos_ape = [np.sqrt(((ape[:,1:4])**2).sum(1)) for ape in apes]
mean_pos_ape = [p.mean() for p in pos_ape]
median_pos_ape = [np.median(p) for p in pos_ape]
std_pos_ape = [np.std(p) for p in pos_ape]
rmse = [np.sqrt(((ape[:,1:4])**2).sum(1).sum(0)/len(ape)) for ape in apes]
# import pdb;pdb.set_trace()
# print("数据名称：",data_names)
# print("绝对平移误差均值：",mean_pos_ape)
# print("绝对平移误差中值：",median_pos_ape)
# print("绝对平移误差标准差：",std_pos_ape)
# print("RMSE: ",rmse)
# print(excu_time_files, [v[:,1].mean()*1000 for v in excu_times])
max_str_len=0
for i in range(len(data_names)):
    if max_str_len<len(data_names[i]): max_str_len=len(data_names[i])
for i in range(len(data_names)):
    print(data_names[i].ljust(max_str_len, ' '),': ', rmse[i])

print("数据名称：",data_names, file=open(os.path.join(data_dir,'readme.md'),'w'))
print("绝对平移误差均值：",mean_pos_ape, file=open(os.path.join(data_dir,'readme.md'),'a'))
print("绝对平移误差中值：",median_pos_ape, file=open(os.path.join(data_dir,'readme.md'),'a'))
print("绝对平移误差标准差：",std_pos_ape, file=open(os.path.join(data_dir,'readme.md'),'a'))
print("RMSE: ",rmse, file=open(os.path.join(data_dir,'readme.md'),'a'))
# exit(0)
# ylabel_names = ['bax','bay','baz','bgx','bgy','bgz']
# fig, axes = plt.subplots(7, 1, sharex=True, sharey=False, figsize=(12, 8))
# for i in range(6):
#     for j in range(len(bias)):
#         data = bias[j]
#         axes[i].plot(data[:,0],data[:,1+i], label=bias_data_names[j])
#     axes[i].grid()
#     axes[i].set_ylabel(ylabel_names[i])
#     if i==0:
#         axes[i].set_title('bias')
#         axes[i].legend(fontsize="14")
# for j in range(len(cur_t_xyz_rpy)):
#     axes[6].plot(apes[j][:,0],pos_ape[j], label=data_names[j])
# axes[6].grid()
# axes[6].set_ylabel('ape(m)')
# axes[6].legend(fontsize="14")
# # plt.subplots_adjust(wspace=0, hspace=0)
# plt.xlabel('t(s)')
# # plt.tight_layout()
# fig.savefig(os.path.join(data_dir,'bias_and_ape.png'),dpi=600,format='png')
# # import pdb;pdb.set_trace()
# # print("位姿绘制")

ylabel_names = ['x(m)','y(m)','z(m)','ape(m)']
if not show2d:
    plt.figure(figsize=(12, 8))
    ax1 = plt.axes(projection='3d')
    for j in range(len(cur_t_xyz_rpy)):
        data = cur_t_xyz_rpy[j]
        ax1.plot(data[:,1],data[:,2],data[:,3], label=data_names[j])
        # ax1.plot(data[:,1],data[:,2],np.zeros_like(data[:,3]), label=data_names[j])
    ax1.legend(fontsize="14")
else:
    plt.figure(figsize=(12, 8))
    # ax1 = plt.axes(projection='3d')
    for j in range(len(cur_t_xyz_rpy)):
        data = cur_t_xyz_rpy[j]
        # ax1.plot(data[:,1],data[:,2],data[:,3], label=data_names[j])
        plt.plot(data[:,1],data[:,2],label=data_names[j])
    plt.legend(fontsize="14")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid()
# import pdb;pdb.set_trace()
fig, axes = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(12, 12))
# for i in range(3):
#     for j in range(len(cur_t_xyz_rpy)):
#         data = cur_t_xyz_rpy[j]
#         axes[i].plot(data[:,0]-start_t,data[:,1+i], label=data_names[j])
#     axes[i].grid()
#     axes[i].set_ylabel(ylabel_names[i])
#     if i==0:
#         axes[i].set_title('trajectory')
#         axes[i].legend(fontsize="14")
for i in range(3):
    for j in range(len(cur_t_xyz_rpy)):
        # data = cur_t_xyz_rpy[j]
        # axes[i].plot(data[:,0]-start_t,data[:,1+i], label=data_names[j])
        data = apes[j]
        # if i==2: data = cur_t_xyz_rpy[j]
        axes[i].plot(data[:,0]-start_t, data[:,1+i], label=data_names[j])
        # if i==0:
        #     print(data_names[j], ', start_t: ', data[0,0], ', end_t: ', data[-1,0])
    axes[i].grid()
    axes[i].set_ylabel(ylabel_names[i])
    if i==0:
        axes[i].set_title('APE')
        axes[i].legend(fontsize="10")
for j in range(len(cur_t_xyz_rpy)):
    axes[3].plot(apes[j][:,0]-start_t,pos_ape[j], label=data_names[j])
axes[3].grid()
axes[3].set_ylabel(ylabel_names[3])
# for j in range(len(poses_list)):
#     if 'pl-lio_loop' in data_names[j]:
#         data = poses_list[j]
#         axes[4].plot(data[:,0]-start_t,data[:,-1], label=data_names[j])
#         axes[4].grid()
# import pdb;pdb.set_trace()
# for j in range(len(feats_down_size_data_names)):
#     data = feats_down_size_list[j]
#     axes[4].plot(data[:,0]-start_t,data[:,1], label=feats_down_size_data_names[j])
for j in range(len(cur_t_xyz_rpy)):
    # data = cur_t_xyz_rpy[j]
    # axes[i].plot(data[:,0]-start_t,data[:,1+i], label=data_names[j])
    data = cur_t_xyz_rpy[j]
    axes[4].plot(data[:,0]-start_t, data[:,3], label=data_names[j])
axes[4].grid()
# axes[4].set_ylabel("")
# for j in range(len(feats_down_size_data_names)):
#     data = feats_down_size_list[j]
# axes[4].plot(frame_trans_certainty[:,0]-start_t,frame_trans_certainty[:,1], label='x')
# axes[4].plot(frame_trans_certainty[:,0]-start_t,frame_trans_certainty[:,2], label='y')
# import pdb;pdb.set_trace()
# mean_ = frame_trans_certainty[:,3].mean()
# len_ = len(frame_trans_certainty[:,3])
# min_vs = []
# for i in range(len_):
#     min_v = frame_trans_certainty[i,1]
#     if min_v>frame_trans_certainty[i,2]: min_v = frame_trans_certainty[i,2]
#     if min_v>frame_trans_certainty[i,3]: min_v = frame_trans_certainty[i,3]
#     min_vs.append(min_v)
# min_vs = np.asarray(min_vs)
# mean_ = min_vs.mean()
# means = [mean_ for v in range(len_)]
# axes[4].plot(frame_trans_certainty[:,0]-start_t,means, label='mean')
# mean_ = key_frame_trans_certainty[:,3].mean()*0.01
# len_ = len(key_frame_trans_certainty[:,3])
# means = [mean_ for v in range(len_)]
# axes[4].plot(frame_trans_certainty[:,0]-start_t,frame_trans_certainty[:,3], label='z')
# axes[4].plot(key_frame_trans_certainty[:,0]-start_t,key_frame_trans_certainty[:,3]*0.01, label='key')
# # axes[4].plot(key_frame_trans_certainty[:,0]-start_t,means, label='mean_key')
# axes[4].legend(fontsize="10")
# axes[4].grid()
# axes[4].set_ylabel("z_tran_cert")
plt.subplots_adjust(wspace=0, hspace=0)
plt.xlabel('t(s)')

ylabel_names = ['roll(rad)','pitch(rad)','yaw(rad)']
# fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(12, 8))
# for i in range(3):
#     for j in range(len(cur_t_xyz_rpy)):
#         data = cur_t_xyz_rpy[j]
#         axes[i].plot(data[:,0],data[:,4+i], label=data_names[j])
#     axes[i].grid()
#     axes[i].set_ylabel(ylabel_names[i])
#     if i==0:
#         axes[i].set_title('trajectory')
#         axes[i].legend(fontsize="14")
# # plt.subplots_adjust(wspace=0, hspace=0)
# plt.xlabel('t(s)')
plt.show()



# import pdb;pdb.set_trace()
# ref_file = [os.path.join(data_dir, f) for f in pose_files if 'gt.txt' in f][0]
# src_files = [os.path.join(data_dir, f) for f in pose_files if 'gt.txt' not in f]
# # # print("位姿绘制")
# # # ref_file = '/home/zhujun/WS/UbuntuData/SLAM/Hilti/exp04_construction_upper_level/poses_gt.txt'
# # cmd = 'evo_traj tum  -a -p --ref '+ref_file
# # for pf in pose_files:
# #     cmd += ' '+data_dir+pf+''
# # os.system(cmd)
# # print("位姿评估")
# # ref_file = None

# cmd = 'evo_ape tum  -a -p -va  '+ref_file
# for pf in pose_files:
#     print(pf)
#     os.system(cmd + ' '+data_dir+pf+'')


# cmd = 'evo_ape  tum /home/zhujun/WS/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/ground_truth/tum_format/gt-nc-quad-easy.csv  /home/zhujun/WS/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/slam_result/quad-easy-003/poses_poslam_ikd.txt   -a  -p'
# # os.system(cmd)

# cmd = 'evo_ape  tum /home/zhujun/WS/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/ground_truth/tum_format/gt-nc-quad-easy.csv  /home/zhujun/WS/UbuntuData/SLAM/NewerCollege/Multi-camera-Lidar-IMU/slam_result/quad-easy-003/poses_poslam_oct.txt   -a  -p'
# # os.system(cmd)
# # print(textlist)





