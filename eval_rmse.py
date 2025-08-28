import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial.transform import Rotation
import subprocess
import re
import matplotlib as mpl
def set_top_tier_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "font.size": 20,                 # 论文可读字号
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "lines.linewidth": 1.8,          # 更粗线条
        "lines.markersize": 3.5,
        "pdf.fonttype": 42,              # 可编辑文字
        "ps.fonttype": 42,
        "text.usetex": False,
        "font.family": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })
set_top_tier_style()


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
    return poses_pair_np

data_dir = "eval_data/MulRan/DCC01" #* 8:51
show_plot = 1
show2d  = 0
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
    pass
if "NCLT" in data_dir and not os.path.exists(os.path.join(data_dir,'poses_gt.txt')):
    lines = []
    gt_files = [f for f in os.listdir(data_dir) if 'groundtruth' in f and '.csv' in f]
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
for i in range(len(ref_poses)): ref_t_xyz_rpy[i,4:7] = Rotation.from_quat(ref_poses[i,4:8]).as_euler('xyz')
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
# print("time_len: ", time_len)
pose_files_t_cut = []
for i in range(len(poses_list)):
    if i is not ref_data_id:
        mask_ = (time_start <= poses_list[i][:,0]) * (time_end >= poses_list[i][:,0])
        poses_list[i] = poses_list[i][mask_]
        np.savetxt(os.path.join(data_dir, data_names[i]+'_t_cut.txt'), poses_list[i][:,:8],fmt='%.7f')
        pose_files_t_cut.append(os.path.join(data_dir, data_names[i]+'_t_cut.txt'))
    else:
        mask_ = (time_start <= ref_t_xyz_rpy[:,0]) * (time_end >= ref_t_xyz_rpy[:,0])
        ref_t_xyz_rpy_cut = ref_t_xyz_rpy[mask_]
        poses_list[i] = poses_list[i][mask_]
        np.savetxt(os.path.join(data_dir, data_names[i]+'_t_cut.txt'), poses_list[i][:,:8],fmt='%.7f')
        pose_files_t_cut.append(os.path.join(data_dir, data_names[i]+'_t_cut.txt'))
ref_file = [os.path.join(data_dir, f) for f in pose_files if 'gt.txt' in f or 'gt.csv' in f][0]
src_files = [os.path.join(data_dir, f) for f in pose_files]
if share_time:
    src_files = pose_files_t_cut
    ref_file = pose_files_t_cut[ref_data_id]
    ref_t_xyz_rpy = ref_t_xyz_rpy_cut
poses_list = [np.loadtxt(f) for f in src_files]
for i in range(len(src_files)):
    if poses_list[i].shape[1]==9:
        np.savetxt(src_files[i], poses_list[i][:,:8],fmt='%.7f')
cmd = 'evo_ape tum  -a   '+ref_file
for i in range(len(src_files)):
    pf = src_files[i]
    if 'gt_t_cut.txt' in pf or "gt.txt" in pf:
        poses_align[i] = [np.eye(3), np.zeros(3), 0.0]
        continue
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
        translation_array = np.array(list(map(float, translation_match.group(1).split())))
        rmse = float(ape_match.group(1))
        poses_align[i] = [rotation_matrix, translation_array, rmse]
        rmses[i] = rmse
    pass
print("rmses: ", rmses)
print("alg_names: ", alg_names)
print("excu_time: ", [v[:,1].mean()*1000 for v in excu_times])
if not show_plot : exit()
excu_times_ext = []
for i in range(len(data_names)):
    find_alg = False
    for j in range(len(alg_names)):
        if alg_names[j] in data_names[i]:
            excu_times_ext.append(excu_times[j])
            find_alg = True
            break
    if not find_alg: excu_times_ext.append([])
alg_names = data_names
for i in range(len(alg_names)):
    if 'fast_lio_sc' in alg_names[i]: alg_names[i] = 'FAST-LIO-SC'
    if 'lio_sam_sc' in alg_names[i]: alg_names[i] = 'LIO-SAM-SC'
    if 'ltaom' in alg_names[i]: alg_names[i] = 'LTA-OM'
    if 'splin' in alg_names[i]: alg_names[i] = 'SPLIN'
    if 'gt' in alg_names[i]: alg_names[i] = 'GT'
start_t = ref_poses[0,0]
cur_t_xyz_rpy = []
similar_trans=[]
for i in range(len(poses_list)):
    t_xyz_rpy = []
    R, s, T = np.eye(3), 1.0, np.zeros(3)
    if i is not ref_data_id and 'imu' not in data_names[i]:
        cur_poses = poses_list[i]
        if len(poses_align[i])==0:
            print("No poses_align !!!")
            exit()
        else:
            R, T = poses_align[i][:2]
            s = 1.0
        t_xyz_rpy = np.zeros((len(cur_poses),7))
        t_xyz_rpy[:,0] = cur_poses[:,0]
        for j in range(len(cur_poses)):
            t_xyz_rpy[j,1:4] = s*R@cur_poses[j,1:4] + T
            cur_rot = Rotation.from_quat(cur_poses[j,4:8]).as_matrix()
            t_xyz_rpy[j,4:7] = Rotation.from_matrix(R@cur_rot).as_euler('xyz')
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
        poses_pair_np = pose_pair(ref_t_xyz_rpy, cur_t_xyz_rpy[i], 0.04)
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
max_str_len=0
for i in range(len(data_names)):
    if max_str_len<len(data_names[i]): max_str_len=len(data_names[i])
for i in range(len(data_names)): print(data_names[i].ljust(max_str_len, ' '),': ', rmse[i])
ylabel_names = ['x(m)','y(m)','z(m)','ape(m)']
# ---------- Trajectory plot (3D or 2D) ----------
if not show2d:
    fig_traj = plt.figure(figsize=(8.5, 5.8))
    ax1 = fig_traj.add_subplot(111, projection='3d')
    for j in range(len(cur_t_xyz_rpy)):
        data = cur_t_xyz_rpy[j]
        label = alg_names[j] if j < len(alg_names) else data_names[j]
        ax1.plot(data[:,1], data[:,2], data[:,3], label=label)
    ax1.view_init(elev=40, azim=-50)
    ax1.set_zlim(10, 30)
    ax1.set_xlabel("x (m)", labelpad=12) # 使用 labelpad 增加与轴的距离
    ax1.set_ylabel("y (m)", labelpad=12)
    ax1.set_zlabel("z (m)", labelpad=12)
    # ax1.set_title("Trajectories")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.85, 0.95), borderaxespad=0.)
    # ax1.legend(loc="upper right")
    fig_traj.tight_layout()
    fig_traj.savefig("fig_traj3d.pdf", bbox_inches="tight")
    fig_traj.savefig("fig_traj3d.png", bbox_inches="tight")
    pass
else:
    fig_traj2d, ax = plt.subplots(figsize=(7.2, 5.4))
    for j in range(len(cur_t_xyz_rpy)):
        data = cur_t_xyz_rpy[j]
        label = alg_names[j] if j < len(alg_names) else data_names[j]
        ax.plot(data[:,1], data[:,2], label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectories (Top View)")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    fig_traj2d.tight_layout()
    fig_traj2d.savefig("fig_traj2d.pdf", bbox_inches="tight")
    fig_traj2d.savefig("fig_traj2d.png", bbox_inches="tight")

# ---------- Compute mean RMSE per algorithm ----------
rmse_rows = []
for j in range(len(cur_t_xyz_rpy)):
    name = alg_names[j] if j < len(alg_names) else data_names[j]
    err = apes[j]   # shape: [T, 4]
    rmse_x = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    rmse_y = float(np.sqrt(np.mean(err[:, 2] ** 2)))
    rmse_z = float(np.sqrt(np.mean(err[:, 3] ** 2)))
    rmse_n = float(np.sqrt(np.mean(pos_ape[j] ** 2)))
    rmse_rows.append([name, rmse_x, rmse_y, rmse_z, rmse_n])

rmse_rows.sort(key=lambda r: r[-1])  # 按范数 RMSE 排序


try:
    _ = start_t
except NameError:
    start_t = 0.0

# 先找出 GT 的索引
gt_idx = None
names_all = [alg_names[j] if j < len(alg_names) else data_names[j] for j in range(len(cur_t_xyz_rpy))]
for j, nm in enumerate(names_all):
    if nm.strip().lower() in {"gt", "ground truth", "groundtruth"}:
        gt_idx = j
        break

# ========= 数据准备（排除 GT） =========
entries = []
n_methods = len(cur_t_xyz_rpy)
for j in range(n_methods):
    if j == gt_idx:
        continue  # 跳过 GT
    name = names_all[j]
    y = np.asarray(pos_ape[j])
    rmse_pos = float(np.sqrt(np.mean(y**2)))
    std_pos  = float(np.std(y))
    entries.append({"idx": j, "name": name, "rmse": rmse_pos, "std": std_pos})
entries = sorted(entries, key=lambda d: d["rmse"])

# ========= 布局：上 APE, 下左右 =========
fig = plt.figure(figsize=(8, 7), constrained_layout=False)
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.8], hspace=0.02, wspace=0.02)

# -------- 上：APE 曲线（跨两列，不包含 GT） --------
ax_ts = fig.add_subplot(gs[0, :])
curve_colors = []
for e in entries:
    j = e["idx"]
    t = apes[j][:,0] - start_t
    y = pos_ape[j]
    line, = ax_ts.plot(t, y, label=e["name"])
    curve_colors.append(line.get_color())
ax_ts.set_title("Absolute Trajectory Error")
ax_ts.set_ylabel("ATE (m)")
ax_ts.set_xlabel("t (s)")
ax_ts.minorticks_on()
ax_ts.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
ax_ts.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
# ax_ts.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax_ts.legend(loc="upper left")

# -------- 左下：Runtime 柱状图（单位：秒） --------
ax_rt = fig.add_subplot(gs[1,0])

runtime_vals = []
runtime_labels = []
runtime_errs = []

for e, color in zip(entries, curve_colors):
    j = e["idx"]
    # 优先取 runtimes_ms，如果没有则用时间戳差近似
    try:
        rt_ms = np.asarray(excu_times_ext[j][:,1]).astype(float)
    except Exception:
        t = np.asarray(apes[j][:,0]) - float(start_t)
        rt_ms = np.diff(t) * 1000.0 if len(t) >= 2 else np.array([])
    if rt_ms.size == 0:
        continue
    # import pdb;pdb.set_trace()
    # print(e["name"])
    rt_s = rt_ms 
    runtime_labels.append(e["name"])
    runtime_vals.append(rt_s.mean())
    runtime_errs.append(rt_s.std())

x = np.arange(len(runtime_labels))
bars_rt = ax_rt.bar(x, runtime_vals, yerr=runtime_errs, capsize=3, color=curve_colors[:len(runtime_labels)])

ax_rt.set_ylabel("Runtime (s)")
ax_rt.set_title("Average Runtime per Frame")
ax_rt.set_xticks(x)
ax_rt.set_xticklabels(runtime_labels, rotation=20, ha="right")
ax_rt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

# 在柱顶标数值（保留 3 位小数）
for rect, val in zip(bars_rt, runtime_vals):
    h = rect.get_height()
    ax_rt.annotate(f'{val:.3f}', xy=(rect.get_x()+rect.get_width()/2., h),
                   xytext=(0,3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=14)


# -------- 右下：RMSE 柱状图（不包含 GT） --------
ax_bar = fig.add_subplot(gs[1,1])
labels = [e["name"] for e in entries]
rmse   = [e["rmse"] for e in entries]
errs   = [e["std"]  for e in entries]
x = np.arange(len(labels))
bars = ax_bar.bar(x, rmse, yerr=errs, capsize=3, color=curve_colors)
ax_bar.set_ylabel(r"RMSE (m)")
ax_bar.set_title("RMSE Summary")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=20, ha="right")
ax_bar.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

for rect, val in zip(bars, rmse):
    h = rect.get_height()
    ax_bar.annotate(f'{val:.3f}', xy=(rect.get_x()+rect.get_width()/2., h),
                    xytext=(0,3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

fig.tight_layout()
fig.savefig("fig_ape_traj3d_bar_noGT.pdf", bbox_inches="tight")
fig.savefig("fig_ape_traj3d_bar_noGT.png", dpi=600,bbox_inches="tight")
plt.show()









