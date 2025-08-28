# SPLIN
## SPLIN: A Structured Plane-aware LiDAR-Inertial SLAM with Incremental Dynamic Object Removal and Covariance-aware Optimization 

## 1. Introduction
**SPLIN** a robust and efficient LiDAR-inertial SLAM framework featuring two key innovations. First, we introduce Incremental Static-referenced Dynamic Object Removal (ISDOR), a coarse-to-fine dynamic object removal strategy that incrementally constructs a voxel-based static map and detects dynamic objects at their first appearance, significantly enhancing front-end robustness while maintaining computational efficiency. Second, we propose a spherically-tessellated plane aggregation method tailored to LiDAR’s native radial geometry for efficient plane extraction, and a tightly coupled Point-Plane LiDAR-Inertial Odometry (PPLIO) system based on an iterated Kalman filter. This filter jointly models the Gaussian distribution of system states and residuals, enabling principled uncertainty propagation via cross-covariances and closed-form posterior updates. The back-end further integrates a covariance-aware factor graph where front-end uncertainty is preserved as information matrices, improving global consistency and long-term localization accuracy.
<div align="center">
<img src="eval_data/pictures/overview.png" width = 98% />
</div>

## 2. We will release the complete code upon acceptance of the paper.

## 3. Evaluation
Run:
```shell
python eval_rmse.py  
```
Expected output:
```c
data_dir:  eval_data/MulRan/DCC01
['poses_gt.txt', 'poses_fast_lio_sc.txt', 'poses_lio_sam_sc.txt', 'poses_ltaom.txt', 'poses_splin.txt']
time_len:  [540.5850977897644, 546.1032583713531, 545.9923703670502, 531.5999147891998, 531.9999170303345]
rmses:  [0.0, 7.580212, 8.615076, 5.344614, 4.768432]
alg_names:  ['fast_lio_sc', 'lio_sam_sc', 'ltaom', 'splin']
excu_time:  [25.542918725171543, 117.86952996389891, 13.969675573208159, 29.773599097146985]
GT          :  4.610385564963174e-10
FAST-LIO-SC :  7.580212541411045
LIO-SAM-SC  :  8.615076866506188
LTA-OM      :  5.344615362293383
SPLIN       :  4.768434514692518
```
<div align="center">
<img src="eval_data/pictures/screenshot0.png" width = 45% />
</div>
<div align="center">
<img src="eval_data/pictures/fig_ape_traj3d_bar_noGT.png" width = 98% />
</div>
<!-- <div align="center">
<img src="eval_data/pictures/Figure_1.png" width = 98% />
</div> -->
<!-- <div align="center">
<img src="eval_data/pictures/Figure_2.png" width = 98% />
</div> -->

## 4. Acknowledgments
Thanks to Fu Zhang, et al. for open-sourcing their excellent work [LTAOM](https://github.com/hku-mars/LTAOM/tree/main) and [R3LIVE](https://github.com/hku-mars/r3live).

## License
The source code is released under [GPLv2](http://www.gnu.org/licenses/) license.