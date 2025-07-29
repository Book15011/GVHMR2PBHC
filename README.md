# 🚀 GVHMR2PBHC Pipeline (+ Enhanced Stability for Unitree G1 Training)

## 🆕 New Post-Retargeting Processing
### 🧠 Core Idea: Selective Motion Locking
Problem: Full-body motion causes instability during RL training  
Solution: Smart Partial Motion Freezing "Lock only the non-essential motion components while preserving key expressive movements"

## 🔄 Workflow Summary (All Converting & Retargeting in GVHMR environment)
1. **GVHMR Extraction**: MP4 → SMPL (.pt)  
2. **Format Conversion**: .pt → .npz
3. **PBHC Retargeting**: .npz → Training-ready .pkl
4. **🆕 Enhanced Stability**: .pkl → .pkl
5. **Motion Smoothing (motion_interpolation_pkl.py)**: pkl → .pkl (+ intro transition frame)
6. **Visualization (vis_q_mj.py)**
7. **Training (train_agent.py)**



## 🛠️ Step-by-Step Guide

### 1. Run GVHMR Motion Extraction [GVHMR](https://github.com/zju3dv/GVHMR) 
#### Input: input/my_video
**FPS: of mp4 must be 30 if not used some online tools to change it frist** 
Check [FPS](https://getrecut.com/variable-frame-rate-check/)
```bash
python tools/demo/demo.py --video=/path/to/input.mp4 -s
```
#### Output: output/demo/
  Should contains: Visualized MP4 & .pt files (SMPL motion data)

![hippo](https://github.com/Book15011/GVHMR2PBHC/blob/main/messi-cel2_3_incam_global_horiz(1).gif)



### 2. Verify .pt Structure
Run Openpt.py to verify the PT files structure:
```
 python Openpt.py #change path inside .py file
```
Expected output: Main Dictionary keys: ['smpl_params_global', 'smpl_params_incam', 'K_fullimg', 'net_outputs']

### 3. Convert PT to NPZ
Use converterV4.py: Modify input/output paths at bottom of the script
Run to convert .pt → .npz
```
python Converter_V4.py
```

### 4. Verify NPZ Files
Run Checker.py to validate NPZ files:
```
python Checker.py 
```
Should get results like this **betas shape: (10,), poses shape: (235, 72), trans shape: (235, 3), gender: neutral**

### 5. Convert NPZ to Retargeted Motion (PBHC Project)
Use convert_fit_V2.py. Download to IsaacGym_Preview_4_Package/PBHC/smpl_retarget/mink_retarget
```
python mink_retarget/convert_fit_motion_V2.py PBHC/example/motion_data/
```
**NO NEED to specific path just end at /motion_data*
**File Structure Requirements:**
```
PBHC/
  └── example/
      └── motion_data/
          └── videomotion/
              └── processed_data/  # Place NPZ file here (only one files in folder)
```
**Note:**
  - For reprocessing same file: either rename or delete previous outputs and related files
  - Outputs go to smpl_retarget/retargeted_motion_data/mink/ as .pkl files
  - The original convert_fit_motion.py can retarget motion, but its output is incompatible with our pipeline due to axis misalignment (swapped Y/Z) and incorrect parameter types. The modified convert_fit_V2.py fixes these issues specifically for our workflow, ensuring valid inputs for train_agent.py.

### 6.Motion Modification 
Use modify_motion.py to freeze body parts:
```
python modify_motion.py /path/to/retargeted.pkl \
  --fix-part lower \
  --split-index
```

**Options: lower, upper, root, all, legs_and_root, legs_static_waist_dynamic (modifiable to fix parts you want)**

| Option                                                                                            | What It                              | Best For                                                                       |   
|---------------------------------------------------------------------------------------------------|--------------------------------------|--------------------------------------------------------------------------------|
| lower                                                                                             | Legs (0-11)                          | Focus on upper body movements                                                  |
| upper                                                                                             | Arms/Waist (12-22)                   | Focus on lower body movements                                                  |
| root                                                                                              | Root position/rotation only          | Stationary upper+lower body                                                    | 
| all                                                                                               | Everything (full freeze)             | Debugging                                                                      |
| legs_and_root                                                                                     | Legs + Full root (position+rotation) | Ultra-stable locomotion                                                        |  
| legs_static_waist_dynamic                                                                         | Legs                                 | Focue on upper body (waist can move only 2 legs freezed, different from lower) |  


You can modified the code to Locked some parts yourself this is **Label of Each Joint**
| Joint Index                                                                                       | Joint Name        | limit(rad)               |  
|---------------------------------------------------------------------------------------------------|-------------------|--------------------------|
| 0                                                                                                 | L_LEG_HIP_PITCH   | -2.5307~2.8798           | 
| 1                                                                                                 | L_LEG_HIP_ROLL    | -0.5236~2.9671           |   
| 2                                                                                                 | L_LEG_HIP_YAW     | -2.7576~2.7576           |   
| 3                                                                                                 | L_LEG_KNEE        | -0.087267~2.8798         |   
| 4                                                                                                 | L_LEG_ANKLE_PITCH | -0.87267~0.5236          |  
| 5                                                                                                 | L_LEG_ANKLE_ROLL  | -0.2618~0.2618           |   
| 6                                                                                                 | R_LEG_HIP_PITCH   | -2.5307~2.8798           |   
| 7                                                                                                 | R_LEG_HIP_ROLL    | -2.9671~0.5236           |  
| 8                                                                                                 | R_LEG_HIP_YAW     | -2.7576~2.7576           |
| 9                                                                                                 | R_LEG_KNEE        | -0.087267~2.8798         |  
| 10                                                                                                | R_LEG_ANKLE_PITCH | -0.87267~0.5236          |  
| 11                                                                                                | R_LEG_ANKLE_ROLL  | -0.2618~0.2618           |  
| 12                                                                                                | WAIST_YAW         | -2.618~2.618             |  
| 13                                                                                                | WAIST_ROLL        | -0.52~0.52               |   
| 14                                                                                                | WAIST_PITCH       | -0.52~0.52               |  
| 15                                                                                                | L_SHOULDER_PITCH  | -3.0892~2.6704           |   
| 16                                                                                                | L_SHOULDER_ROLL   | -1.5882~2.2515           |  
| 17                                                                                                | L_SHOULDER_YAW    | -2.618~2.618             |  
| 18                                                                                                | L_ELBOW           | -1.0472~2.0944           |   
| 19                                                                                                | L_WRIST_ROLL      | -1.972222054~1.972222054 |   
| 20                                                                                                | L_WRIST_PITCH     | -1.614429558~1.614429558 |  
| 21                                                                                                | L_WRIST_YAW       | -1.614429558~1.614429558 |   
| 22                                                                                                | R_SHOULDER_PITCH  | -3.0892~2.6704           |  
| 23                                                                                                | R_SHOULDER_ROLL   | -2.2515~1.5882           |  
| 24                                                                                                | R_SHOULDER_YAW    | -2.618~2.618             |  
| 25                                                                                                | R_ELBOW           | -1.0472~2.0944           |   
| 26                                                                                                | R_WRIST_ROLL      | -1.972222054~1.972222054 |  
| 27                                                                                                | R_WRIST_PITCH     | -1.614429558~1.614429558 |  
| 28                                                                                                | R_WRIST_YAW       | -1.614429558~1.614429558 |   



 Calculate Formula (**!!Consider the max angle of that Joint. Exceed it will damage the motor!!""**)
```
Angle_rad=Angle_deg×(180 / pi)
```

**Key Features:**
  - Humanly Impossible Motions (Example: 90° waist twists without leg movement. How? The script preserves original joint angles mathematically, even if biomechanically unrealistic.)
  - Gradual Control Adjust --split-index to change where "upper" and "lower" body split (default=12).

Example (legs_static_waist_dynamic)

### 7. Motion Inspection
Use motion_readpkl_V2.py (updated from original) to debug motion files:
```
# Full analysis
python motion_readpkl_V2.py /path/to/motion.pkl --output analysis.txt

# Extract specific data
python motion_readpkl_V2.py /path/to/motion.pkl --output data.csv --keys dof,root_trans_offse
```

### 8. Motion Smoothing & Interpolation (Same files from PBHC)
Use motion_interpolation_pkl.py to add smooth transitions:
```
python motion_interpolation_pkl.py \
  --origin_file_name /path/to/Motion_modification.pkl \
  --start_inter_frame 30 \
  --end_inter_frame 30 \
  --knee_modify True
```
Creates _inter files with smooth start/end transitions

### 9. Visualize Results
  ```
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python robot_motion_process/vis_q_mj.py +motion_file=/path/to/your.pkl
```
![hippo](https://github.com/Book15011/GVHMR2PBHC/blob/main/Visual_Moton_PKL(1).gif)




### 10. Train Motion (Back to [PBHC](https://github.com/TeleHuman/PBHC/blob/main/humanoidverse/README.md) Environment and Pipeline) 
```
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/benchmark \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_nil \
+rewards=motion_tracking/main \
experiment_name=benchmark \
robot.motion.motion_file="path/to/your/retarget/result/.pkl" \
seed=1 \
+device=cuda:0
```

## Environment Notes:
Steps 1-6: Use GVHMR environment

Step 7: Use PBHC environment

