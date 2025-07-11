# GVHMR2PBHC PBHC Motion Processing Pipeline


## 🔄 Workflow Summary (All Converting & Retargeting in GVHMR environment)
1. **GVHMR Extraction**: MP4 → SMPL (.pt)  
2. **Format Conversion**: .pt → .npz  
3. **PBHC Retargeting**: .npz → Training-ready .pkl  

## 🛠️ Step-by-Step Guide

### 1. Run GVHMR Motion Extraction [GVHMR](https://github.com/zju3dv/GVHMR) 
#### Input: input/my_video
```bash
python tools/demo/demo.py --video=/path/to/input.mp4 -s
```

#### Output: output/demo/
  Should contains: Visualized MP4 & .pt files (SMPL motion data)

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
python mink_retarget/convert_fit_motion_V2.py /path/to/PBHC/example/motion_data/
```
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

### 6. Visualize Results
  ```
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python robot_motion_process/vis_q_mj.py +motion_file=/path/to/your.pkl
```

### 7. Train Motion (Back to [PBHC](https://github.com/TeleHuman/PBHC/blob/main/humanoidverse/README.md) Environment and Pipeline) 
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

