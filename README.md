# GVHMR2PBHC
# GVHMR → PBHC Motion Processing Pipeline


## 🔄 Workflow Summary
1. **GVHMR Extraction**: MP4 → SMPL (.pt)  
2. **Format Conversion**: .pt → .npz  
3. **PBHC Retargeting**: .npz → Training-ready .pkl  

## 🛠️ Step-by-Step Guide

### 1. Run GVHMR Motion Extraction (https://github.com/zju3dv/GVHMR) 
```bash
python tools/demo/demo.py --video=/path/to/input.mp4 -s

# Output: output/demo/
  contains: Visualized MP4 & .pt files (SMPL motion data)
