# GVHMR2PBHC
# GVHMR → PBHC Motion Processing Pipeline

![Pipeline Diagram](https://via.placeholder.com/800x200?text=GVHMR+.pt+→+NPZ+→+PBHC+Retargeting)

## 🔄 Workflow Summary
1. **GVHMR Extraction**: MP4 → SMPL (.pt)  
2. **Format Conversion**: .pt → .npz  
3. **PBHC Retargeting**: .npz → Training-ready .pkl  

## 🛠️ Step-by-Step Guide

### 1. Run GVHMR Motion Extraction
```bash
python tools/demo/demo.py --video=/path/to/input.mp4 -s
