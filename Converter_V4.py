import torch
import numpy as np
from pathlib import Path
import pickle

def axis_angle_to_quaternion(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    sin_half = np.sin(angle / 2)
    cos_half = np.cos(angle / 2)
    return np.array([
        cos_half,
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half
    ])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_to_axis_angle(q):
    q_norm = q / np.linalg.norm(q)
    w = q_norm[0]
    w = np.clip(w, -1.0, 1.0)
    angle = 2 * np.arccos(w)
    sin_half = np.sqrt(1 - w**2)
    if sin_half < 1e-8:
        return np.zeros(3)
    axis = q_norm[1:] / sin_half
    return axis * angle


def convert_pt_to_npz(input_path, output_dir, output_name="demo_sequence"):
    """Convert HMR4D .pt file to AMASS-compatible .npz with all parameters"""
    # Load .pt file with version compatibility
    try:
        data = torch.load(input_path, map_location='cpu', pickle_module=pickle)
    except Exception as e:
        print(f"❌ Error loading {input_path}: {str(e)}")
        if "unsupported pickle protocol" in str(e):
            print("💡 Try installing cloudpickle: pip install cloudpickle")
        return

    print("="*50)
    print("Successfully loaded PT file keys:", list(data.keys()))

    # Process SMPL global parameters
    smpl_global = data["smpl_params_global"]
    print("\n🔍 SMPL Global Parameters:")
    print(f"global_orient: {smpl_global['global_orient'].shape}")
    print(f"body_pose: {smpl_global['body_pose'].shape}")
    print(f"transl: {smpl_global['transl'].shape}")
    print(f"betas: {smpl_global['betas'].shape}")

    # Handle betas - ensure it's (10,)
    betas = smpl_global["betas"].numpy()
    if betas.ndim == 2:
        if betas.shape[0] > 1:
            print(f"⚠️ Using first frame's betas from {betas.shape}")
            betas = betas[0]
        betas = betas.squeeze()

    # Handle gender
    gender = data.get("gender", "neutral")
    if isinstance(gender, torch.Tensor):
        gender = "male" if gender.item() == 0 else "female"

    # Process motion data
    global_orient = smpl_global["global_orient"].numpy()
    body_pose = smpl_global["body_pose"].numpy()
    transl = smpl_global["transl"].numpy()
# ===== COORDINATE TRANSFORMATION: Z-up to Y-up =====
    R_fix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    transl = (R_fix @ transl.T).T  # Fix translation

    # Convert fixed rotation to quaternion
    angle_fix = np.pi / 2
    q_fix = np.array([
        np.cos(angle_fix / 2),
        np.sin(angle_fix / 2),
        0,
        0
    ])

    global_orient_fixed = np.zeros_like(global_orient)
    for i in range(global_orient.shape[0]):
        q_orig = axis_angle_to_quaternion(global_orient[i])
        q_composed = quaternion_multiply(q_fix, q_orig)
        r_new = quaternion_to_axis_angle(q_composed)
        global_orient_fixed[i] = r_new
        
    global_orient = global_orient_fixed  # Use fixed orientation
    
    # Body pose dimension adjustment
    REQUIRED_BODY_DIM = 69
    current_dim = body_pose.shape[1]
    
    if current_dim < REQUIRED_BODY_DIM:
        pad_width = REQUIRED_BODY_DIM - current_dim
        print(f"⚠️ Padding body pose from {current_dim} to {REQUIRED_BODY_DIM}")
        body_pose = np.concatenate([
            body_pose,
            np.zeros((body_pose.shape[0], pad_width))
        ], axis=1)
    elif current_dim > REQUIRED_BODY_DIM:
        print(f"⚠️ Truncating body pose from {current_dim} to {REQUIRED_BODY_DIM}")
        body_pose = body_pose[:, :REQUIRED_BODY_DIM]

    poses = np.concatenate([global_orient, body_pose], axis=1)

    # Process other parameters
    def extract_tensors(obj, prefix=""):
        """Recursively extract tensors from nested structures"""
        results = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                results.update(extract_tensors(v, f"{prefix}{k}_"))
        elif isinstance(obj, torch.Tensor):
            results[prefix[:-1]] = obj.numpy()  # Remove trailing underscore
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                results.update(extract_tensors(item, f"{prefix}{i}_"))
        else:
            results[prefix[:-1]] = obj
        return results

    # Extract all parameters
    all_params = {}
    for key in data:
        if key == "smpl_params_global":
            continue  # Already processed
        all_params.update(extract_tensors(data[key], f"{key}_"))

    print("\n📦 Extracted Additional Parameters:")
    for k, v in all_params.items():
        if isinstance(v, np.ndarray):
            print(f"- {k}: {v.shape} {v.dtype}")
        else:
            print(f"- {k}: {type(v).__name__}")

    # Prepare output data
    output_data = {
        "poses": poses.astype(np.float32),
        "trans": transl.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": gender,
        "mocap_framerate": np.array(30),
        "K_fullimg": data["K_fullimg"].numpy().astype(np.float32),
        **all_params  # Include all additional parameters
    }

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.npz"

    # Save as compressed NPZ
    np.savez_compressed(output_path, **output_data)
    
    print("="*50)
    print(f"✅ Successfully saved to: {output_path}")
    print(f"Total size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("File contains:")
    with np.load(output_path) as npz:
        for k in npz.files:
            item = npz[k]
            if isinstance(item, np.ndarray):
                print(f"- {k}: {item.shape} {item.dtype}")
            else:
                print(f"- {k}: {item}")

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    INPUT_PT_PATH = "/home/nathan/IsaacGym_Preview_4_Package/GVHMR/outputs/demo/messi-cel2/hmr4d_results_messi_cel2.pt"  # Path to your .pt file
    OUTPUT_DIR = "/home/nathan/IsaacGym_Preview_4_Package/PBHC/example/motion_data/video_motion/processed_data/"
    OUTPUT_NAME = "messi_cel2"  # Without extension
    
    convert_pt_to_npz(INPUT_PT_PATH, OUTPUT_DIR, OUTPUT_NAME)