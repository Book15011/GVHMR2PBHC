import numpy as np
import argparse
import os
import joblib
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Optional

# Based on analysis of the project files, this is the assumed joint order for the 23 DOFs.
DOF_NAMES = [
    "Left Hip Yaw", "Left Hip Roll", "Left Hip Pitch", "Left Knee Pitch", "Left Ankle Pitch", "Left Ankle Roll",  # 0-5
    "Right Hip Yaw", "Right Hip Roll", "Right Hip Pitch", "Right Knee Pitch", "Right Ankle Pitch", "Right Ankle Roll", # 6-11
    "Waist Pitch", "Waist Roll", "Waist Yaw", # 12-14
    "Left Shoulder Pitch", "Left Shoulder Roll", "Left Shoulder Yaw", "Left Elbow Pitch", # 15-18
    "Right Shoulder Pitch", "Right Shoulder Roll", "Right Shoulder Yaw", "Right Elbow Pitch" # 19-22
]

def recalculate_pose_aa(dof, root_rot):
    """
    Recalculates the pose_aa from modified dof and root_rot to keep visualization markers in sync.
    """
    # Construct a path to dof_axis.npy relative to this script's location
    dof_axis_file = Path(__file__).parent.parent / "description/robots/g1/dof_axis.npy"
    if not dof_axis_file.exists():
        print(f"Error: dof_axis.npy not found at {dof_axis_file.resolve()}")
        print("Cannot recalculate 'pose_aa'.")
        return None

    dof_axis = np.load(dof_axis_file, allow_pickle=True).astype(np.float32)

    num_dof = dof.shape[1]
    if dof_axis.shape[0] != num_dof:
        print(f"Error: Mismatch between dof_axis shape ({dof_axis.shape[0]}) and dof shape ({num_dof}).")
        print("Cannot recalculate 'pose_aa'.")
        return None

    # Convert root rotation quaternion to axis-angle
    root_aa = R.from_quat(root_rot).as_rotvec()

    # Calculate joint axis-angle representation (angle * axis)
    joint_aa = dof_axis * np.expand_dims(dof, axis=2)

    # Concatenate root and joint axis-angles
    pose_aa_main = np.concatenate(
        (np.expand_dims(root_aa, axis=1), joint_aa),
        axis=1
    )

    # The original format has 27 joints in pose_aa: 1 (root) + 23 (dof) + 3 (placeholders).
    num_frames = dof.shape[0]
    num_placeholder_joints = 27 - pose_aa_main.shape[1]
    if num_placeholder_joints > 0:
        placeholder_zeros = np.zeros((num_frames, num_placeholder_joints, 3), dtype=np.float32)
        pose_aa = np.concatenate((pose_aa_main, placeholder_zeros), axis=1)
    else:
        pose_aa = pose_aa_main

    return pose_aa.astype(np.float32)

def modify_motion_data(
    motion_file: Path, 
    output_folder: Optional[Path] = None, 
    part_to_fix: str = 'lower', 
    dof_split_index: int = 12
):
    """
    Modifies motion data from a .pkl file and saves it to a new file.

    Args:
        motion_file (Path): Path to the input motion file.
        output_folder (Optional[Path]): Path to the folder to save the modified file. 
                                        If None, saves to a 'modified' subdir next to the input file.
        part_to_fix (str): Part of the body/motion to freeze ('lower', 'upper', 'root', 'all', or 'none').
        dof_split_index (int): The DOF index that separates lower and upper body.
    """
    # Load the motion data
    if not motion_file.exists():
        print(f"Error: Motion file not found at {motion_file.resolve()}")
        return
        
    out_motion_data = joblib.load(motion_file)
    original_key = next(iter(out_motion_data.keys()))
    motion_data = out_motion_data[original_key]

    # Make copies of data to be modified
    dof = motion_data['dof'].copy()
    root_trans = motion_data.get('root_trans_offset', np.zeros((dof.shape[0], 3))).copy()

    # Correctly handle default root rotation
    if 'root_rot' in motion_data and motion_data['root_rot'] is not None:
        root_rot = motion_data['root_rot'].copy()
    else:
        # If root_rot is missing or None, create an identity quaternion for all frames
        print("Warning: 'root_rot' not found or is None in motion data. Defaulting to identity rotation.")
        root_rot = np.zeros((dof.shape[0], 4))
        root_rot[:, 3] = 1.0 # w=1 for identity quaternion (x,y,z,w)

    print("\n--- Data before modification ---")
    print(f"DOF Shape: {dof.shape}")
    print(f"First 5 frames of DOFs:\n{dof[:5, :12]}")
    print(f"First 5 frames of root translation:\n{root_trans[:5]}")
    print(f"First 5 frames of root rotation:\n{root_rot[:5]}")

    # --- Store first frame for freezing ---
    first_frame_dof = dof[0, :].copy()
    first_frame_root_trans = root_trans[0, :].copy()
    first_frame_root_rot = root_rot[0, :].copy()

    # --- Apply modifications ---
    if part_to_fix == 'lower':
        print(f"\nFreezing the first {dof_split_index} DOFs (lower body) to initial pose.")
        print("Affected joints:")
        for i in range(dof_split_index):
            print(f"  DOF {i}: {DOF_NAMES[i]}")
        dof[:, :dof_split_index] = first_frame_dof[:dof_split_index]
        # Also freeze root translation, but preserve rotation for natural waist movement
        print("Freezing root translation only (rotation is preserved).")
        root_trans[:] = first_frame_root_trans
    elif part_to_fix == 'upper':
        print(f"\nFreezing DOFs from index {dof_split_index} onwards (upper body) to initial pose. Root motion is preserved.")
        print("Affected joints:")
        for i in range(dof_split_index, len(DOF_NAMES)):
            print(f"  DOF {i}: {DOF_NAMES[i]}")
        dof[:, dof_split_index:] = first_frame_dof[dof_split_index:]
    elif part_to_fix == 'root':
        print(f"\nFreezing root translation and rotation to initial frame.")
        root_trans[:] = first_frame_root_trans
        root_rot[:] = first_frame_root_rot
    elif part_to_fix == 'all':
        print(f"\nFreezing ALL DOFs to initial pose AND root motion.")
        dof[:, :] = first_frame_dof
        root_trans[:] = first_frame_root_trans
        root_rot[:] = first_frame_root_rot
    elif part_to_fix == 'legs_and_root':
        print(f"\nFreezing legs (DOFs 0-11) and root motion completely. Waist and arms remain free.")
        print("Affected joints (Legs):")
        for i in range(12):
            print(f"  DOF {i}: {DOF_NAMES[i]}")
        dof[:, :12] = first_frame_dof[:12]

        print("Freezing root translation and rotation.")
        root_trans[:] = first_frame_root_trans
        root_rot[:] = first_frame_root_rot
    elif part_to_fix == 'legs_static_waist_dynamic':
        print(f"\nFreezing legs (DOFs 0-11) and root. Transferring root yaw to waist for dynamic motion.")
        
        # 1. Freeze legs and root translation
        dof[:, :12] = first_frame_dof[:12]
        root_trans[:] = first_frame_root_trans

        # 2. Extract FORWARD VECTOR from root rotation
        rotations = R.from_quat(root_rot)
        forward_vecs = rotations.apply(np.array([1, 0, 0]))  # Body's forward direction
        
        # 3. Project to horizontal plane and compute world yaw
        horizontal_forward = forward_vecs.copy()
        horizontal_forward[:, 2] = 0  # Remove vertical component
        norm = np.linalg.norm(horizontal_forward, axis=1, keepdims=True)
        horizontal_forward /= (norm + 1e-8) # Add epsilon for stability
        world_yaws = np.arctan2(horizontal_forward[:, 1], horizontal_forward[:, 0])

        # 4. Unwrap angles and compute delta yaw
        unwrapped_yaws = np.unwrap(world_yaws)
        delta_yaw = unwrapped_yaws - unwrapped_yaws[0]

        # 5. Add the transferred yaw to the waist's yaw DOF (12) and freeze the root rotation.
        dof[:, 12] += delta_yaw
        root_rot[:] = first_frame_root_rot

    elif part_to_fix == 'waist_only':
        print(f"\nFreezing everything EXCEPT the waist (DOFs 12-14) to initial pose. Root motion is preserved.")
        
        # Freeze lower body (0-11)
        print("Affected joints (Legs):")
        for i in range(12):
            print(f"  DOF {i}: {DOF_NAMES[i]}")
        dof[:, :12] = first_frame_dof[:12]
        
        # Freeze upper body arms (15-22)
        print("Affected joints (Arms):")
        for i in range(15, 23):
            print(f"  DOF {i}: {DOF_NAMES[i]}")
        dof[:, 15:] = first_frame_dof[15:]
    elif part_to_fix == 'none':
        print("\nNo parts frozen. Passing motion through unchanged.")
    else:
        print(f"Warning: Unknown part '{part_to_fix}'. No modifications made.")

    # --- Update motion_data dictionary with modified arrays ---
    # This needs to be done for all cases where modifications happen
    if part_to_fix != 'none':
        motion_data['dof'] = dof
        motion_data['root_trans_offset'] = root_trans
        motion_data['root_rot'] = root_rot

        # Recalculate pose_aa if it exists to keep visualization markers in sync
        if 'pose_aa' in motion_data:
            print("\nRecalculating 'pose_aa' from modified DOFs and root rotation...")
            new_pose_aa = recalculate_pose_aa(dof, root_rot)
            if new_pose_aa is not None:
                if new_pose_aa.shape == motion_data['pose_aa'].shape:
                    motion_data['pose_aa'] = new_pose_aa
                    print("Successfully updated 'pose_aa'.")
                else:
                    print(f"Warning: Shape mismatch for 'pose_aa'. Original: {motion_data['pose_aa'].shape}, New: {new_pose_aa.shape}. Not updating.")
            else:
                print("Warning: Failed to update 'pose_aa'. Visualization markers may be incorrect.")

    print("\n--- Data after modification ---")
    print(f"First 5 frames of DOFs:\n{motion_data['dof'][:5, :12]}")
    print(f"First 5 frames of root translation:\n{motion_data['root_trans_offset'][:5]}")
    print(f"First 5 frames of root rotation:\n{motion_data['root_rot'][:5]}")



    # --- Handle output path and filename ---
    if output_folder is None:
        output_folder = motion_file.parent / "modified"
    
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create a descriptive output filename
    if part_to_fix != 'none':
        suffix = f"_fix-{part_to_fix}"
        if part_to_fix in ['lower', 'upper']:
            suffix += f"_split-{dof_split_index}"
        output_filename = f"{motion_file.stem}{suffix}.pkl"
    else:
        output_filename = f"{motion_file.stem}_unmodified.pkl"
        
    output_file = output_folder / output_filename
    
    # The key for the new dictionary should be the new filename's stem.
    output_key = output_file.stem

    # Save the modified motion data back to the new .pkl file
    joblib.dump({output_key: motion_data}, output_file)

    print(f"\n✅ Modified motion data saved to: {output_file.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="Modify motion data (.pkl) by freezing parts of the body or root motion.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('motion_file', type=str, help='Path to the motion file to modify.')
    parser.add_argument('--output-folder', '-o', type=str, default=None, 
                        help='Folder to save the modified file.\n(default: a "modified" sub-directory next to the input file)')
    parser.add_argument('--fix-part', type=str, default='lower', choices=['lower', 'upper', 'root', 'all', 'legs_and_root', 'legs_static_waist_dynamic', 'waist_only', 'none'], 
                        help="Part of the body/motion to freeze.\n"
                             "'lower': freezes lower body DOFs AND root translation (rotation is preserved).\n"
                             "'upper': freezes upper body DOFs but PRESERVES root motion.\n"
                             "'root': freezes ONLY root translation and rotation.\n"
                             "'all': freezes all DOFs and root motion.\n"
                             "'legs_and_root': freezes leg DOFs (0-11) AND root translation/rotation.\n"
                             "'legs_static_waist_dynamic': freezes legs/root, but transfers root yaw to the waist.\n"
                             "'waist_only': freezes everything except waist DOFs (12-14), PRESERVES root motion.\n"
                             "'none': no modification, just re-saves the file")
    parser.add_argument('--split-index', type=int, default=12, 
                        help="The DOF index to split lower and upper body.\n(default: 12, which is after both legs)")
    args = parser.parse_args()

    motion_file_path = Path(args.motion_file)
    output_folder_path = Path(args.output_folder) if args.output_folder else None

    modify_motion_data(
        motion_file_path,
        output_folder_path,
        part_to_fix=args.fix_part,
        dof_split_index=args.split_index
    )

if __name__ == "__main__":
    main()
