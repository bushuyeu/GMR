import argparse
import pathlib
import os
import time
import joblib
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import get_smplx_data

def matrix_to_rotvec(matrix):
    """Convert rotation matrix to rotation vector."""
    return R.from_matrix(matrix).as_rotvec()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to PromptHMR results.pkl")
    parser.add_argument("--person_idx", type=int, default=0, help="Index of the person to retarget")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    args = parser.parse_args()

    # Asset paths
    HERE = pathlib.Path(__file__).parent.absolute()
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    print(f"Loading results from {args.results_file}...")
    data = joblib.load(args.results_file)
    
    people_ids = list(data['people'].keys())
    if args.person_idx >= len(people_ids):
        print(f"[red]Error: person_idx {args.person_idx} out of range (max {len(people_ids)-1})[/red]")
        return
    
    person_key = people_ids[args.person_idx]
    person = data['people'][person_key]
    
    print(f"Processing person: {person_key}")
    
    # Extract SMPL-X parameters
    # PromptHMR usually outputs world or camera frame params.
    if 'smplx_world' in person:
        pose = person['smplx_world']['pose'] # (N, 165) or (N, 55, 3, 3)
        transl = person['smplx_world']['trans'] # (N, 3)
        betas = person['smplx_world']['shape'] # (N, 10 or 16)
        is_rotvec = (pose.shape[-1] == 165 or (len(pose.shape) == 3 and pose.shape[1] == 55 and pose.shape[2] == 3))
    else:
        pose = person['smplx_pose'] # (N, 22, 3, 3)
        transl = person['smplx_transl'] # (N, 3)
        betas = person['smplx_betas'] # (N, 10)
        is_rotvec = False

    num_frames = pose.shape[0]
    
    # Convert rotation matrices to rotvecs if necessary
    print(f"Preparing pose data (is_rotvec={is_rotvec})...")
    if is_rotvec:
        if pose.shape[-1] == 165:
            root_orient = pose[:, :3]
            body_pose = pose[:, 3:66]
            # Optional: hands, jaw, eyes
            jaw_pose = pose[:, 66:69]
            leye_pose = pose[:, 69:72]
            reye_pose = pose[:, 72:75]
            left_hand_pose = pose[:, 75:120]
            right_hand_pose = pose[:, 120:165]
        else:
            # (N, 55, 3) format
            root_orient = pose[:, 0]
            body_pose = pose[:, 1:22].reshape(num_frames, -1)
            # ...
    else:
        # matrix format (N, 22, 3, 3)
        root_orient = []
        body_pose = []
        for f in range(num_frames):
            root_orient.append(matrix_to_rotvec(pose[f, 0]))
            body_pose.append(matrix_to_rotvec(pose[f, 1:]).flatten())
        root_orient = np.array(root_orient)
        body_pose = np.array(body_pose)
    
    # Pad betas to 16 if necessary
    if betas.shape[1] < 16:
        betas = np.pad(betas, ((0, 0), (0, 16 - betas.shape[1])))
    
    # Initialize SMPL-X model
    print(f"Initializing SMPL-X model for {num_frames} frames...")
    body_model = smplx.create(
        str(SMPLX_FOLDER), # Use assets/body_models which contains 'smplx' folder
        "smplx",
        gender="neutral",
        use_pca=False,
        ext="pkl",
        batch_size=num_frames,
    )
    
    # Forward pass to get joints
    print("Computing global joints...")
    # Use CPU for now as cuda might have issues in background or with small batch
    # body_model.to(device)
    
    # We need to pass data in batches if it's too large, but 395 frames is small.
    # Note: smplx library wants tensors.
    with torch.no_grad():
        smplx_output = body_model(
            betas=torch.tensor(betas).float(),
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(body_pose).float(),
            transl=torch.tensor(transl).float(),
            return_full_pose=True
        )
    
    # Move results back to CPU
    smplx_output.joints = smplx_output.joints.cpu()
    smplx_output.global_orient = smplx_output.global_orient.cpu()
    smplx_output.full_pose = smplx_output.full_pose.cpu()
    
    # Compute human height
    human_height = 1.66 + 0.1 * betas[0, 0]
    
    # Initialize GMR
    print(f"Initializing GMR for {args.robot}...")
    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    fps = 30 # Default for PromptHMR/GMR
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=fps,
        record_video=args.record_video,
        video_path=args.save_path.replace('.pkl', '.mp4') if args.save_path and args.record_video else "videos/prompthmr_demo.mp4"
    )
    
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        qpos_list = []

    print("Starting retargeting loop...")
    i = 0
    while True:
        if args.loop:
            i = (i + 1) % num_frames
        else:
            if i >= num_frames:
                break
        
        # Get frame data in dict format for GMR
        frame_data = get_smplx_data(None, body_model, smplx_output, i)
        
        # Retarget
        qpos = retarget.retarget(frame_data)
        
        # Visualize
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            rate_limit=args.rate_limit,
        )
        
        if args.save_path:
            qpos_list.append(qpos)
            
        i += 1

    if args.save_path:
        import pickle
        motion_data = {
            "fps": fps,
            "root_pos": np.array([q[:3] for q in qpos_list]),
            "root_rot": np.array([q[3:7][[1,2,3,0]] for q in qpos_list]), # wxyz to xyzw
            "dof_pos": np.array([q[7:] for q in qpos_list]),
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    viewer.close()

if __name__ == "__main__":
    main()
