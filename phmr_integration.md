This guide documents the integration of **PromptHMR** results into the **GMR (General Motion Retargeting)** system, specifically for retargeting high-quality human motion captured from video to the **Unitree G1** robot.

## 1. Workflow Overview

This integration requires **two separate projects** and **two separate virtual environments**. Do not attempt to merge them into a single environment, as they have conflicting dependency requirements (especially PyTorch versions and CUDA extensions).

1.  **PromptHMR (`phmr-env`)**: Used to extract human motion from video.
2.  **GMR (`gmr-env`)**: Used to retarget those results to the robot.

**Installation Order**:
- First, set up **PromptHMR** following [Step 3](#3-prerequisites) and [readme_uv.md](../PromptHMR/readme_uv.md).
- Second, set up **GMR** following [Step 3](#3-prerequisites) (GMR Installation section).

## 2. Path Configuration

Define the paths to your local clones of GMR and PromptHMR:

```bash
# Set these variables to your actual installation paths
export PATH_TO_GMR=~/Documents/unitree/GMR
export PATH_TO_PROMPTHMR=~/Documents/unitree/PromptHMR
```

## 3. Prerequisites

### PromptHMR
- **Accessible**: The repository must be cloned and located at the path defined by `$PATH_TO_PROMPTHMR`.
- **Installed**: Environment and dependencies must be set up following [PromptHMR/readme_uv.md](../PromptHMR/readme_uv.md). This includes:
    - `uv` environment with Python 3.12.
    - All requirements and pre-compiled wheels installed.
    - **Body Models** (SMPL/SMPL-X) downloaded via `fetch_smplx.sh`.
- **Results**: PromptHMR results must be generated (e.g., in `$PATH_TO_PROMPTHMR/results/boxing/results.pkl`).

### GMR
- **Accessible**: The repository must be cloned and located at the path defined by `$PATH_TO_GMR`.
- **Installed**: Follow these steps to prepare your GMR environment:
    ```bash
    cd $PATH_TO_GMR
    # 1. Create and activate environment
    uv venv --python 3.12

    # 2. Install GMR in editable mode
    uv pip install -e .

    # 3. Install integration dependencies
    uv pip install joblib

    # 4. Link SMPL-X Body Models
    # GMR requires SMPL-X models to process human poses. To save disk space,
    # we link the models already present in PromptHMR:
    mkdir -p assets/body_models/smplx
    ln -sf $PATH_TO_PROMPTHMR/data/body_models/smplx/SMPLX_NEUTRAL.pkl assets/body_models/smplx/
    ln -sf $PATH_TO_PROMPTHMR/data/body_models/smplx/SMPLX_FEMALE.pkl assets/body_models/smplx/
    ln -sf $PATH_TO_PROMPTHMR/data/body_models/smplx/SMPLX_MALE.pkl assets/body_models/smplx/
    ```

---

## 4. Retargeting Script

We developed a bridge script `scripts/prompthmr_to_robot.py` (inside GMR) that translates PromptHMR's complex world-frame results (55 joints, rotation matrices) into the GMR format.

### Features:
- Handles **world-frame** and **camera-frame** results from PromptHMR.
- Automatically converts **(N, 165)** rotation vector formats to global joint positions.
- Configures SMPL-X batch size dynamically for any video length.
- Supports all GMR-compatible robots (Unitree G1, H1, etc.).

---

## 5. How to Reproduce

To retarget PromptHMR results to the Unitree G1:

Use the same `<your_video_name>` as the one you used in the last step of PromptHMR/readme_uv.md.

```bash
# 1. Activate your GMR environment (from GMR directory)
cd $PATH_TO_GMR

export VIDEO_NAME=<your_video_name>

# 2. Run the retargeting script pointing to PromptHMR results
uv run python scripts/prompthmr_to_robot.py \
    --results_file $PATH_TO_PROMPTHMR/results/$VIDEO_NAME/results.pkl \
    --robot unitree_g1 \
    --person_idx 0 \
    --rate_limit
```

### Script Arguments:
- `--results_file`: Path to the PromptHMR `.pkl` file.
- `--robot`: Robot model (default: `unitree_g1`).
- `--person_idx`: Which person to retarget (e.g., `0` or `1` for multi-person videos).
- `--rate_limit`: Sync playback speed with real-time.
- `--record_video`: Save the visualization to `.mp4`.
- `--save_path`: Save the retargeted robot joint angles to a `.pkl` file.

---

## 6. Technical Implementation Details

During the integration, the following key steps were taken:

1.  **Format Discovery**: PromptHMR results were identified as `joblib`-compressed dictionaries containing `smplx_world` or `smplx_pose` parameters.
2.  **Coordinate Transformation**: PromptHMR often outputs rotation matrices `(N, 55, 3, 3)` or flat rotation vectors `(N, 165)`. The script robustly handles both and converts them into global coordinate snapshots for GMR's IK solver.
3.  **SMPL-X Initialization**: The SMPL-X model loader was modified to pass the `ext="pkl"` flag and the correct `batch_size`, matching the PromptHMR model structure.
4.  **Height Scaling**: The script extracts SMPL-X `betas` from the result file to calculate the correct human height for accurate retargeting scaling.
