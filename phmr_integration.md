# PromptHMR Integration with GMR

This guide documents the integration of **PromptHMR** results into the **GMR (General Motion Retargeting)** system, specifically for retargeting high-quality human motion captured from video to the **Unitree G1** robot.

## 1. Prerequisites

- **GMR** installed in `~/Documents/unitree/GMR`
- **PromptHMR** installed in `~/Documents/unitree/PromptHMR`
- PromptHMR results generated (e.g., in `~/Documents/unitree/PromptHMR/results/boxing/results.pkl`)

## 2. Environment Setup

### Install Dependencies
PromptHMR results use `joblib` for serialization. Install it in your GMR environment:

```bash
cd ~/Documents/unitree/GMR
uv pip install joblib
```

### Link SMPL-X Body Models
GMR requires SMPL-X models to process human poses. To save disk space, we link the models already present in PromptHMR:

```bash
mkdir -p assets/body_models/smplx
ln -sf ~/Documents/unitree/PromptHMR/data/body_models/smplx/SMPLX_NEUTRAL.pkl assets/body_models/smplx/
ln -sf ~/Documents/unitree/PromptHMR/data/body_models/smplx/SMPLX_FEMALE.pkl assets/body_models/smplx/
ln -sf ~/Documents/unitree/PromptHMR/data/body_models/smplx/SMPLX_MALE.pkl assets/body_models/smplx/
```

---

## 3. Retargeting Script

We developed a bridge script `scripts/prompthmr_to_robot.py` that translates PromptHMR's complex world-frame results (55 joints, rotation matrices) into the GMR format.

### Features:
- Handles **world-frame** and **camera-frame** results from PromptHMR.
- Automatically converts **(N, 165)** rotation vector formats to global joint positions.
- Configures SMPL-X batch size dynamically for any video length.
- Supports all GMR-compatible robots (Unitree G1, H1, etc.).

---

## 4. How to Reproduce

To retarget PromptHMR boxing results to the Unitree G1:

```bash
# Activate your gmr environment
source .venv/bin/activate  # or conda activate gmr

# Run the retargeting
python scripts/prompthmr_to_robot.py \
    --results_file ~/Documents/unitree/PromptHMR/results/boxing/results.pkl \
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

## 5. Technical Implementation Details

During the integration, the following key steps were taken:

1.  **Format Discovery**: PromptHMR results were identified as `joblib`-compressed dictionaries containing `smplx_world` or `smplx_pose` parameters.
2.  **Coordinate Transformation**: PromptHMR often outputs rotation matrices `(N, 55, 3, 3)` or flat rotation vectors `(N, 165)`. The script robustly handles both and converts them into global coordinate snapshots for GMR's IK solver.
3.  **SMPL-X Initialization**: The SMPL-X model loader was modified to pass the `ext="pkl"` flag and the correct `batch_size`, matching the PromptHMR model structure.
4.  **Height Scaling**: The script extracts SMPL-X `betas` from the result file to calculate the correct human height for accurate retargeting scaling.
