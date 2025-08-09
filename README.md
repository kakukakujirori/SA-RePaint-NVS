# My-NVS-Solver

## Installation

1. Create a conda environment
    ```bash
    conda create -n myenv python=3.12
    conda activate myenv
    pip3 install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
    ```

2. Clone the repository
    ```bash
    git clone --recursive https://github.com/kakukakujirori/My-NVS-Solver.git
    cd My-NVS-Solver
    bash patch/apply_patch.sh  # apply patch to submodules
    pip3 install -r requirements.txt
    ```

3. Download model weights
    ```bash
    mkdir checkpoints

    # Depth Anything V2
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
    mv depth_anything_v2_vitl.pth\?download\=true checkpoints/depth_anything_v2_vitl.pth

    # Video Depth Anything
    wget https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
    mv video_depth_anything_vitl.pth checkpoints/video_depth_anything_vitl.pth

    # Trajectory Attention (only for evaluation comparison)
    wget https://huggingface.co/zeqixiao/TrajectoryAttention/resolve/main/trajattn_temp.pth?download=true
    mv trajattn_temp.pth\?download\=true checkpoints/trajattn_temp.pth
    ```


## Run

```bash
# image input
bash image_control.sh

# video input
bash video_control.sh
```
You will find the results in the ```output``` folder.
