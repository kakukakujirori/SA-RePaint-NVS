# Prioritizing Faithfulness: Efficient Zero-Shot Novel View Synthesis with Adaptive Latent Modulation

![elephant](assets/elephant_00025_zoomout_1.0.gif)
![breakdance](assets/breakdance_00000_horizontal_-0.25.gif)

## Installation

1. Create a conda environment
    ```bash
    conda create -n myenv python=3.12
    conda activate myenv
    # conda install conda-forge::imagemagick  # If imagemagick is not installed
    # conda install conda-forge::ffmpeg       # If ffmpeg is not installed
    pip3 install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
    ```

2. Clone the repository
    ```bash
    git clone --recursive git@github.com:kakukakujirori/My-NVS-Solver.git
    cd My-NVS-Solver
    bash patch/apply_patch.sh  # apply patch to submodules
    pip3 install -r requirements.txt
    # pip3 install --no-build-isolation -e ./tools/vipe  # used only for video-to-video evaluation
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

    # # Trajectory Attention (only for evaluation comparison)
    # wget https://huggingface.co/zeqixiao/TrajectoryAttention/resolve/main/trajattn_temp.pth?download=true
    # mv trajattn_temp.pth\?download\=true checkpoints/trajattn_temp.pth

    # # Diffusion As Shader (only for evaluation comparison)
    # git clone https://huggingface.co/EXCAI/Diffusion-As-Shader checkpoints/Diffusion-As-Shader
    ```

## Run

```bash
# image input
bash image_control.sh

# video input
bash video_control.sh
```
You will find the results in the ```output``` folder.

## Evaluation

We assume the following data structure:

```bash
$(data_root)
├── DAVIS
│   ├── Annotations
│   │   └── Full-Resolution
│   ├── ImageSets
│   │   ├── 2016
│   │   └── 2017
│   └── JPEGImages
│       └── Full-Resolution
├── MannequinChallenge
│   ├── test
│   ├── train
│   └── validation
│       ├── 00c9878266685887.txt
│       ├── 0370e2174d04548b.txt
│       └── ......
└── TanksAndTemples
    ├── Auditorium
    ├── Ballroom
    └── ......
```
Then, change the `data_root` under the `if __name__ == '__main__':` block of `eval_dataset_[i/v]2v.py` to your own $(data_root).
You may also want to change the `GPUS` and `MAX_WORKER_NUM` for multiprocessing at the top of these python files.

### DAVIS / Tanks and Temples

```bash
python eval_dataset_i2v.py [davis/tanks] --method mine --use_mesh --scratch
```

- `--method` : which inpainting method to use (`mine`, `trajcrafter`, `trajattn`, `nvssolver`, `das`)
- `--use_mesh` : If set, mesh-based rendering is used. If not, NVS-Solver's bilinear splatting is used.
- `--scratch` : Set when you run the evaluation for the first time. If not set, the code assumes that rendering has already completed and the results are stored in a particular folder.

### Mannequin Challenge

TODO: provide `download_extract.py` (The following works only after you run `download_extract.py` and download all the videos).

If it's the first time, comment out the following two lines in line 676 of `eval_dataset_i2v`:
```python
# if args.dataset == "mannequin":
#     reorganize_frames(data_root)
```
Then run the following:
```bash
python eval_dataset_i2v.py mannequin --method mine --use_mesh --scratch
```

### DAVIS Vdeos

```bash
python eval_dataset_v2v.py davis --method mine --use_mesh --scratch
```
