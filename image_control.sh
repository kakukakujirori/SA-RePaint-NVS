ITEM=ride
CAMERA_MOTION_MODE=horizontal
DEGREE=1.0
NUM_FRAMES=25

set -e

# resize to 1024x576
find "data/${ITEM}/images/" -maxdepth 1 -type f -print0 | xargs -0 -I@ convert @ -resize 1024x576! @

# depth estimation
rm -rf "data/${ITEM}/depths"
echo "Running Depth Anything V2..."
python tools/Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/images  \
  --outdir data/${ITEM}/depths

# trajectory extraction
echo "Running Trajectory Extraction..."
python src/trajectory_extraction.py \
  --image_folder data/${ITEM}/images/ \
  --depth_folder data/${ITEM}/depths/ \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --depth_format npy \
  --invert_depth \
  --focal_len 260 \
  --degrees_per_frame ${DEGREE} \
  --camera_motion_mode ${CAMERA_MOTION_MODE} \
  --major_radius 80 \
  --minor_radius 70 \
  --num_frames ${NUM_FRAMES} \
  --control_mode image \
  --no_occlusion_revealing \
  --use_mesh

# generaiton
python src/generate.py \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/generated \
  --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --num_frames ${NUM_FRAMES} \
  --num_inference_steps 50 \
  --denoise_start_step 16 \
  --repaint_iter_num 2 \
  --min_guidance_scale 1.0 \
  --max_guidance_scale 3.0 \
  --seed 12345
