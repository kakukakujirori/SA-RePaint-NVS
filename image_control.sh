ITEM=basketball
CAMERA_MOTION_MODE=horizontal
DEGREE=1.0
NUM_FRAMES=25

set -e

# resize to 1024x576
ls data/${ITEM}/images/* | xargs -I@ convert @ -resize 1024x576! @

# depth estimation
python tools/Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/images  \
  --outdir data/${ITEM}/depth

# trajectory extraction
python src/trajectory_extraction.py \
  --image_folder data/${ITEM}/images/ \
  --depth_folder data/${ITEM}/depth/ \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --degrees_per_frame ${DEGREE} \
  --camera_motion_mode ${CAMERA_MOTION_MODE} \
  --major_radius 80 \
  --minor_radius 70 \
  --num_frames ${NUM_FRAMES}
  #--no_occlusion_revealing

# generaiton
python src/generate.py \
  --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --num_frames ${NUM_FRAMES} \
  --num_inference_steps 25 \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/generated \
  --seed 12345
