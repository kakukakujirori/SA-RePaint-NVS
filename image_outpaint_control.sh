ITEM=basketball
CAMERA_MOTION_MODE=horizontal
DEGREE=1.0

set -e

# resize to 1024x576
ls data/${ITEM}/images/* | xargs -I@ convert @ -resize 1024x576! @

# outpaint
python src/outpaint.py \
  --image_path data/${ITEM}/images \
  --save_dir data/${ITEM}/outpaint \
  --outpaint_extend_times 0.5

# depth estimation
python tools/Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/outpaint  \
  --outdir data/${ITEM}/outpaint_depth

# # trajectory extraction
# python src/trajectory_extraction.py \
#   --image_folder data/${ITEM}/outpaint/ \
#   --depth_folder data/${ITEM}/outpaint_depth/ \
#   --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
#   --degrees_per_frame ${DEGREE} \
#   --camera_motion_mode ${CAMERA_MOTION_MODE} \
#   --major_radius 80 \
#   --minor_radius 70 \
#   --num_frames 25

# # generaiton
# python src/generate.py \
#   --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
#   --num_frames 25 \
#   --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/generated
