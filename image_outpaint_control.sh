ITEM=basketball
CAMERA_MOTION_MODE=horizontal
DEGREE=1.0

set -e

# resize to 1024x576
ls data/${ITEM}/images/* | xargs -I@ convert @ -resize 1024x576! @

# outpaint
python src/outpaint.py \
  --image_path data/${ITEM}/images \
  --save_dir data/${ITEM}/images_outpaint \
  --outpaint_extend_times 0.5

# depth estimation
python tools/Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/images  \
  --outdir data/${ITEM}/depth

python tools/Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/images_outpaint  \
  --outdir data/${ITEM}/depth_outpaint

# depth scale alignment
python src/outpaint_depth_scale_align.py \
  --original_depth_dir data/${ITEM}/depth \
  --outpaint_depth_dir data/${ITEM}/depth_outpaint \

# trajectory extraction
python src/trajectory_extraction.py \
  --image_folder data/${ITEM}/images_outpaint/ \
  --depth_folder data/${ITEM}/depth_outpaint/ \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --degrees_per_frame ${DEGREE} \
  --camera_motion_mode ${CAMERA_MOTION_MODE} \
  --major_radius 80 \
  --minor_radius 70 \
  --num_frames 25 \
  --no_occlusion_revealing

# generaiton
python src/generate.py \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/generated \
  --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --num_frames ${NUM_FRAMES} \
  --num_inference_steps 25 \
  --enable_resample \
  --denoise_start_step 8 \
  --seed 12345