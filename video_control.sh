ITEM=bear
CAMERA_MOTION_MODE=horizontal
DEGREE=0.5
NUM_FRAMES=25
STRIDE=1

set -e

# check the existence and the length of the video
INPUT_VIDEO=data/${ITEM}/${ITEM}.mp4
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: input video not found: $INPUT_VIDEO"
    exit 1
fi

TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:nw=1 "$INPUT_VIDEO")
REQUIRED_FRAMES=$(( (NUM_FRAMES - 1) * STRIDE ))
if [ "$TOTAL_FRAMES" -lt "$REQUIRED_FRAMES" ]; then
    echo "Error: Not enough frames in the video to perform the operation."
    echo "Required frames are at least up to index $REQUIRED_FRAMES, but total frames are $TOTAL_FRAMES."
    exit 1
fi

# extract NUM_FRAMES images -> resize to 1024x576
mkdir -p data/${ITEM}/images
ffmpeg -i $INPUT_VIDEO -vf "select='not(mod(n,$STRIDE))'" -vframes "$NUM_FRAMES" -vsync vfr "data/${ITEM}/images/%05d.png"
ls data/${ITEM}/images/* | xargs -I@ convert @ -resize 1024x576! @


# depth estimation
ffmpeg -y -framerate 10 -i "data/${ITEM}/images/%05d.png" -c:v libx264 -r 10 -pix_fmt rgb24 -crf 0 "data/${ITEM}/${ITEM}_tmp.mp4"

python tools/Video-Depth-Anything/run.py \
  --encoder vitl \
  --input_video data/${ITEM}/${ITEM}_tmp.mp4  \
  --output_dir data/${ITEM}/depth \
  --save_npz

rm "data/${ITEM}/${ITEM}_tmp.mp4"

# trajectory extraction
python src/trajectory_extraction.py \
  --image_folder data/${ITEM}/images/ \
  --depth_folder data/${ITEM}/depth/ \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/warped \
  --degrees_per_frame ${DEGREE} \
  --camera_motion_mode ${CAMERA_MOTION_MODE} \
  --major_radius 80 \
  --minor_radius 70 \
  --num_frames ${NUM_FRAMES} \
  --control_mode video \
  --no_occlusion_revealing

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
