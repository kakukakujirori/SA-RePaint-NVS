import argparse
import glob
import os

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scale and align depth map after outpainting.")
    parser.add_argument("--original_depth_dir", type=str, required=True, help="Path to the original-sized depth maps before outpainting.")
    parser.add_argument("--outpaint_depth_dir", type=str, required=True, help="Path to the large-size depth maps after outpainting.")
    args = parser.parse_args()

    for depth_original in glob.glob(os.path.join(args.original_depth_dir, "*.npy")):
        basename = os.path.basename(depth_original)
        depth_outpaint = os.path.join(args.outpaint_depth_dir, basename.replace(".npy", "_outpaint.npy"))

        assert os.path.isfile(depth_outpaint), f"Outpaint depth file {depth_outpaint} does not exist."

        depth_new = np.load(depth_original)
        depth_old = np.load(depth_outpaint)

        height_new, width_new = depth_new.shape
        height_old, width_old = depth_old.shape
        assert height_new >= height_old and width_new >= width_old, f"{height_new=}, {width_new=}, {height_old=}, {width_old=}"

        sy = (height_new - height_old) // 2
        sx = (width_new - width_old) // 2

        depth_new_cropped = depth_new[sy:sy+height_old, sx:sx+width_old]
        depth_new_median = np.median(depth_new_cropped)
        depth_old_median = np.median(depth_old)
        depth_new_rescaled = depth_new * depth_old_median / depth_new_median
        print(f"depth_new_median: {depth_new_median}, depth_old_median: {depth_old_median}")

        np.save(depth_outpaint, depth_new_rescaled)