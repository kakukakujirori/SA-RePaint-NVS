import argparse
import concurrent.futures
import os
import shutil
import tempfile
from tqdm import tqdm

from eval_dataset_i2v import run_pixelwise_metrics_calculation, run_fid_kid_calculation, run_fvd_calculation, run_camera_pose_error_calculation, run_sed_calculation

NUM_FRAMES = 25
MAX_WORKER_NUM = 16
GPUS = [0, 1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-to-Video Evaluation")
    parser.add_argument("--data_root", type=str, default="/home/ryotaro/data", help="Root directory for the dataset")
    parser.add_argument("--suffix", type=str, default="nvssolver", help="Suffix for output directories")
    args = parser.parse_args()

    davis_output_root = f"./davis_output_{args.suffix}"
    mannequin_challenge_output_root = f"./mannequin_challenge_output_{args.suffix}"
    tanks_and_temples_output_root = f"./tanks_and_temples_output_{args.suffix}"

    davis_data_root = f"{args.data_root}/DAVIS/JPEGImages/Full-Resolution"
    mannequin_challenge_data_root = f"{args.data_root}/MannequinChallengeHQ/validation_frames"
    tanks_and_temples_data_root = f"{args.data_root}/TanksAndTemples"

    # 1. Gather all data
    def copy_folder(src: str, dst: str):
        shutil.copytree(src, dst)
        for i in range(1, NUM_FRAMES):
            warped_img = os.path.join(dst, "warped", f"{i:04d}.png")
            warped_mask = os.path.join(dst, "warped", f"{i:04d}_mask.png")
            if os.path.isfile(warped_img):
                os.remove(warped_img)
            if os.path.isfile(warped_mask):
                os.remove(warped_mask)

    with tempfile.TemporaryDirectory(dir=args.data_root) as td:  # NOTE: assuming args.data_root is on HDD
        data_root = os.path.join(td, "tmp_data_root")
        os.mkdir(data_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_task_info = {}
            for src_dir in [davis_data_root, mannequin_challenge_data_root, tanks_and_temples_data_root]:
                for scene in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, scene)
                    dst_path = os.path.join(data_root, scene)
                    if os.path.isdir(src_path):
                        future = executor.submit(shutil.copytree, src_path, dst_path)
                        future_to_task_info[future] = (src_path, dst_path)

            for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Copying GT data"):
                src_path, dst_path = future_to_task_info[future]
                task_desc = f"{src_path=}, {dst_path=}"
                try:
                    result_message = future.result()
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc:
                    print(f"Main loop caught exception for {task_desc}: {exc}")
                    raise exc


        output_root = os.path.join(td, "tmp_output_root")
        os.mkdir(output_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_task_info = {}
            for src_dir in [davis_output_root, mannequin_challenge_output_root, tanks_and_temples_output_root]:
                for scene in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, scene)
                    dst_path = os.path.join(output_root, scene)
                    if os.path.isdir(src_path):
                        future = executor.submit(copy_folder, src_path, dst_path)
                        future_to_task_info[future] = (src_path, dst_path)

            for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Copying outputs"):
                src_path, dst_path = future_to_task_info[future]
                task_desc = f"{src_path=}, {dst_path=}"
                try:
                    result_message = future.result()
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc:
                    print(f"Main loop caught exception for {task_desc}: {exc}")
                    raise exc

        try:
            # 4. Pixelwise metrics calculation
            allow_resize = ("trajcrafter" in args.suffix) or ("das" in args.suffix)
            pixelwise_results, _ = run_pixelwise_metrics_calculation(output_root, allow_resize=allow_resize)

            # 5. FID/FVD calculation
            fid_score, kid_score = run_fid_kid_calculation(
                data_root=data_root,
                output_root=output_root,
            )
            fvd_videogpt, fvd_stylegan = run_fvd_calculation(
                data_root=data_root,
                output_root=output_root,
            )

            # 6. Camera pose error calculation
            camera_pose_results, _ = run_camera_pose_error_calculation(output_root)

            # 7. SED calculation
            sed_results = run_sed_calculation(output_root)

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, shutting down.")
        finally:
            print("Program finished.")
