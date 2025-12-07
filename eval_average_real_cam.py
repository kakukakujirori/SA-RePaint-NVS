import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

from eval_dataset_i2v_real_cam import run_pixelwise_metrics_calculation, run_fid_kid_calculation, run_fvd_calculation, run_camera_pose_error_calculation, run_sed_calculation, run_met3r_calculation

NUM_FRAMES = 25
MAX_WORKER_NUM = 1
GPUS = [0, 1]


def load_vbench_scores(json_path: str):
    if not os.path.isfile(json_path):
        raise ValueError(f"{json_path} doesn't exist.")
    with open(json_path, "r") as f:
        results = json.load(f)
    for dimension in results.keys():
        results[dimension] = results[dimension][0]  # take the average score
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-to-Video Evaluation")
    parser.add_argument("--data_root", type=str, default="/mnt/data", help="Root directory for the dataset")
    parser.add_argument("--suffix", type=str, default="nvssolver", help="Suffix for output directories")
    args = parser.parse_args()

    mannequin_output_root = f"./mannequin_challenge_output_given_cam_{args.suffix}"
    dl3dv_output_root = f"./dl3dv_half_output_given_cam_{args.suffix}"

    mannequin_data_root = f"{args.data_root}/MannequinChallengeHQ/validation_frames"
    dl3dv_data_root_original = f"{args.data_root}/DL3DV-Evaluation-img4/images"

    # >>> create tmp data folder for DL3DV >>>
    dl3dv_data_root = "/tmp/dl3dv_half"

    if os.path.isdir(dl3dv_data_root):
        shutil.rmtree(dl3dv_data_root)
    os.makedirs(dl3dv_data_root)

    for idx, scene in enumerate(tqdm(sorted(os.listdir(dl3dv_data_root_original)), desc="Copying DL3DV scenes")):
        scene_path = os.path.join(dl3dv_data_root_original, scene)
        assert os.path.isdir(scene_path), f"Scene path {scene_path} is not a directory."

        if idx % 2 == 1: continue  # reduce the data amount by half

        # copy images
        src_scene_path = os.path.join(dl3dv_data_root_original, scene, scene, "gaussian_splat/images_4")
        dst_scene_path = os.path.join(dl3dv_data_root, scene)
        os.makedirs(dst_scene_path, exist_ok=True)
        for imgpath in glob.glob(os.path.join(src_scene_path, "*.png")):
            Image.open(imgpath).save(os.path.join(dst_scene_path, os.path.basename(imgpath).replace(".png", ".jpg")), "JPEG")
    # <<< create tmp data folder for DL3DV <<<

    # 1. Gather all data
    def copy_folder(src: str, dst: str):
        shutil.copytree(src, dst)
        for i in range(1, NUM_FRAMES):
            warped_img = os.path.join(dst, "warped", f"{i:04d}.png")
            warped_mask = os.path.join(dst, "warped", f"{i:04d}_mask.png")
            if os.path.isfile(warped_img):
                print(f"Warning: {warped_img} shouldn't exist at this directory. Removing it...")
                os.remove(warped_img)
            if os.path.isfile(warped_mask):
                print(f"Warning: {warped_mask} shouldn't exist at this directory. Removing it...")
                os.remove(warped_mask)

    with tempfile.TemporaryDirectory(dir=args.data_root) as td:  # NOTE: assuming args.data_root is on HDD
        print(f"{td=}")
        data_root = os.path.join(td, "tmp_data_root")
        os.mkdir(data_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_task_info = {}
            for src_dir in [mannequin_data_root, dl3dv_data_root]:
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
            for src_dir in [mannequin_output_root, dl3dv_output_root]:
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
            allow_resize = ("trajcrafter" in args.suffix) or ("das" in args.suffix) or ("wan" in args.suffix)
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

            # 8. MET3R calculation
            met3r_results, _ = run_met3r_calculation(output_root, process_size=256, resize_mode="area")

            # 9. VBench
            mannequin_vbench = load_vbench_scores(f"./vbench_results/mannequin_challenge_output_{args.suffix}_eval_results.json")
            dl3dv_vbench = load_vbench_scores(f"./vbench_results/dl3dv_half_output_{args.suffix}_eval_results.json")
            assert set(mannequin_vbench.keys()) == set(dl3dv_vbench.keys())
            vbench_average_scores = {}
            for dimension in mannequin_vbench.keys():
                vbench_average_scores[dimension] = (mannequin_vbench[dimension] + dl3dv_vbench[dimension]) / 2
            print(tabulate(vbench_average_scores.items(), floatfmt=".4f"))


        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, shutting down.")
        finally:
            print("Program finished.")

