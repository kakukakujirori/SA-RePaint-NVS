import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import subprocess
import tempfile
from itertools import product
from tqdm import tqdm

import imagesize
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from cleanfid import fid
from diffusers.utils import load_image
from fvdcal import FVDCalculation
from pathos.multiprocessing import ProcessingPool

from src.eval_sed import eval_sed

NUM_FRAMES = 25
GPUS = [0, 1, 2, 3]


def run_fid_calculation(
        data_root: str,
        output_root: str,
    ):
    with tempfile.TemporaryDirectory() as td:
        print(f"[run_fid_calculation] Images are copied to {td} for FID calculation.")
        gt_folder = os.path.join(td, "gt")
        generated_folder = os.path.join(td, "generated")
        os.mkdir(gt_folder)
        os.mkdir(generated_folder)

        # Copy GT images
        for scene in os.listdir(data_root):
            for imgpath in glob.glob(os.path.join(data_root, scene, "*.jpg")):
                imgpath = os.path.abspath(imgpath)
                shutil.copy(imgpath, os.path.join(gt_folder, imgpath.replace("/", "_")))

        # Copy generated images
        for scene in os.listdir(output_root):
            scene_path = os.path.join(output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                for imgpath in glob.glob(os.path.join(output_root, scene, motion_degree, "generated", "*.png")):
                    imgpath = os.path.abspath(imgpath)
                    dst_path = os.path.join(generated_folder, imgpath.replace("/", "_"))
                    shutil.copy(imgpath, dst_path)

        # calculate FID
        score = fid.compute_fid(gt_folder, generated_folder)
        print(f"FID: {score}")
        return score.item()


def run_fvd_calculation(
        davis_data_root: str,
        davis_output_root: str,
    ):

    # Taken from https://github.com/ZGCTroy/CamI2V/blob/main/evaluation/fvd_test.py
    class MyFVDCalculation(FVDCalculation):
        def calculate_fvd_by_video_list(
            self,
            real_videos: torch.Tensor,
            generated_videos: torch.Tensor,
            model_path: str = "tools/FVD/model",
            device: str = "cuda:0",
        ):
            model = self._load_model(model_path, device)
            fvd = self._compute_fvd_between_video(model, real_videos, generated_videos, device)
            return fvd.detach().cpu().numpy()

    # load models
    fvd_videogpt = MyFVDCalculation(method="videogpt")
    fvd_stylegan = MyFVDCalculation(method="stylegan")

    # load videos
    start_frames_per_scene = {}
    for scene_imgname in os.listdir(davis_output_root):
        if not os.path.isdir(os.path.join(davis_output_root, scene_imgname)):
            continue
        scene, imgname = scene_imgname.split("_")
        if scene not in start_frames_per_scene:
            start_frames_per_scene[scene] = []
        start_frames_per_scene[scene].append(imgname)

    gt_video_list = []
    sample_video_list = []

    def run_task(scene: str, start_frames: list[str]):
        gt_frames = sorted(glob.glob(os.path.join(davis_data_root, scene, "*.jpg")))

        for start in sorted(start_frames):
            scene_imgname = f"{scene}_{start}"

            # find the start frame
            indices = [i for i, s in enumerate(gt_frames) if os.path.basename(s).startswith(start)]
            assert len(indices) == 1, f"Found multiple indices for {scene=}, {start=}, {indices=}"
            idx = indices[0]

            # load gt video
            gt_video = gt_frames[idx:idx+NUM_FRAMES]
            if len(gt_video) < NUM_FRAMES:
                gt_video += [gt_video[-1]] * (NUM_FRAMES - len(gt_video))
            gt_video = [load_image(p) for p in gt_video]

            # load sample video
            for motion in os.listdir(os.path.join(davis_output_root, scene_imgname)):
                sample_video = [os.path.join(davis_output_root, scene_imgname, motion, f"generated/{i:04d}.png") for i in range(25)]
                sample_video = [load_image(p) for p in sample_video]

            # NOTE: FDV inputs are uint8
            gt_video = [torch.from_numpy(np.array(x)) for x in gt_video]
            sample_video = [torch.from_numpy(np.array(x)) for x in sample_video]

            gt_video = torch.stack(gt_video, dim=0).permute(0, 3, 1, 2)
            sample_video = torch.stack(sample_video, dim=0).permute(0, 3, 1, 2)

            if max(gt_video.shape[-2:]) >= 1024:
                gt_video = F.interpolate(gt_video, scale_factor=0.5, mode='bilinear', align_corners=False)
            if max(sample_video.shape[-2:]) >= 1024:
                sample_video = F.interpolate(sample_video, scale_factor=0.5, mode='bilinear', align_corners=False)

            gt_video_list.append(gt_video)  # (T, C, H, W)
            sample_video_list.append(sample_video)  # (T, C, H, W)


    with concurrent.futures.ThreadPoolExecutor(max_workers=31) as executor:
        future_to_task_info = {}
        for scene, start_frames in start_frames_per_scene.items():
            future = executor.submit(run_task,  scene, start_frames)
            future_to_task_info[future] = (scene, start_frames)

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Loading videos for FVD calculation"):
            scene, start_frames = future_to_task_info[future]
            task_desc = f"{scene=}, {start_frames=}"
            try:
                result_message = future.result()
                # print(f"Result for {task_desc}: {result_message}")
            except Exception as exc:
                print(f"Main loop caught exception for {task_desc}: {exc}")
                raise exc

    # align size to samples and batchfy
    assert len(gt_video_list) == len(sample_video_list), f"Length mismatch: {len(gt_video_list)} vs {len(sample_video_list)}"
    gt_video_list = [F.interpolate(gt_vid, size=sample_vid.shape[-2:], mode='bilinear', align_corners=False) for gt_vid, sample_vid in zip(gt_video_list, sample_video_list)]
    gt_video_batch = torch.stack(gt_video_list, dim=0)  # (N, T, C, H, W)
    sample_video_batch = torch.stack(sample_video_list, dim=0)  # (N, T, C, H, W)

    # calculate
    score_videogpt = fvd_videogpt.calculate_fvd_by_video_list(gt_video_batch, sample_video_batch).item()
    score_stylegan = fvd_stylegan.calculate_fvd_by_video_list(gt_video_batch, sample_video_batch).item()
    print(f"FVD: {score_videogpt=}, {score_stylegan=}")

    return score_videogpt, score_stylegan


def run_sed_calculation(output_root: str):

    def run_task(args: tuple[str, float, int]):
        data_dir, gt_focal_len, gpu_id = args

        video_path = os.path.join(data_dir, "generated/generated.mp4")
        if not os.path.isfile(video_path):
            return (data_dir, None)

        # Load the gt image size and camera poses
        gt_width, gt_height = imagesize.get(os.path.join(data_dir, f"warped/0000.png"))
        camera_paths = os.path.join(data_dir, "warped/*_pose.npy")
        poses = [np.load(p) for p in sorted(glob.glob(camera_paths))]

        with tempfile.TemporaryDirectory() as colmap_root:
            consistent_ratios, sed_summary = eval_sed(
                colmap_root=colmap_root,
                video_path=video_path,
                poses=poses,
                gt_width=gt_width,
                gt_height=gt_height,
                gt_focal_len=gt_focal_len,
                save_sed_graph_to=None,
                gpu_id=gpu_id,
            )
            return (data_dir, consistent_ratios)

    tasks = []
    for idx, scene in enumerate(os.listdir(output_root)):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue
        for motion_degree in os.listdir(scene_path):
            data_dir = os.path.join(output_root, scene, motion_degree)
            assert os.path.isdir(data_dir)
            tasks.append((data_dir, 260.0, GPUS[idx % len(GPUS)]))

    with ProcessingPool(nodes=32) as pool:
        results = list(tqdm(pool.imap(run_task, tasks), total=len(tasks), desc="Calculating SED"))

    total_results = {}
    missing = []
    for data_dir, result in results:
        if result is None:
            missing.append(data_dir)
        else:
            total_results[data_dir] = result

    total_consistent_ratios = {}
    for result in total_results.values():
        for key, value in result.items():
            if key not in total_consistent_ratios:
                total_consistent_ratios[key] = 0
            total_consistent_ratios[key] += value
    for key in total_consistent_ratios:
        total_consistent_ratios[key] /= len(total_results)
        print(f"Total SED Mean (threshold {key:.2f}): {total_consistent_ratios[key]:.3f}")

    total_results["total_sed_mean"] = total_consistent_ratios
    total_results["missing_videos"] = missing

    return total_results, missing


if __name__ == '__main__':

    SUFFIX = "nvssolver"
    davis_output_root = f"./davis_output_{SUFFIX}"
    mannequin_challenge_output_root = f"./mannequin_challenge_output_{SUFFIX}"
    tanks_and_temples_output_root = f"./tanks_and_temples_output_{SUFFIX}"

    DATAROOT= "/home/ryotaro/data"
    davis_data_root = f"{DATAROOT}/DAVIS/JPEGImages/Full-Resolution"
    mannequin_challenge_data_root = f"{DATAROOT}/MannequinChallengeHQ/validation_frames"
    tanks_and_temples_data_root = f"{DATAROOT}/TanksAndTemples"

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

    with tempfile.TemporaryDirectory(dir=DATAROOT) as td:
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
            # 5. FID/FVD calculation
            fid_score = run_fid_calculation(
                data_root=data_root,
                output_root=output_root,
            )
            fvd_videogpt, fvd_stylegan = run_fvd_calculation(
                davis_data_root=data_root,
                davis_output_root=output_root,
            )
            # 7. SED calculation
            sed_results = run_sed_calculation(output_root)

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, shutting down.")
        finally:
            print("Program finished.")
