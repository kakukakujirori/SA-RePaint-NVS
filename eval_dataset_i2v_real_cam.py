import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import subprocess
import tempfile
from functools import partial
from itertools import product
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

import cv2
import imagesize
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from cleanfid import fid
from depth_anything_3.api import DepthAnything3
from diffusers.utils.loading_utils import load_image
from einops import rearrange
from fvdcal import FVDCalculation
from met3r import MEt3R
from pathos.multiprocessing import ProcessingPool
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio

from src.eval_sed import eval_sed
from src.eval_ssim import ssim
from src.eval_trajectories import eval_trajectories, run_glomap
from src.trajectory_extraction import render_mesh, forward_warp, save_trajectories


NUM_FRAMES = 25
NUM_INFERECE_STEPS = 50
DENOISE_START_STEP = NUM_INFERECE_STEPS // 3
REPAINT_ITER_NUM = 2

GPUS = [0, 1, 2, 3]
MAX_WORKER_NUM = 16


def resize_image(filepath: str, width: int, height: int):
    try:
        subprocess.run(["magick", "mogrify", "-resize", f"{width}x{height}!", filepath],
            check=True, capture_output=True, text=True)
        return (filepath, "Success")
    except subprocess.CalledProcessError as e:
        return (filepath, f"Error: {e.stderr.strip()}")


def reorganize_frames(mannequin_challenge_data_root: str):
    """This script is expected to be run after
    `python download_extract.py` is executed."""
    for split in ["validation", "test", "train"]:
        split_root = os.path.join(mannequin_challenge_data_root, split, "data")
        output_root = os.path.join(mannequin_challenge_data_root, f"{split}_frames")
        assert os.path.isdir(split_root), f"Directory {split_root} does not exist."

        if os.path.isdir(output_root):
            shutil.rmtree(output_root)
        os.makedirs(output_root)

        cnt = 0
        for uid in os.listdir(split_root):
            frame_dir = os.path.join(split_root, uid, "frames")
            if not os.path.isdir(frame_dir):
                continue

            dst_dir = os.path.join(output_root, uid)
            if os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(frame_dir, dst_dir)
            cnt += 1

        print(f"[extract_frames] Finished reorganizing '{split}' ({cnt}/{len(os.listdir(split_root))})")


def organize_images_and_depth(data_root: str, input_root: str, output_root: str):
    assert os.path.isdir(data_root), f"Folder not found: {data_root}"
    if os.path.isdir(input_root):
        shutil.rmtree(input_root)
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    os.makedirs(input_root)
    os.makedirs(output_root)

    # extract keyframes from each scene
    for i, scene in enumerate(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(scene):
            continue

        scene_name = os.path.basename(scene)
        print(scene_name)

        current_dir = ""
        for cnt, imgpath in enumerate(sorted(glob.glob(os.path.join(scene, "*.jpg")))):
            # pool images
            if cnt % NUM_FRAMES == 0:
                img_num = os.path.basename(imgpath).split(".")[0]
                current_dir = os.path.join(input_root, scene_name + "_" + img_num)
                os.makedirs(current_dir)
                os.makedirs(os.path.join(current_dir, "images"))

            shutil.copy(imgpath, os.path.join(current_dir, f"images/{cnt % NUM_FRAMES:04d}.jpg"))

        # discard the last dir if not enough frames
        if len(os.listdir(os.path.join(current_dir, "images"))) < NUM_FRAMES:
            shutil.rmtree(current_dir)

    # resize to 1024x576
    file_list = glob.glob(os.path.join(input_root, "*/images/*.jpg"))
    with Pool(processes=os.cpu_count()) as pool:
        resize_image_1024x576 = partial(resize_image, width=1024, height=576)
        results = list(tqdm(pool.imap_unordered(resize_image_1024x576, file_list), total=len(file_list), desc="Resizing images to 1024x576"))
        errors = [r for r in results if r[1] != "Success"]
        if errors:
            print("\n--- The following files failed to process: ---")
            for filepath, error_message in errors:
                print(f"- {filepath}\n  {error_message}")

    # output directories
    process_dir_list = []
    for scene_list in os.listdir(input_root):
        input_image_dir = os.path.join(input_root, scene_list, "images")
        output_depth_dir = os.path.join(input_root, scene_list, "depths")
        output_camera_dir = os.path.join(input_root, scene_list, "cameras")
        os.makedirs(output_depth_dir, exist_ok=True)
        os.makedirs(output_camera_dir, exist_ok=True)
        process_dir_list.append((input_image_dir, output_depth_dir, output_camera_dir))

    # DA3
    da3_models = {gpu_id: DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(f"cuda:{gpu_id}").eval() for gpu_id in GPUS}

    process_dir_list = [(da3_models[GPUS[i % len(GPUS)]], *process_dir) for i, process_dir in enumerate(process_dir_list)]

    # Use ThreadPoolExecutor to avoid CUDA initialization issues in forked processes
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = {executor.submit(run_da3, *args): args for args in process_dir_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(process_dir_list), desc="Running DA3"):
            try:
                _ = future.result()
            except Exception as e:
                # Capture exception and the arguments that caused it
                args = futures[future]
                print(f"Error processing {args[1]}: {e}")


@torch.no_grad()
def run_da3(model: DepthAnything3, input_image_dir: str, output_depth_dir: str, output_camera_dir: str):
    image_paths = sorted(glob.glob(os.path.join(input_image_dir, "*.jpg")))
    width_ori, height_ori = imagesize.get(image_paths[0])

    # run
    prediction = model.inference(image_paths)
    _, height_sh, width_sh, _ = prediction.processed_images.shape
    rgb = prediction.processed_images
    depth_map = prediction.depth
    extrinsics_3x4 = prediction.extrinsics
    intrinsics = prediction.intrinsics

    # postprocess
    extrinsics = np.eye(4)[None, :, :].repeat(extrinsics_3x4.shape[0], axis=0)
    extrinsics[:, :3, :4] = extrinsics_3x4
    intrinsics = np.mean(intrinsics, axis=0)  # prevent flickering
    intrinsics[0, :] *= width_ori / width_sh
    intrinsics[1, :] *= height_ori / height_sh

    # save
    np.save(os.path.join(output_camera_dir, f"intrinsics.npy"), intrinsics)
    for i, (dep, extr) in enumerate(zip(depth_map, extrinsics)):
        dep = cv2.resize(dep, (width_ori, height_ori))
        np.save(os.path.join(output_depth_dir, f"{i:04d}.npy"), dep)
        np.save(os.path.join(output_camera_dir, f"{i:04d}_extr.npy"), extr)

    return depth_map, extrinsics, intrinsics


def render_point_cloud(
        input_scene_dir: str,
        output_warped_dir: str,
        no_occlusion_revealing: bool = True,
        save_trajectory_type: str | None = None,
        use_mesh: bool = True,
    ):

    assert os.path.isdir(input_scene_dir)
    assert os.path.isdir(output_warped_dir)

    # load data
    image_path = sorted(glob.glob(os.path.join(input_scene_dir, "images/*.jpg")))
    image_list = [np.array(Image.open(ip)) for ip in image_path]

    depth_path_npy = sorted(glob.glob(os.path.join(input_scene_dir, "depths/*.npy")))
    depth_list = [np.load(dp) for dp in depth_path_npy]

    camera_path_extr = sorted(glob.glob(os.path.join(input_scene_dir, "cameras/*_extr.npy")))
    camera_list_extr = [np.load(dp) for dp in camera_path_extr]

    assert len(image_list) == len(depth_list) == len(camera_list_extr)

    # render
    img_s = image_list[0]
    dep_s = depth_list[0]
    extr_s = camera_list_extr[0]
    intr_s = np.load(os.path.join(input_scene_dir, "cameras/intrinsics.npy"))
    never_occluded = np.ones_like(depth_list[0], dtype=bool)

    trans_coordinates_3D_list = []
    trans_valid_list = []

    for i, extr_t in enumerate(camera_list_extr):
        np.save(os.path.join(output_warped_dir, str(i).zfill(4)+"_cam_extr.npy"), extr_t)

        if use_mesh:
            warped_frame2, mask2, trans_coordinates_3D, trans_valid = render_mesh(
                img_s,
                dep_s,
                extr_s,
                extr_t,
                intr_s,
                None,
                mask_deocclusion=no_occlusion_revealing,
            )
            if i == 0:
                # warped_frame2 = img_s  # rendering results are geometrically slightly distorted, so we don't use the original frame
                mask2 = np.zeros_like(mask2)

            # binarize masks
            mask2 = np.where(mask2 > 0, 255, 0).astype(np.uint8)

            # save images
            warped_frame2[mask2 > 0] = 0
            Image.fromarray(mask2).save(os.path.join(output_warped_dir, str(i).zfill(4)+"_mask.png"))
            Image.fromarray(warped_frame2).save(os.path.join(output_warped_dir, str(i).zfill(4)+".png"))

        else:
            warped_frame2, mask2, trans_coordinates_3D, trans_valid = forward_warp(
                img_s,
                never_occluded,
                dep_s,
                extr_s,
                extr_t,
                intr_s,
                None,
            )
            if no_occlusion_revealing:
                never_occluded *= trans_valid

            # save images
            mask = 1 - mask2
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = np.repeat(mask[:,:,np.newaxis]*255., repeats=3, axis=2)

            kernel = np.ones((5,5), np.uint8)
            mask_erosion = cv2.dilate(np.array(mask), kernel)
            mask_erosion = Image.fromarray(np.uint8(mask_erosion))
            mask_erosion.save(os.path.join(output_warped_dir, str(i).zfill(4)+"_mask.png"))

            mask_erosion_ = np.array(mask_erosion)/255.
            mask_erosion_[mask_erosion_ < 0.5] = 0
            mask_erosion_[mask_erosion_ >= 0.5] = 1
            warped_frame2 = Image.fromarray(np.uint8(warped_frame2 * (1-mask_erosion_)))
            warped_frame2.save(os.path.join(output_warped_dir, str(i).zfill(4)+".png"))

        trans_coordinates_3D_list.append(trans_coordinates_3D)
        trans_valid_list.append(trans_valid)

    # trajectory
    save_trajectories(
        output_warped_dir,
        save_trajectory_type,
        trans_coordinates_3D_list,
        trans_valid_list,
    )


def run_generation_task(input_root: str, output_root: str, scene: str, gpu_id: int, method: str = "faithful_svd") -> str:
    task_id = f"Scene: {scene}, GPU: {gpu_id}"
    print(f"STARTING task: {task_id}")
    with torch.cuda.device(f'cuda:{gpu_id}'):
        torch.cuda.empty_cache()
    try:
        if method == "faithful_svd":
            result = subprocess.run(["python", "src/generate_faithful_svd.py",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", f"{NUM_INFERECE_STEPS}",
                "--denoise_start_step", f"{DENOISE_START_STEP}",
                "--repaint_iter_num", f"{REPAINT_ITER_NUM}",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "faithful_wan":
            result = subprocess.run(["python", "src/generate_faithful_wan.py",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", f"{NUM_INFERECE_STEPS}",
                "--repaint_iter_num", f"{REPAINT_ITER_NUM}",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "nvssolver":
            result = subprocess.run(["python", "src/generate_nvssolver.py",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "100",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "trajattn":
            result = subprocess.run(["python", "src/generate_trajattn.py",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "25",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "trajcrafter":
            result = subprocess.run(["python", "src/generate_trajcrafter.py",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "50",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "das":
            result = subprocess.run(["python", "src/generate_das.py",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "50",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "invstitch":
            result = subprocess.run(["python", "src/generate_invstitch.py",
                "--input_folder", f"{input_root}/{scene}",
                "--output_folder", f"{output_root}/{scene}/generated",
                "--trajectory_folder", f"{output_root}/{scene}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--outpaint_frame_interval", "5",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")
        print(f"COMPLETED task: {task_id}\nSTDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"STDERR for {task_id}:\n{result.stderr.strip()}")
        msg = f"Successfully processed {task_id}"
    except subprocess.CalledProcessError as e:
        error_message = (
            f"ERROR for task: {task_id}\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"Stdout:\n{e.stdout.strip()}\n"
            f"Stderr:\n{e.stderr.strip()}"
        )
        print(error_message)
        msg = f"Failed processing {task_id}: {e.returncode}"
    except Exception as e:
        print(f"UNEXPECTED ERROR for task: {task_id}: {e}")
        msg = f"Unexpected error for {task_id}: {e}"

    with torch.cuda.device(f'cuda:{gpu_id}'):
        torch.cuda.empty_cache()

    return msg


def run_pixelwise_metrics_calculation(input_root: str, output_root: str, allow_resize: bool = False, allow_missing_frames: bool = False):
    PSNR_MODULES = {i: PeakSignalNoiseRatio(data_range=1.0).eval().to(f"cuda:{i}") for i in GPUS}
    LPIPS_MODULES = {i: lpips.LPIPS(net='alex', spatial=True).eval().to(f"cuda:{i}") for i in GPUS}

    total_results = {}
    missing = []

    def run_task(in_data_dir:str, out_data_dir: str, gpu_id: int):
        # select evaluator
        PSNR = PSNR_MODULES[gpu_id]
        LPIPS = LPIPS_MODULES[gpu_id]
        device = f"cuda:{gpu_id}"

        # load warped frames and generated frames
        generated_frame_paths = sorted(glob.glob(os.path.join(out_data_dir, "generated", "*.png")))
        valid_frame_ids = [int(os.path.basename(x).split(".")[0]) for x in generated_frame_paths]
        gt_frame_paths = [os.path.join(in_data_dir, "images", f"{i:04d}.jpg") for i in valid_frame_ids]
        mask_frame_paths = [os.path.join(out_data_dir, "warped", f"{i:04d}_mask.png") for i in valid_frame_ids]
        warped_frame_paths = [os.path.join(out_data_dir, "warped", f"{i:04d}.png") for i in valid_frame_ids]

        if not allow_missing_frames:
            assert len(gt_frame_paths) == len(mask_frame_paths) == len(warped_frame_paths) == len(generated_frame_paths) == NUM_FRAMES

        gt_frames = [load_image(x) for x in gt_frame_paths]
        generated_frames = [load_image(x) for x in generated_frame_paths]
        mask_frames = [load_image(x) for x in mask_frame_paths]
        warped_frames = [load_image(x) for x in warped_frame_paths]

        # batchfy the frames
        gt_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in gt_frames], dim=0).to(device)
        mask_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in mask_frames], dim=0).to(device)
        warped_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in warped_frames], dim=0).to(device)
        generated_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in generated_frames], dim=0).to(device)

        # align shapes (generated tensor may be smaller than warped_tensor)
        if allow_resize:
            print(f"[run_pixelwise_metrics_calculation] WARNING: Resizing input videos to match the generated video shapes.")
            _, _, h_gt, w_gt = gt_tensor.shape
            generated_tensor = F.interpolate(generated_tensor, (h_gt, w_gt), mode="bicubic")

        # binarize the mask
        mask_tensor_bool = mask_tensor < 0.5
        mask_tensor_float = mask_tensor_bool.float().mean(dim=1, keepdim=True)

        # [NOTE: IMPORTANT] fill invalid pixels (prevent black pixels from seeping into valid regions)
        warped_tensor = torch.where(mask_tensor_bool, warped_tensor, generated_tensor)

        # calculate psnr, ssim, lpips (NOTE: image range is [-1, 1] for LPIPS)
        results = {}
        with torch.no_grad():
            # PSNR
            psnr_score = PSNR(warped_tensor[mask_tensor_bool], generated_tensor[mask_tensor_bool])
            results["psnr_with_warped_on_mask"] = psnr_score.item()

            psnr_score = PSNR(gt_tensor[mask_tensor_bool], generated_tensor[mask_tensor_bool])
            results["psnr_with_gt_on_mask"] = psnr_score.item()

            psnr_score = PSNR(gt_tensor, generated_tensor)
            results["psnr_with_gt_full"] = psnr_score.item()

            # SSIM
            ssim_score = ssim(warped_tensor, generated_tensor, mask=mask_tensor_bool, data_range=1.0, size_average=True)
            results["ssim_with_warped_on_mask"] = ssim_score.item()

            ssim_score = ssim(gt_tensor, generated_tensor, mask=mask_tensor_bool, data_range=1.0, size_average=True)
            results["ssim_with_gt_on_mask"] = ssim_score.item()

            ssim_score = ssim(gt_tensor, generated_tensor, data_range=1.0, size_average=True)
            results["ssim_with_gt_full"] = ssim_score.item()

            # LPIPS
            lpips_full = LPIPS(warped_tensor * 2 - 1, generated_tensor * 2 - 1)
            lpips_score = torch.sum(lpips_full * mask_tensor_float) / torch.sum(mask_tensor_float)
            results["lpips_with_warped_on_mask"] = lpips_score.item()

            lpips_full = LPIPS(gt_tensor * 2 - 1, generated_tensor * 2 - 1)
            lpips_score = torch.sum(lpips_full * mask_tensor_float) / torch.sum(mask_tensor_float)
            results["lpips_with_gt_on_mask"] = lpips_score.item()

            lpips_full = LPIPS(gt_tensor * 2 - 1, generated_tensor * 2 - 1)
            lpips_score = torch.mean(lpips_full)
            results["lpips_with_gt_full"] = lpips_score.item()

        total_results[out_data_dir] = results


    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        future_to_task_info = {}
        for idx, scene in enumerate(sorted(os.listdir(output_root))):
            scene_path = os.path.join(output_root, scene)
            if not os.path.isdir(scene_path):
                continue

            in_data_dir = os.path.join(input_root, scene)
            out_data_dir = os.path.join(output_root, scene)
            assert os.path.isdir(out_data_dir)

            if not os.path.isdir(os.path.join(out_data_dir, "generated")):
                print(f"Missing {os.path.join(out_data_dir, 'generated')}")
                missing.append(out_data_dir)
                continue

            future = executor.submit(run_task, in_data_dir, out_data_dir, GPUS[idx % len(GPUS)])
            future_to_task_info[future] = (in_data_dir, out_data_dir, GPUS[idx % len(GPUS)])

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Calculating pixelwise metrics"):
            in_data_dir, out_data_dir, gpu_id = future_to_task_info[future]
            task_desc = f"{in_data_dir=}, {out_data_dir=}, {gpu_id=}"
            try:
                result_message = future.result()
                # print(f"Result for {task_desc}: {result_message}")
            except Exception as exc:
                print(f"Main loop caught exception for {task_desc}: {exc}")
                raise exc


    total_psnr_with_warped_on_mask = sum([result["psnr_with_warped_on_mask"] for result in total_results.values()]) / len(total_results)
    total_psnr_with_gt_on_mask = sum([result["psnr_with_gt_on_mask"] for result in total_results.values()]) / len(total_results)
    total_psnr_with_gt_full = sum([result["psnr_with_gt_full"] for result in total_results.values()]) / len(total_results)
    total_ssim_with_warped_on_mask = sum([result["ssim_with_warped_on_mask"] for result in total_results.values()]) / len(total_results)
    total_ssim_with_gt_on_mask = sum([result["ssim_with_gt_on_mask"] for result in total_results.values()]) / len(total_results)
    total_ssim_with_gt_full = sum([result["ssim_with_gt_full"] for result in total_results.values()]) / len(total_results)
    total_lpips_with_warped_on_mask = sum([result["lpips_with_warped_on_mask"] for result in total_results.values()]) / len(total_results)
    total_lpips_with_gt_on_mask = sum([result["lpips_with_gt_on_mask"] for result in total_results.values()]) / len(total_results)
    total_lpips_with_gt_full = sum([result["lpips_with_gt_full"] for result in total_results.values()]) / len(total_results)
    print("----------------------------------------------------------------")
    print(f"Total PSNR with warped on mask: {total_psnr_with_warped_on_mask}")
    print(f"Total PSNR with gt on mask: {total_psnr_with_gt_on_mask}")
    print(f"Total PSNR with gt full: {total_psnr_with_gt_full}")
    print("----------------------------------------------------------------")
    print(f"Total SSIM with warped on mask: {total_ssim_with_warped_on_mask}")
    print(f"Total SSIM with gt on mask: {total_ssim_with_gt_on_mask}")
    print(f"Total SSIM with gt full: {total_ssim_with_gt_full}")
    print("----------------------------------------------------------------")
    print(f"Total LPIPS with warped on mask: {total_lpips_with_warped_on_mask}")
    print(f"Total LPIPS with gt on mask: {total_lpips_with_gt_on_mask}")
    print(f"Total LPIPS with gt full: {total_lpips_with_gt_full}")
    print("----------------------------------------------------------------")
    print(f"Missing dirs: {len(missing)}")
    total_results["total_psnr_with_warped_on_mask"] = total_psnr_with_warped_on_mask
    total_results["total_psnr_with_gt_on_mask"] = total_psnr_with_gt_on_mask
    total_results["total_psnr_with_gt_full"] = total_psnr_with_gt_full
    total_results["total_ssim_with_warped_on_mask"] = total_ssim_with_warped_on_mask
    total_results["total_ssim_with_gt_on_mask"] = total_ssim_with_gt_on_mask
    total_results["total_ssim_with_gt_full"] = total_ssim_with_gt_full
    total_results["total_lpips_with_warped_on_mask"] = total_lpips_with_warped_on_mask
    total_results["total_lpips_with_gt_on_mask"] = total_lpips_with_gt_on_mask
    total_results["total_lpips_with_gt_full"] = total_lpips_with_gt_full
    total_results["missing_dirs"] = missing

    return total_results, missing


def run_fid_kid_calculation(data_root: str, output_root: str):
    with tempfile.TemporaryDirectory() as td:
        print(f"[run_fid_calculation] Images are copied to {td} for FID/KID calculation.")
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
            for imgpath in glob.glob(os.path.join(output_root, scene, "generated", "*.png")):
                imgpath = os.path.abspath(imgpath)
                dst_path = os.path.join(generated_folder, imgpath.replace("/", "_"))
                shutil.copy(imgpath, dst_path)

        # calculate FID/KID
        score_fid = fid.compute_fid(gt_folder, generated_folder).item()
        score_kid = fid.compute_kid(gt_folder, generated_folder)
        print(f"FID: {score_fid}, KID: {score_kid}")
        return score_fid, score_kid


def run_fvd_calculation(data_root: str, output_root: str):
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
    for scene_imgname in os.listdir(output_root):
        if not os.path.isdir(os.path.join(output_root, scene_imgname)):
            continue
        scene, imgname = scene_imgname.split("_", maxsplit=1)
        if scene not in start_frames_per_scene:
            start_frames_per_scene[scene] = []
        start_frames_per_scene[scene].append(imgname)

    gt_video_list = []
    sample_video_list = []

    def run_task(scene: str, start_frames: list[str]):
        gt_frames = sorted(glob.glob(os.path.join(data_root, scene, "*.jpg")))

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
            sample_video = [os.path.join(output_root, scene_imgname, f"generated/{i:04d}.png") for i in range(NUM_FRAMES)]
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


    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER_NUM) as executor:
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


def run_camera_pose_error_calculation(output_root: str, gt_focal_len: float = 260.0):

    def run_task(args: tuple[str, int]):
        data_dir, gpu_id = args

        if not os.path.isfile(os.path.join(data_dir, "generated/generated.mp4")):
            print(f"Missing video for {data_dir}")
            return (data_dir, None)

        # Load the generated frames
        image_names = sorted(glob.glob(os.path.join(data_dir, f"generated/0*.png")))
        image_ids = [int(os.path.basename(x).split(".")[0]) for x in image_names]

        # Load the gt image size
        gt_width, gt_height = imagesize.get(os.path.join(data_dir, f"warped/0000.png"))

        # Load the camera poses
        gt_poses_w2c = [np.load(os.path.join(data_dir, f"warped/{i:04d}_cam_extr.npy")) for i in image_ids]
        gt_poses_c2w = [np.linalg.inv(p) for p in gt_poses_w2c]
        gt_K = np.array([
            [gt_focal_len, 0, gt_width/2],
            [0, gt_focal_len, gt_height/2],
            [0, 0, 1],
        ], dtype=np.float32)

        # Estimate the camera poses from the generated video
        estimated_poses_c2w = run_glomap(image_names, gt_width=gt_width, gt_height=gt_height, gt_focal_len=gt_focal_len, gpu_id=gpu_id)
        estimated_K = gt_K

        # Filter out missing poses
        missing_estimated_poses_ids = [i for (i, pose) in enumerate(estimated_poses_c2w) if pose is None]
        estimated_poses_c2w = [pose for (i, pose) in enumerate(estimated_poses_c2w) if i not in missing_estimated_poses_ids]
        gt_poses_c2w = [pose for (i, pose) in enumerate(gt_poses_c2w) if i not in missing_estimated_poses_ids]
        if len(estimated_K.shape) == 3:  # (N, 3, 3)
            estimated_K = np.array([estimated_K[i] for i in range(len(estimated_K)) if i not in missing_estimated_poses_ids])

        # Compute errors
        if estimated_poses_c2w:
            results, estimated_abs_poses_c2w, gt_abs_poses_c2w = eval_trajectories(estimated_poses_c2w, gt_poses_c2w, estimated_K, gt_K)
            return (data_dir, results)
        else:
            print(f"[WARNING] `estimated_poses_c2w` was empty for {data_dir}. This is usually unexpected; check the data manually.")
            return (data_dir, None)

    tasks = []
    for idx, scene in enumerate(os.listdir(output_root)):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue

        data_dir = os.path.join(output_root, scene)
        assert os.path.isdir(data_dir)
        tasks.append((data_dir, GPUS[idx % len(GPUS)]))

    with ProcessingPool(nodes=MAX_WORKER_NUM) as pool:
        results = list(tqdm(pool.imap(run_task, tasks), total=len(tasks), desc="Calculating camera pose errors"))

    total_results = {}
    missing = []
    for data_dir, result in results:
        if result is None:
            missing.append(data_dir)
        else:
            total_results[data_dir] = result

    total_ape_mean = sum([result["ape_mean"] for result in total_results.values()]) / len(total_results)
    total_rre_mean = sum([result["rre_mean"] for result in total_results.values()]) / len(total_results)
    total_rte_mean = sum([result["rte_mean"] for result in total_results.values()]) / len(total_results)
    total_ape_median = np.median(np.array([result["ape_mean"] for result in total_results.values()]))
    total_rre_median = np.median(np.array([result["rre_mean"] for result in total_results.values()]))
    total_rte_median = np.median(np.array([result["rte_mean"] for result in total_results.values()]))
    print(f"Total APE Mean: {total_ape_mean}, Median: {total_ape_median}")
    print(f"Total RRE Mean: {total_rre_mean}, Median: {total_rre_median}")
    print(f"Total RTE Mean: {total_rte_mean}, Median: {total_rte_median}")
    print(f"Missing videos: {len(missing)}")
    total_results["total_ape_mean"] = total_ape_mean
    total_results["total_rre_mean"] = total_rre_mean
    total_results["total_rte_mean"] = total_rte_mean
    total_results["total_ape_median"] = total_ape_median
    total_results["total_rre_median"] = total_rre_median
    total_results["total_rte_median"] = total_rte_median
    total_results["missing_videos"] = missing

    return total_results, missing


def run_sed_calculation(output_root: str):

    def run_task(args: tuple[str, float, int]):
        data_dir, gt_focal_len, gpu_id = args

        video_path = os.path.join(data_dir, "generated/generated.mp4")
        if not os.path.isfile(video_path):
            return (data_dir, None)

        # Load the gt image size and camera poses
        gt_width, gt_height = imagesize.get(os.path.join(data_dir, f"warped/0000.png"))
        camera_paths = os.path.join(data_dir, "warped/*_cam_extr.npy")
        poses = [np.load(p) for p in sorted(glob.glob(camera_paths))]

        with tempfile.TemporaryDirectory() as colmap_root:
            consistent_ratios, sed_summary = eval_sed(
                colmap_root=colmap_root,
                video_path=video_path,
                poses=poses,
                gt_width=gt_width,
                gt_height=gt_height,
                gt_focal_len=gt_focal_len,
                sed_thresholds=(0.0, 10.0),
                save_sed_graph_to=None,
                gpu_id=gpu_id,
            )
            return (data_dir, consistent_ratios)

    tasks = []
    for idx, scene in enumerate(os.listdir(output_root)):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue
        data_dir = os.path.join(output_root, scene)
        assert os.path.isdir(data_dir)
        tasks.append((data_dir, 260.0, GPUS[idx % len(GPUS)]))

    with ProcessingPool(nodes=MAX_WORKER_NUM) as pool:
        results = list(tqdm(pool.imap(run_task, tasks), total=len(tasks), desc="Calculating SED"))

    total_results = {}
    missing = []
    for data_dir, result in results:
        if result is None:
            missing.append(data_dir)
        else:
            total_results[data_dir] = result

    total_consistent_ratios = {}
    probably_failed = []
    for data_dir, result in total_results.items():
        for key, value in result.items():
            if key not in total_consistent_ratios:
                total_consistent_ratios[key] = 0
            total_consistent_ratios[key] += value
        # sanity check
        if max(result.values()) == 0:
            probably_failed.append(data_dir)

    for key in total_consistent_ratios:
        total_consistent_ratios[key] /= len(total_results)
        print(f"Total SED Mean (threshold {key:.2f}): {total_consistent_ratios[key]:.3f}")
    if probably_failed:
        print("[run_sed_calcluation] All SED values are 0 in the following data_dir. "
              "This may indicate that the multiprocess worker failed to process the data. "
              "Consider reducing MAX_WORKER_NUM.")
        for data_dir in probably_failed:
            print("  " + data_dir)


    total_results["total_sed_mean"] = total_consistent_ratios
    total_results["missing_videos"] = missing

    return total_results, missing


def run_met3r_calculation(output_root: str, process_size: int = 256, resize_mode: str = "area"):
    # Initialize MEt3R
    MET3R_MODULES = {i: MEt3R(
            img_size=None, # Default to 256, set to `None` to use the input resolution on the fly!
            use_norm=True, # Default to True
            backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
            feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
            feature_backbone_weights="mhamilton723/FeatUp", # Default
            upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
            distance="cosine", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
            freeze=True, # Default to True
        ).to(f"cuda:{i}") for i in GPUS}

    def run_task(data_dir: str, gpu_id: int, process_size: int, resize_mode: str):
        # select evaluator
        metric = MET3R_MODULES[gpu_id]
        device = f"cuda:{gpu_id}"

        # load warped frames and generated frames
        generated_frames = [load_image(os.path.join(data_dir, "generated", f"{i:04d}.png")) for i in range(NUM_FRAMES)]
        generated_tensor_source = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in generated_frames[:-1]], dim=0)
        generated_tensor_target = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in generated_frames[1:]], dim=0)
        generated_tensor = torch.stack([generated_tensor_source, generated_tensor_target], dim=1) * 2 - 1

        # resize (NOTE: size must be multiple of 32)
        height, width = generated_tensor.shape[-2:]
        if width >= height:
            width_sh = process_size
            height_sh = int(32 * round(process_size * height / (32 * width)))
        else:
            height_sh = process_size
            width_sh = int(32 * round(process_size * width / (32 * height)))
        generated_tensor = F.interpolate(
            rearrange(generated_tensor, "b v c h w -> (b v) c h w", v=2),
            size=(height_sh, width_sh),
            mode=resize_mode,
        )
        generated_tensor = rearrange(generated_tensor, "(b v) c h w -> b v c h w", v=2)
        # print(f"Process size: {generated_tensor.shape}")

        # Evaluate MEt3R
        scores = {}
        for idx, inputs in enumerate(generated_tensor):
            s, *_ = metric(
                images=inputs.unsqueeze(0).to(device),
                return_overlap_mask=False, # Default
                return_score_map=False, # Default
                return_projections=False # Default
            )
            scores[f"{idx}_{idx+1}"] = s.cpu().item()
        total_results[data_dir] = scores

        # Clear up GPU memory
        torch.cuda.empty_cache()


    total_results = {}
    missing = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        future_to_task_info = {}
        for idx, scene in enumerate(sorted(os.listdir(output_root))):
            scene_path = os.path.join(output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            data_dir = os.path.join(output_root, scene)
            assert os.path.isdir(data_dir)

            if not os.path.isdir(os.path.join(data_dir, "generated")):
                print(f"Missing {os.path.join(data_dir, 'generated')}")
                missing.append(data_dir)
                continue

            future = executor.submit(run_task, data_dir, GPUS[idx % len(GPUS)], process_size, resize_mode)
            future_to_task_info[future] = (data_dir, GPUS[idx % len(GPUS)])

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Calculating MEt3R metrics"):
            data_dir, gpu_id = future_to_task_info[future]
            task_desc = f"{data_dir=}, {gpu_id=}"
            try:
                result_message = future.result()
                # print(f"Result for {task_desc}: {result_message}")
            except Exception as exc:
                print(f"Main loop caught exception for {task_desc}: {exc}")
                raise exc

    total_met3r_values = []
    for scores in total_results.values():
        total_met3r_values += list(scores.values())
    total_met3r_values = np.array(total_met3r_values)
    total_met3r_mean = np.mean(total_met3r_values)
    total_met3r_median = np.median(total_met3r_values)
    print(f"Total MEt3R Mean: {total_met3r_mean}")
    print(f"Total MEt3R Median: {total_met3r_median}")
    print(f"Missing dirs: {len(missing)}")
    total_results["total_met3r_mean"] = total_met3r_mean
    total_results["total_met3r_median"] = total_met3r_median
    total_results["missing_dirs"] = missing

    return total_results, missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-to-Video Evaluation")
    parser.add_argument("dataset", type=str, choices=["mannequin", "dl3dv_half"], help="Dataset to use for evaluation.")
    parser.add_argument("--data_root", type=str, default="/data/ryotaro/data/", help="Root directory of the datasets.")
    parser.add_argument("--method", type=str, default="faithful_svd", choices=["faithful_svd", "faithful_wan", "nvssolver", "trajattn", "trajcrafter", "das", "invstitch"], help="Method to use for generation. 'nvssolver' uses NVS-Solver, 'trajattn' uses Trajectory Attention, and 'das' uses DiffusionAsShader.")
    parser.add_argument("--use_mesh", action="store_true", help="If set, use mesh for trajectory extraction.")
    parser.add_argument("--scratch", action="store_true", help="If set, all the images, depth, and trajectories are re-organized and re-generated.")
    args = parser.parse_args()

    # 0. Set up paths
    if args.dataset == "mannequin":
        data_root = os.path.join(args.data_root, "MannequinChallengeHQ/validation_frames")
        input_root = "./mannequin_challenge_input_real_cam"
        output_root = "./mannequin_challenge_output_real_cam"
    elif args.dataset == "dl3dv_half":
        original_data_root = os.path.join(args.data_root, "DL3DV-Evaluation-img4/images")
        input_root = "./dl3dv_half_input_real_cam"
        output_root = "./dl3dv_half_output_real_cam"

        # create tmp data folder
        data_root = "/tmp/dl3dv_half"

        if args.scratch or not os.path.isdir(data_root):
            if os.path.isdir(data_root):
                shutil.rmtree(data_root)
            os.makedirs(data_root)

            for idx, scene in enumerate(tqdm(sorted(os.listdir(original_data_root)), desc="Copying DL3DV scenes")):
                scene_path = os.path.join(original_data_root, scene)
                assert os.path.isdir(scene_path), f"Scene path {scene_path} is not a directory."

                if idx % 2 == 1: continue  # reduce the data amount by half

                # copy images
                src_scene_path = os.path.join(original_data_root, scene, scene, "gaussian_splat/images_4")
                dst_scene_path = os.path.join(data_root, scene)
                os.makedirs(dst_scene_path, exist_ok=True)
                for imgpath in glob.glob(os.path.join(src_scene_path, "*.png")):
                    Image.open(imgpath).save(os.path.join(dst_scene_path, os.path.basename(imgpath).replace(".png", ".jpg")), "JPEG")

    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented.")

    # 1. Organize RGB images & Depth estimation
    if args.scratch:
        # if args.dataset == "mannequin":
        #     reorganize_frames(data_root)
        organize_images_and_depth(data_root=data_root, input_root=input_root, output_root=output_root)

        # 2. Trajectory Extraction
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKER_NUM, mp_context=multiprocessing.get_context('spawn')) as executor:
            future_to_task_info = {}
            for scene in os.listdir(input_root):
                input_scene = os.path.join(input_root, scene)
                output_warped_dir = os.path.join(output_root, scene, "warped")
                os.makedirs(output_warped_dir, exist_ok=True)

                future = executor.submit(
                    render_point_cloud,
                    input_scene,
                    output_warped_dir,
                    save_trajectory_type=("2d_npy" if args.method == "trajattn" else "3d_rgb" if args.method == "das" else "2d_homography" if args.method.startswith("faithful_") else None),
                    use_mesh=args.use_mesh,
                )
                future_to_task_info[future] = scene

            # Collect results (and handle exceptions)
            for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Rendering"):
                scene = future_to_task_info[future]
                task_desc = f"Scene: {scene}"
                try:
                    result_message = future.result()
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc:
                    print(f"Main loop caught exception for {task_desc}: {exc}")

    # count unprocessed tasks
    assert os.path.isdir(output_root)
    scene_list = []
    for scene in sorted(os.listdir(output_root)):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue

        assert os.path.isdir(os.path.join(output_root, scene, "warped"))
        if os.path.isdir(os.path.join(output_root, scene, "generated")):
            continue
        scene_list.append(scene)

    print(f"Total number of tasks to run: {len(scene_list)}")

    try:
        # 3. Generation
        job_idx_for_gpu_assignment = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
            future_to_task_info = {}
            for scene in scene_list:
                gpu_id_for_task = GPUS[job_idx_for_gpu_assignment % len(GPUS)]
                future = executor.submit(run_generation_task, input_root, output_root, scene, gpu_id_for_task, args.method)
                future_to_task_info[future] = (scene, gpu_id_for_task)
                job_idx_for_gpu_assignment += 1 # This ensures round-robin submission to GPUs

            # Collect results (and handle exceptions)
            for future in concurrent.futures.as_completed(future_to_task_info):
                scene, gpu_id = future_to_task_info[future]
                task_desc = f"Scene: {scene}, GPU: {gpu_id}"
                try:
                    result_message = future.result() # This will re-raise exceptions from run_generation_task
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc: # Should be caught by try/except in run_generation_task but good to have a fallback here.
                    print(f"Main loop caught exception for {task_desc}: {exc}")
                    raise exc

        # 4. Pixelwise metrics calculation
        pixelwise_results, _ = run_pixelwise_metrics_calculation(
            input_root,
            output_root,
            allow_resize=(args.method in ["faithful_wan", "trajcrafter", "das"]),
            allow_missing_frames=(args.method == "invstitch"),
        )
        with open(os.path.join(output_root, "pixelwise_results.txt"), "w") as f:
            json.dump(pixelwise_results, f, indent=4)

        # 5. Camera pose error calculation
        camera_pose_results, _ = run_camera_pose_error_calculation(output_root)
        with open(os.path.join(output_root, "camera_pose_results.txt"), "w") as f:
            json.dump(camera_pose_results, f, indent=4)

        # 6-1. FID/KID calculation
        fid_score, kid_score = run_fid_kid_calculation(
            data_root=data_root,
            output_root=output_root,
        )
        with open(os.path.join(output_root, "fid_fvd.txt"), "w") as f:
            f.write(f"FID: {fid_score}\nKID: {kid_score}")

        if args.method == "invstitch":
            print("InvStitch skips FVD and SED/MEt3R calculation.")
            exit()

        # 6-2. FVD calculation
        fvd_videogpt, fvd_stylegan = run_fvd_calculation(
            data_root=data_root,
            output_root=output_root,
        )
        with open(os.path.join(output_root, "fid_fvd.txt"), "a") as f:
            f.write(f"\nFVD (VideoGPT): {fvd_videogpt}\nFVD (StyleGAN): {fvd_stylegan}")

        # 7. SED calculation
        sed_results = run_sed_calculation(output_root)
        with open(os.path.join(output_root, "sed.txt"), "w") as f:
            json.dump(sed_results, f, indent=4)

        # 8. MEt3R calculation
        met3r_results, _ = run_met3r_calculation(output_root, process_size=256, resize_mode="area")
        with open(os.path.join(output_root, "met3r.txt"), "w") as f:
            json.dump(met3r_results, f, indent=4)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, shutting down.")
    finally:
        print("Program finished.")
