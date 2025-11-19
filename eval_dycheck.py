import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import subprocess
import tempfile
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import imagesize
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from cleanfid import fid
from diffusers.utils import load_image
from einops import rearrange
from fvdcal import FVDCalculation
from met3r import MEt3R
from pathos.multiprocessing import ProcessingPool
from torchmetrics.image import PeakSignalNoiseRatio

from src.eval_sed import eval_sed
from src.eval_ssim import ssim
from src.eval_trajectories import eval_trajectories, run_glomap


NUM_FRAMES = 25
NUM_INFERECE_STEPS = 50
DENOISE_START_STEP = NUM_INFERECE_STEPS // 3
REPAINT_ITER_NUM = 2

PROCESS_INTERVAL = 10

GPUS = [0, 1]
MAX_WORKER_NUM = 16


def run_generation_task(output_root: str, scene: str, gpu_id: int, method: str = "mine") -> str:
    task_id = f"Scene: {scene}, GPU: {gpu_id}"
    print(f"STARTING task: {task_id}")
    with torch.cuda.device(f'cuda:{gpu_id}'):
        torch.cuda.empty_cache()
    try:
        if method == "mine":
            result = subprocess.run(["python", "src/generate.py",
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


def run_pixelwise_metrics_calculation(output_root: str, allow_resize: bool = False):
    PSNR_MODULES = {i: PeakSignalNoiseRatio(data_range=1.0).eval().to(f"cuda:{i}") for i in GPUS}
    LPIPS_MODULES = {i: lpips.LPIPS(net='alex', spatial=True).eval().to(f"cuda:{i}") for i in GPUS}

    total_results = {}
    missing = []

    def run_task(data_dir: str, gpu_id: int):
        # select evaluator
        PSNR = PSNR_MODULES[gpu_id]
        LPIPS = LPIPS_MODULES[gpu_id]
        device = f"cuda:{gpu_id}"

        # load warped frames and generated frames
        mask_frames = [load_image(os.path.join(data_dir, "warped", f"{i:04d}_mask.png")) for i in range(NUM_FRAMES)]
        warped_frames = [load_image(os.path.join(data_dir, "warped", f"{i:04d}.png")) for i in range(NUM_FRAMES)]
        generated_frames = [load_image(os.path.join(data_dir, "generated", f"{i:04d}.png")) for i in range(NUM_FRAMES)]

        # batchfy the frames
        mask_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in mask_frames], dim=0).to(device)
        warped_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in warped_frames], dim=0).to(device)
        generated_tensor = torch.stack([torch.from_numpy(np.array(x).astype(np.float32) / 255.0).permute(2, 0, 1) for x in generated_frames], dim=0).to(device)

        # align shapes (generated tensor may be smaller than warped_tensor)
        if allow_resize:
            print(f"[run_pixelwise_metrics_calculation] WARNING: Resizing input videos to match the generated video shapes.")
            _, _, h_generated, w_generated = generated_tensor.shape
            mask_tensor = F.interpolate(mask_tensor, (h_generated, w_generated), mode="area")
            warped_tensor = F.interpolate(warped_tensor, (h_generated, w_generated), mode="bilinear")

        # binarize the mask
        mask_tensor_bool = mask_tensor < 0.5
        mask_tensor_float = mask_tensor_bool.float().mean(dim=1, keepdim=True)

        # [NOTE: IMPORTANT] fill invalid pixels (prevent black pixels from seeping into valid regions)
        warped_tensor = torch.where(mask_tensor_bool, warped_tensor, generated_tensor)

        # calculate psnr, ssim, lpips (NOTE: image range is [-1, 1] for LPIPS)
        results = {}
        with torch.no_grad():
            psnr_score = PSNR(warped_tensor[mask_tensor_bool], generated_tensor[mask_tensor_bool])
            results["psnr"] = psnr_score.item()

            ssim_score = ssim(warped_tensor, generated_tensor, mask=mask_tensor_bool, data_range=1.0, size_average=True)
            results["ssim"] = ssim_score.item()

            lpips_full = LPIPS(warped_tensor * 2 - 1, generated_tensor * 2 - 1)
            lpips_score = torch.sum(lpips_full * mask_tensor_float) / torch.sum(mask_tensor_float)
            results["lpips"] = lpips_score.item()

        total_results[data_dir] = results


    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        future_to_task_info = {}
        for idx, scene in enumerate(sorted(os.listdir(output_root))):
            scene_path = os.path.join(output_root, scene)
            if not os.path.isdir(scene_path):
                continue

            if not os.path.isdir(os.path.join(scene_path, "generated")):
                print(f"Missing {os.path.join(scene_path, 'generated')}")
                missing.append(scene_path)
                continue

            future = executor.submit(run_task, scene_path, GPUS[idx % len(GPUS)])
            future_to_task_info[future] = (scene, GPUS[idx % len(GPUS)])

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Calculating pixelwise metrics"):
            scene, gpu_id = future_to_task_info[future]
            task_desc = f"{scene=}, {gpu_id=}"
            try:
                result_message = future.result()
                # print(f"Result for {task_desc}: {result_message}")
            except Exception as exc:
                print(f"Main loop caught exception for {task_desc}: {exc}")
                raise exc


    total_psnr_mean = sum([result["psnr"] for result in total_results.values()]) / len(total_results)
    total_ssim_mean = sum([result["ssim"] for result in total_results.values()]) / len(total_results)
    total_lpips_mean = sum([result["lpips"] for result in total_results.values()]) / len(total_results)
    print(f"Total PSNR Mean: {total_psnr_mean}")
    print(f"Total SSIM Mean: {total_ssim_mean}")
    print(f"Total LPIPS Mean: {total_lpips_mean}")
    print(f"Missing dirs: {len(missing)}")
    total_results["total_psnr_mean"] = total_psnr_mean
    total_results["total_ssim_mean"] = total_ssim_mean
    total_results["total_lpips_mean"] = total_lpips_mean
    total_results["missing_dirs"] = missing

    return total_results, missing


def last_frame_comparison(output_root: str):
    PSNR_MODULE = PeakSignalNoiseRatio(data_range=1.0).eval().cuda()
    LPIPS_MODULE = lpips.LPIPS(net='alex', spatial=True).eval().cuda()

    total_results = {}
    missing = []

    for scene in os.listdir(output_root):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue

        warped_frame_path = os.path.join(output_root, scene, f"warped/{NUM_FRAMES - 1:04d}.png")
        mask_frame_path = os.path.join(output_root, scene, f"warped/{NUM_FRAMES - 1:04d}_mask.png")
        gt_frame_path = os.path.join(output_root, scene, f"warped/{NUM_FRAMES - 1:04d}_target.png")
        gen_frame_path = os.path.join(output_root, scene, f"generated/{NUM_FRAMES - 1:04d}.png")

        warped_frame = load_image(warped_frame_path).convert("RGB")
        mask_frame = load_image(mask_frame_path).convert("RGB")
        gt_frame = load_image(gt_frame_path).convert("RGB")
        generated_frame = load_image(gen_frame_path).convert("RGB")

        warped_tensor = torch.from_numpy(np.array(warped_frame).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        mask_tensor = torch.from_numpy((np.array(mask_frame).astype(np.float32) / 255.0) < 0.5).permute(2, 0, 1).unsqueeze(0).cuda()
        generated_tensor = torch.from_numpy(np.array(generated_frame).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        gt_tensor = torch.from_numpy(np.array(gt_frame).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

        # binarize the mask
        mask_tensor_bool = mask_tensor < 0.5
        mask_tensor_float = mask_tensor_bool.float().mean(dim=1, keepdim=True)

        # fill black pixels in warped tensor
        warped_tensor[~mask_tensor_bool] = gt_tensor[~mask_tensor_bool]

        # calculate psnr, ssim, lpips (NOTE: image range is [-1, 1] for LPIPS)
        results = {}
        with torch.no_grad():
            # with generated
            psnr_score = PSNR_MODULE(gt_tensor, generated_tensor)
            results["psnr_with_generated"] = psnr_score.item()

            ssim_score = ssim(gt_tensor, generated_tensor, data_range=1.0, size_average=True)
            results["ssim_with_generated"] = ssim_score.item()

            lpips_full = LPIPS_MODULE(gt_tensor * 2 - 1, generated_tensor * 2 - 1)
            results["lpips_with_generated"] = lpips_full.mean().item()

            # with warped
            psnr_score = PSNR_MODULE(gt_tensor[mask_tensor_bool], warped_tensor[mask_tensor_bool])
            results["psnr_with_warped"] = psnr_score.item()

            ssim_score = ssim(gt_tensor, warped_tensor, mask=mask_tensor_bool, data_range=1.0, size_average=True)
            results["ssim_with_warped"] = ssim_score.item()

            lpips_full = LPIPS_MODULE(gt_tensor * 2 - 1, warped_tensor * 2 - 1)
            lpips_score = torch.sum(lpips_full * mask_tensor_float) / torch.sum(mask_tensor_float)
            results["lpips_with_warped"] = lpips_score.item()

        total_results[scene] = results

    # per-scene results
    sequences = sorted(list({scene.split("_")[0] for scene in total_results.keys()}))
    per_scene_results = {
        "psnr": {seq: 0 for seq in sequences},
        "ssim": {seq: 0 for seq in sequences},
        "lpips": {seq: 0 for seq in sequences},
        "count": {seq: 0 for seq in sequences},
    }
    for scene, result in total_results.items():
        seq = scene.split("_")[0]
        per_scene_results["psnr"][seq] += result["psnr_with_generated"]
        per_scene_results["ssim"][seq] += result["ssim_with_generated"]
        per_scene_results["lpips"][seq] += result["lpips_with_generated"]
        per_scene_results["count"][seq] += 1
    for seq in sequences:
        per_scene_results["psnr"][seq] /= per_scene_results["count"][seq]
        per_scene_results["ssim"][seq] /= per_scene_results["count"][seq]
        per_scene_results["lpips"][seq] /= per_scene_results["count"][seq]
        print(f"[{seq}] PSNR: {per_scene_results['psnr'][seq]}, SSIM: {per_scene_results['ssim'][seq]}, LPIPS: {per_scene_results['lpips'][seq]}")

    # total results
    total_psnr_mean_with_generated = sum([result["psnr_with_generated"] for result in total_results.values()]) / len(total_results)
    total_ssim_mean_with_generated = sum([result["ssim_with_generated"] for result in total_results.values()]) / len(total_results)
    total_lpips_mean_with_generated = sum([result["lpips_with_generated"] for result in total_results.values()]) / len(total_results)
    print(f"Total PSNR Mean with Last-Frame GT: {total_psnr_mean_with_generated}")
    print(f"Total SSIM Mean with Last-Frame GT: {total_ssim_mean_with_generated}")
    print(f"Total LPIPS Mean with Last-Frame GT: {total_lpips_mean_with_generated}")
    print(f"Missing dirs: {len(missing)}")
    total_results["total_psnr_mean_with_generated"] = total_psnr_mean_with_generated
    total_results["total_ssim_mean_with_generated"] = total_ssim_mean_with_generated
    total_results["total_lpips_mean_with_generated"] = total_lpips_mean_with_generated
    total_results["missing_dirs"] = missing

    return total_results, missing


def run_camera_pose_error_calculation(output_root: str, gt_focal_len: float = 260.0):

    def run_task(args: tuple[str, int]):
        data_dir, gpu_id = args

        if not os.path.isfile(os.path.join(data_dir, "generated/generated_resized_518x518.mp4")):
            print(f"Missing video for {data_dir}")
            return (data_dir, None)

        # Load the generated frames
        image_names = sorted(glob.glob(os.path.join(data_dir, f"generated/0*.png")))
        num_frames = len(image_names)

        # Load the gt image size
        gt_width, gt_height = imagesize.get(os.path.join(data_dir, f"warped/0000.png"))

        # Load the camera poses
        gt_poses_w2c = [np.load(os.path.join(data_dir, f"warped/{i:04d}_pose.npy")) for i in range(num_frames)]
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
        if len(estimated_K) == num_frames:
            estimated_K = np.array([estimated_K[i] for i in range(num_frames) if i not in missing_estimated_poses_ids])

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
        tasks.append((scene_path, GPUS[idx % len(GPUS)]))

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

        video_path = os.path.join(data_dir, "generated/generated_resized_518x518.mp4")
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
        tasks.append((scene_path, 260.0, GPUS[idx % len(GPUS)]))

    with ProcessingPool(nodes=MAX_WORKER_NUM) as pool:
        results = list(tqdm(pool.imap(run_task, tasks), total=len(tasks), desc="Calculating SED"))

    total_results = {}
    missing = []
    for scene, result in results:
        if result is None:
            missing.append(scene)
        else:
            total_results[scene] = result

    total_consistent_ratios = {}
    probably_failed = []
    for scene, result in total_results.items():
        for key, value in result.items():
            if key not in total_consistent_ratios:
                total_consistent_ratios[key] = 0
            total_consistent_ratios[key] += value
        # sanity check
        if max(result.values()) == 0:
            probably_failed.append(scene)

    for key in total_consistent_ratios:
        total_consistent_ratios[key] /= len(total_results)
        print(f"Total SED Mean (threshold {key:.2f}): {total_consistent_ratios[key]:.3f}")
    if probably_failed:
        print("[run_sed_calcluation] All SED values are 0 in the following scene. "
              "This may indicate that the multiprocess worker failed to process the data. "
              "Consider reducing MAX_WORKER_NUM.")
        for scene in probably_failed:
            print("  " + scene)


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

            if not os.path.isdir(os.path.join(scene_path, "generated")):
                print(f"Missing {os.path.join(scene_path, 'generated')}")
                missing.append(scene_path)
                continue

            future = executor.submit(run_task, scene_path, GPUS[idx % len(GPUS)], process_size, resize_mode)
            future_to_task_info[future] = (scene, GPUS[idx % len(GPUS)])

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Calculating MEt3R metrics"):
            scene, gpu_id = future_to_task_info[future]
            task_desc = f"{scene=}, {gpu_id=}"
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


def resize_image(filepath: str, width: int, height: int):
    try:
        subprocess.run(["magick", "mogrify", "-resize", f"{width}x{height}!", filepath],
            check=True, capture_output=True, text=True)
        return (filepath, "Success")
    except subprocess.CalledProcessError as e:
        return (filepath, f"Error: {e.stderr.strip()}")


def resize_video(filepath: str, width: int, height: int):
    assert filepath.lower().endswith(".mp4")
    assert width % 2 == 0 and height % 2 == 0, "Width and height must be even numbers."
    basename, _ = os.path.splitext(filepath)
    output_path = f"{basename}_resized_{width}x{height}.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", filepath,
            "-vf", f"scale={width}:{height}",
            "-c:a", "copy",
            output_path],
            check=True, capture_output=True, text=True)
        return (filepath, "Success")
    except subprocess.CalledProcessError as e:
        return (filepath, f"Error: {e.stderr.strip()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-to-Video Evaluation")
    parser.add_argument("--dataset", type=str, choices=["dycheck"], default="dycheck", help="Dataset to use for evaluation.")
    parser.add_argument("--scratch", action="store_true", help="If set, all the images, depth, and trajectories are re-organized and re-generated.")
    parser.add_argument("--method", type=str, default="mine", choices=["mine", "nvssolver", "trajattn", "trajcrafter", "das"], help="Method to use for generation. 'nvssolver' uses NVS-Solver, 'trajattn' uses Trajectory Attention, and 'das' uses DiffusionAsShader.")
    parser.add_argument("--use_mesh", action="store_true", help="If set, use mesh for trajectory extraction.")
    args = parser.parse_args()

    # 0. Set up paths
    if args.dataset == "dycheck":
        data_root = "/mnt/hdd1/ryotaro/data/iphone"
        rendered_root = "./dycheck_rendered"
        output_root = "./dycheck_output"
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented.")

    # 1. Organize RGB images & Depth estimation
    available_scenes = ['apple', 'block', 'teddy', 'paper-windmill', 'spin']

    if args.scratch:
        # 2. Trajectory Extraction
        # for sequence in available_scenes:
        #     cmd = ["python", "src/render_dycheck.py", "--data_root", data_root, "--output_folder", rendered_root, "--sequence", sequence]
        #     cmd += ["--save_trajectory_type", "2d_npy"] if args.method == "trajattn" else []
        #     cmd += ["--use_mesh"] if args.use_mesh else []
        #     subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

        # thin out processing scenes
        os.makedirs(output_root, exist_ok=True)
        process_scenes = {seq: [] for seq in available_scenes}
        for scene in os.listdir(rendered_root):
            seq = scene.split("_")[0]
            process_scenes[seq].append(scene)
        for seq in available_scenes:
            process_scenes[seq] = sorted(process_scenes[seq])[::PROCESS_INTERVAL]
        for scene_list in process_scenes.values():
            for scene in scene_list:
                src_dir = os.path.join(rendered_root, scene)
                dst_dir = os.path.join(output_root, scene)
                shutil.copytree(src_dir, dst_dir)

        # resize to 1024x576
        file_list = glob.glob(os.path.join(output_root, "*/warped/*.png"))
        with Pool(processes=os.cpu_count()) as pool:
            resize_image_1024x576 = partial(resize_image, width=1024, height=576)
            results = list(tqdm(pool.imap_unordered(resize_image_1024x576, file_list), total=len(file_list), desc="Resizing images to 1024x576"))
            errors = [r for r in results if r[1] != "Success"]
            if errors:
                print("\n--- The following files failed to process: ---")
                for filepath, error_message in errors:
                    print(f"- {filepath}\n  {error_message}")

    else:
        assert os.path.isdir(rendered_root)
        assert os.path.isdir(output_root)

    # identify scenes to process
    total_scene_num = 0
    scenes_to_process = []
    for scene in sorted(os.listdir(output_root)):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue
        assert os.path.isdir(os.path.join(scene_path, "warped"))
        total_scene_num += 1
        if os.path.isdir(os.path.join(scene_path, "generated")):
            continue
        scenes_to_process.append(scene)

    print(f"Total number of tasks to run: {len(scenes_to_process)}/{total_scene_num}")

    try:
        # 3. Generation
        job_idx_for_gpu_assignment = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
            future_to_task_info = {}
            for scene in scenes_to_process:
                gpu_id_for_task = GPUS[job_idx_for_gpu_assignment % len(GPUS)]
                future = executor.submit(run_generation_task, output_root, scene, gpu_id_for_task, args.method)
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

        # check if all scenes were processed
        for scene in os.listdir(output_root):
            scene_path = os.path.join(output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            assert os.path.isdir(os.path.join(scene_path, "warped"))
            assert os.path.isdir(os.path.join(scene_path, "generated")), f"Incomplete! {scene}"

        # resize generated images back to 518x518
        file_list = glob.glob(os.path.join(output_root, "*/generated/*.png"))
        with Pool(processes=os.cpu_count()) as pool:
            resize_image_518x518 = partial(resize_image, width=518, height=518)
            results = list(tqdm(pool.imap_unordered(resize_image_518x518, file_list), total=len(file_list), desc="Resizing images to 518x518"))
            errors = [r for r in results if r[1] != "Success"]
            if errors:
                print("\n--- The following files failed to process: ---")
                for filepath, error_message in errors:
                    print(f"- {filepath}\n  {error_message}")

        # resize generated videos back to 518x518
        file_list = glob.glob(os.path.join(output_root, "*/generated/generated.mp4"))
        with Pool(processes=os.cpu_count()) as pool:
            resize_video_518x518 = partial(resize_video, width=518, height=518)
            results = list(tqdm(pool.imap_unordered(resize_video_518x518, file_list), total=len(file_list), desc="Resizing videos to 518x518"))
            errors = [r for r in results if r[1] != "Success"]
            if errors:
                print("\n--- The following files failed to process: ---")
                for filepath, error_message in errors:
                    print(f"- {filepath}\n  {error_message}")

        # resize warped images back to 518x518
        for scene in os.listdir(output_root):
            if not os.path.isdir(os.path.join(output_root, scene)):
                continue
            warped_ori = os.path.join(rendered_root, scene, "warped")
            warped_resized = os.path.join(output_root, scene, "warped")
            shutil.copytree(warped_ori, warped_resized, dirs_exist_ok=True)

        # 4. Pixelwise metrics calculation
        pixelwise_results, _ = run_pixelwise_metrics_calculation(output_root)
        with open(os.path.join(output_root, "pixelwise_results.txt"), "w") as f:
            json.dump(pixelwise_results, f, indent=4)

        # 4.5. Pixelwise metrics calculation with the whole target frame
        target_pixelwise_results, _ = last_frame_comparison(output_root)
        with open(os.path.join(output_root, "target_frame_pixelwise_results.txt"), "w") as f:
            json.dump(target_pixelwise_results, f, indent=4)

        # 6. Camera pose error calculation
        camera_pose_results, _ = run_camera_pose_error_calculation(output_root)
        with open(os.path.join(output_root, "camera_pose_results.txt"), "w") as f:
            json.dump(camera_pose_results, f, indent=4)

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
