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
from torchmetrics.image import PeakSignalNoiseRatio

from src.eval_sed import eval_sed
from src.eval_ssim import ssim
from src.eval_trajectories import eval_trajectories, run_glomap


davis_input_root = "./davis_video_input"
davis_output_root = "./davis_video_output"
NUM_FRAMES = 25
NUM_INFERECE_STEPS = 50
DENOISE_START_STEP = NUM_INFERECE_STEPS // 3
REPAINT_ITER_NUM = 2
MOTION_MODES = ["horizontal", "vertical", "zoomout"]
DEGREE_LIST = [-0.5, -0.25, 0.25, 0.5]
MOTION_DEGREE_PAIRS = [x for x in product(MOTION_MODES, DEGREE_LIST) if x not in [('vertical', -0.5), ('vertical', 0.5)]]
MAJOR_RADIUS = 80
MINOR_RADIUS = 70
GPUS = [0, 1]


def organize_videos_and_depth(davis_data_root: str, davis_input_root: str, davis_output_root: str):
    assert os.path.isdir(davis_data_root), f"Folder not found: {davis_data_root}"
    if os.path.isdir(davis_input_root):
        shutil.rmtree(davis_input_root)
    if os.path.isdir(davis_output_root):
        shutil.rmtree(davis_output_root)
    os.makedirs(davis_input_root)
    os.makedirs(davis_output_root)

    # define tasks
    def to_chunk(chunk_img_paths: list[str], outdir: str):
        with tempfile.TemporaryDirectory() as td:
            # copy images
            for idx, imgpath in enumerate(chunk_img_paths):
                dst_img = os.path.join(td, f"{idx:04d}.jpg")
                shutil.copy(imgpath, dst_img)

            # resize images
            subprocess.run(["magick", "mogrify", "-resize", "1024x576!", os.path.join(td, "*.jpg")])

            # to mkv (= lossless video format)
            try:
                scene_name = os.path.basename(os.path.dirname(chunk_img_paths[0])).split(".")[0]
                start_img_num = os.path.basename(chunk_img_paths[0]).split(".")[0]
                dst_mkv = os.path.join(outdir, f"{scene_name}_{start_img_num}.mkv")
                result = subprocess.run([
                    "ffmpeg", "-y",
                    "-framerate", "10",
                    "-i", os.path.join(td, "%04d.jpg"),
                    "-c:v", "ffv1",
                    "-g", "1",
                    dst_mkv,
                ], check=True, capture_output=True, text=True, encoding='utf-8')
            except subprocess.CalledProcessError as e:
                error_message = (
                    f"ERROR!!!\n"
                    f"Command: {' '.join(e.cmd)}\n"
                    f"Return code: {e.returncode}\n"
                    f"Stdout:\n{e.stdout.strip()}\n"
                    f"Stderr:\n{e.stderr.strip()}"
                )
                print(error_message)
                raise e

    # separate frames in chunks from each scene
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_to_task_info = {}
        for scene in glob.glob(os.path.join(davis_data_root, "*")):
            if not os.path.isdir(scene):
                continue

            imgpaths = sorted(glob.glob(os.path.join(scene, "*.jpg")))
            num_chunks = len(imgpaths) // NUM_FRAMES

            for chunk_idx in range(num_chunks):
                chunk_imgs = imgpaths[chunk_idx * NUM_FRAMES : (chunk_idx + 1) * NUM_FRAMES]
                if not chunk_imgs:
                    continue
                future = executor.submit(to_chunk, chunk_imgs, davis_input_root)
                future_to_task_info[future] = (chunk_imgs, davis_input_root)

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Separating videos to chunks"):
            chunk_imgs, davis_input_root = future_to_task_info[future]
            task_desc = f"{chunk_imgs=}, {davis_input_root=}"
            try:
                _ = future.result()
            except Exception as exc:
                print(f"Main loop caught exception for {task_desc}: {exc}")
                raise exc

    # depth estimation
    video_path_list = glob.glob(os.path.join(davis_input_root, "*.mkv"))
    task_per_GPU = (len(video_path_list) + len(GPUS) - 1) // len(GPUS)
    processes = []
    for i, gpu_id in enumerate(GPUS):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        video_path_per_GPU = video_path_list[task_per_GPU * i : task_per_GPU * (i + 1)]
        vidlist_path = os.path.join(davis_input_root, f"tmp_{gpu_id}.txt")
        with open(vidlist_path, "w") as f:
            for imgpath in video_path_per_GPU:
                f.write(imgpath + "\n")

        proc = subprocess.Popen([
                "python", "tools/Video-Depth-Anything/run.py",
                "--encoder", "vitl",
                "--input_video", vidlist_path,
                "--output_dir", davis_output_root,
                "--save_npz",
            ], stdout=None, stderr=None, text=True, encoding='utf-8', env=env)
        processes.append((proc, gpu_id, vidlist_path))

    # wait for all the process to finish
    for proc, gpu_id, vidlist_path in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"[GPU {gpu_id}] ERROR: returncode={proc.returncode}")
            print(f"[GPU {gpu_id}] STDOUT:\n{stdout}")
            print(f"[GPU {gpu_id}] STDERR:\n{stderr}")
        else:
            print(f"[GPU {gpu_id}] Finished successfully.")

        os.remove(vidlist_path)

    # store in each folder
    for vidpath in glob.glob(os.path.join(davis_input_root, "*.mkv")):
        vidname = os.path.basename(vidpath).split(".")[0]
        image_folder_path = os.path.join(davis_input_root, vidname, "images")
        depth_folder_path = os.path.join(davis_input_root, vidname, "depth")
        os.makedirs(image_folder_path)
        os.makedirs(depth_folder_path)
        shutil.move(os.path.join(davis_output_root, vidname + "_depths.npz"), depth_folder_path)
        subprocess.run([
            "ffmpeg", "-i", vidpath,
            "-start_number", "0",
            os.path.join(image_folder_path, "%04d.png"),
        ], check=True, capture_output=True, text=True, encoding='utf-8')
        os.remove(vidpath)
        os.remove(os.path.join(davis_output_root, vidname + "_src.mp4"))
        os.remove(os.path.join(davis_output_root, vidname + "_vis.mp4"))


def run_trajectory_extraction(scene: str, motion_mode: str, degree: float, no_occlusion_revealing: bool = True, save_trajectory: bool = False):
    task_id = f"Scene: {scene}, Motion: {motion_mode + '_' + str(degree)}"
    print(f"STARTING task: {task_id}")
    try:
        result = subprocess.run(["python", "src/trajectory_extraction.py",
            "--image_folder", f"{davis_input_root}/{scene}/images/",
            "--depth_folder", f"{davis_input_root}/{scene}/depth/",
            "--output_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/warped",
            "--depth_format", "npz",
            "--invert_depth",
            "--focal_len", "260",
            "--degrees_per_frame", f"{degree}",
            "--camera_motion_mode", f"{motion_mode}",
            "--major_radius", f"{MAJOR_RADIUS}",
            "--minor_radius", f"{MINOR_RADIUS}",
            "--num_frames", f"{NUM_FRAMES}",
            "--control_mode", "video",
            ] + (
                ["--no_occlusion_revealing"] if no_occlusion_revealing else []
            ) + (
                ["--save_trajectory"] if save_trajectory else []
            ),
            check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"COMPLETED task: {task_id}\nSTDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"STDERR for {task_id}:\n{result.stderr.strip()}")
        return f"Successfully processed {task_id}"
    except subprocess.CalledProcessError as e:
        error_message = (
            f"ERROR for task: {task_id}\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"Stdout:\n{e.stdout.strip()}\n"
            f"Stderr:\n{e.stderr.strip()}"
        )
        print(error_message)
        return f"Failed processing {task_id}: {e.returncode}"
    except Exception as e:
        print(f"UNEXPECTED ERROR for task: {task_id}: {e}")
        return f"Unexpected error for {task_id}: {e}"


def run_generation_task(scene: str, motion_mode: str, degree: float, gpu_id: int, method: str = "mine") -> str:
    task_id = f"Scene: {scene}, Motion: {motion_mode + '_' + str(degree)}, GPU: {gpu_id}"
    print(f"STARTING task: {task_id}")
    with torch.cuda.device(f'cuda:{gpu_id}'):
        torch.cuda.empty_cache()
    try:
        if method == "nvssolver":
            result = subprocess.run(["python", "src/generate_nvssolver.py",
                "--output_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "100",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "trajattn":
            result = subprocess.run(["python", "src/generate_trajattn.py",
                "--output_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "25",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "trajcrafter":
            result = subprocess.run(["python", "src/generate_trajcrafter.py",
                "--output_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "50",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "mine":
            result = subprocess.run(["python", "src/generate.py",
                "--output_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{davis_output_root}/{scene}/{motion_mode}_{degree}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", f"{NUM_INFERECE_STEPS}",
                "--denoise_start_step", f"{DENOISE_START_STEP}",
                "--repaint_iter_num", f"{REPAINT_ITER_NUM}",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
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
            for motion_degree in os.listdir(scene_path):
                data_dir = os.path.join(output_root, scene, motion_degree)
                assert os.path.isdir(data_dir)

                if not os.path.isdir(os.path.join(data_dir, "generated")):
                    print(f"Missing {os.path.join(data_dir, 'generated')}")
                    missing.append(data_dir)
                    continue

                future = executor.submit(run_task, data_dir, GPUS[idx % len(GPUS)])
                future_to_task_info[future] = (data_dir, GPUS[idx % len(GPUS)])

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Calculating pixelwise metrics"):
            data_dir, gpu_id = future_to_task_info[future]
            task_desc = f"{data_dir=}, {gpu_id=}"
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


def run_camera_pose_error_calculation(output_root: str, gt_focal_len: float = 260.0):

    def run_task(args: tuple[str, int]):
        data_dir, gpu_id = args

        if not os.path.isfile(os.path.join(data_dir, "generated/generated.mp4")):
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
        results, estimated_abs_poses_c2w, gt_abs_poses_c2w = eval_trajectories(estimated_poses_c2w, gt_poses_c2w, estimated_K, gt_K)
        return (data_dir, results)

    tasks = []
    for idx, scene in enumerate(os.listdir(output_root)):
        scene_path = os.path.join(output_root, scene)
        if not os.path.isdir(scene_path):
            continue
        for motion_degree in os.listdir(scene_path):
            data_dir = os.path.join(output_root, scene, motion_degree)
            assert os.path.isdir(data_dir)
            tasks.append((data_dir, GPUS[idx % len(GPUS)]))

    with ProcessingPool(nodes=32) as pool:
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
    print(f"Total APE Mean: {total_ape_mean}")
    print(f"Total RRE Mean: {total_rre_mean}")
    print(f"Total RTE Mean: {total_rte_mean}")
    print(f"Missing videos: {len(missing)}")
    total_results["total_ape_mean"] = total_ape_mean
    total_results["total_rre_mean"] = total_rre_mean
    total_results["total_rte_mean"] = total_rte_mean
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
    parser = argparse.ArgumentParser(description="DAVIS Evaluation")
    parser.add_argument("--data_root", type=str, default="/home/ryotaro/data/DAVIS/JPEGImages/Full-Resolution")
    parser.add_argument("--scratch", action="store_true", help="If set, all the images, depth, and trajectories are re-organized and re-generated.")
    parser.add_argument("--method", type=str, default="mine", choices=["nvssolver", "trajattn", "trajcrafter", "mine"], help="Method to use for generation. 'nvssolver' uses NVS-Solver, 'trajattn' uses Trajectory Attention, and 'mine' uses the custom method.")
    args = parser.parse_args()

    # 1. Organize RGB images & Depth estimation
    if args.scratch:
        organize_videos_and_depth(
            davis_data_root=args.data_root,
            davis_input_root=davis_input_root,
            davis_output_root=davis_output_root,
        )

        scene_motion_degree_pairs = []
        for i, scene in enumerate(sorted(os.listdir(davis_input_root))):
            motion_mode, degree = MOTION_DEGREE_PAIRS[i % len(MOTION_DEGREE_PAIRS)]
            scene_motion_degree_pairs.append((scene, motion_mode, degree))

        # 2. Trajectory Extraction
        job_idx = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=31) as executor:
            future_to_task_info = {}
            for scene, motion, degree in scene_motion_degree_pairs:
                future = executor.submit(run_trajectory_extraction, scene, motion, degree, save_trajectory=(args.method == "trajattn"))
                future_to_task_info[future] = (scene, motion, degree, None)
                job_idx += 1 # This ensures round-robin submission to GPUs

            # Collect results (and handle exceptions)
            for future in concurrent.futures.as_completed(future_to_task_info):
                scene, motion, degree, gpu_id = future_to_task_info[future]
                task_desc = f"Scene: {scene}, Motion: {motion + '_' + str(degree)}, GPU: {gpu_id}"
                try:
                    result_message = future.result()
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc:
                    print(f"Main loop caught exception for {task_desc}: {exc}")

    else:
        assert os.path.isdir(davis_output_root)

        scene_motion_degree_pairs = []
        for scene in sorted(os.listdir(davis_output_root)):
            scene_path = os.path.join(davis_output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                assert os.path.isdir(os.path.join(davis_output_root, scene, motion_degree, "warped"))
                if os.path.isdir(os.path.join(davis_output_root, scene, motion_degree, "generated")):
                    continue
                motion_mode, degree = motion_degree.split("_")
                scene_motion_degree_pairs.append((scene, motion_mode, degree))

    print(f"Total number of tasks to run: {len(scene_motion_degree_pairs)}")

    try:
        # 3. Generation
        job_idx_for_gpu_assignment = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
            future_to_task_info = {}
            for scene, motion, degree in scene_motion_degree_pairs:
                gpu_id_for_task = GPUS[job_idx_for_gpu_assignment % len(GPUS)]
                future = executor.submit(run_generation_task, scene, motion, degree, gpu_id_for_task, args.method)
                future_to_task_info[future] = (scene, motion, degree, gpu_id_for_task)
                job_idx_for_gpu_assignment += 1 # This ensures round-robin submission to GPUs

            # Collect results (and handle exceptions)
            for future in concurrent.futures.as_completed(future_to_task_info):
                scene, motion, degree, gpu_id = future_to_task_info[future]
                task_desc = f"Scene: {scene}, Motion: {motion + '_' + str(degree)}, GPU: {gpu_id}"
                try:
                    result_message = future.result() # This will re-raise exceptions from run_generation_task
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc: # Should be caught by try/except in run_generation_task but good to have a fallback here.
                    print(f"Main loop caught exception for {task_desc}: {exc}")
                    raise exc

        # 4. Pixelwise metrics calculation
        pixelwise_results, _ = run_pixelwise_metrics_calculation(davis_output_root, allow_resize=(args.method == "trajcrafter"))
        with open(os.path.join(davis_output_root, "pixelwise_results.txt"), "w") as f:
            json.dump(pixelwise_results, f, indent=4)

        # 5. FID/FVD calculation
        fid_score = run_fid_calculation(
            data_root=args.data_root,
            output_root=davis_output_root,
        )
        fvd_videogpt, fvd_stylegan = run_fvd_calculation(
            davis_data_root=args.data_root,
            davis_output_root=davis_output_root,
        )
        with open(os.path.join(davis_output_root, "fid_fvd.txt"), "w") as f:
            f.write("FID: " + str(fid_score) + "\nFVD (VideoGPT): " + str(fvd_videogpt) + "\nFVD (StyleGAN): " + str(fvd_stylegan) + "\n")

        # 6. Camera pose error calculation
        camera_pose_results, _ = run_camera_pose_error_calculation(davis_output_root)
        with open(os.path.join(davis_output_root, "camera_pose_results.txt"), "w") as f:
            json.dump(camera_pose_results, f, indent=4)

        # 7. SED calculation
        sed_results = run_sed_calculation(davis_output_root)
        with open(os.path.join(davis_output_root, "sed.txt"), "w") as f:
            json.dump(sed_results, f, indent=4)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, shutting down.")
    finally:
        print("Program finished.")
