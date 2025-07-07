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

import lpips
import numpy as np
import torch
import torch_fidelity
from diffusers.utils import load_image, load_video
from torchmetrics.image import PeakSignalNoiseRatio

from src.eval_trajectories import eval_trajectories
from src.eval_sed import eval_sed


mannequin_challenge_input_root = "./mannequin_challenge_input"
mannequin_challenge_output_root = "./mannequin_challenge_output"
NUM_FRAMES = 25
NUM_INFERECE_STEPS = 50
DENOISE_START_STEP = NUM_INFERECE_STEPS // 3
REPAINT_ITER_NUM = 2
MOTION_MODES = ["horizontal", "vertical", "zoomout"]
DEGREE_LIST = [-1.0, -0.5, 0.5, 1.0]
MOTION_DEGREE_PAIRS = [x for x in product(MOTION_MODES, DEGREE_LIST) if x not in [('vertical', -1.0), ('vertical', 1.0)]]
NUM_GPUS = 4


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


def organize_images_and_depth(mannequin_challenge_data_root: str):
    if os.path.isdir(mannequin_challenge_input_root):
        shutil.rmtree(mannequin_challenge_input_root)
    if os.path.isdir(mannequin_challenge_output_root):
        shutil.rmtree(mannequin_challenge_output_root)
    os.makedirs(mannequin_challenge_input_root)
    os.makedirs(mannequin_challenge_output_root)

    # extract keyframes from each scene
    for i, scene in enumerate(glob.glob(os.path.join(mannequin_challenge_data_root, "*"))):
        if not os.path.isdir(scene):
            continue

        scene_name = os.path.basename(scene)
        print(scene_name)

        for cnt, imgpath in enumerate(sorted(glob.glob(os.path.join(scene, "*.jpg")))):
            # pool images
            if cnt % NUM_FRAMES == 0:
                img_num = os.path.basename(imgpath).split(".")[0]
                dst_path = os.path.join(mannequin_challenge_input_root, scene_name + "_" + img_num + ".jpg")
                shutil.copy(imgpath, dst_path)

    # resize to 1024x576
    subprocess.run(["magick", "mogrify", "-resize", "1024x576!", os.path.join(mannequin_challenge_input_root, "*.jpg")])

    # depth estimation
    imglist = glob.glob(os.path.join(mannequin_challenge_input_root, "*.jpg"))
    imglist_path = os.path.join(mannequin_challenge_input_root, "tmp.txt")
    with open(imglist_path, "a") as f:
        for imgpath in imglist:
            f.write(imgpath + "\n")

    subprocess.run(["python", "tools/Depth-Anything-V2/run.py",
        "--encoder", "vitl",
        "--img-path", imglist_path,
        "--outdir", mannequin_challenge_output_root])  # TEMPORAL USE

    os.remove(imglist_path)

    # store in each folder
    for imgpath in glob.glob(os.path.join(mannequin_challenge_input_root, "*.jpg")):
        imgname = os.path.basename(imgpath).split(".")[0]
        image_folder_path = os.path.join(mannequin_challenge_input_root, imgname, "images")
        depth_folder_path = os.path.join(mannequin_challenge_input_root, imgname, "depth")
        os.makedirs(image_folder_path)
        os.makedirs(depth_folder_path)
        shutil.move(imgpath, image_folder_path)
        shutil.move(os.path.join(mannequin_challenge_output_root, imgname + ".png"), depth_folder_path)
        shutil.move(os.path.join(mannequin_challenge_output_root, imgname + ".npy"), depth_folder_path)


def run_trajectory_extraction(scene: str, motion_mode: str, degree: float, no_occlusion_revealing: bool = True, save_trajectory: bool = False) -> str:
    task_id = f"Scene: {scene}, Motion: {motion_mode + '_' + str(degree)}"
    print(f"STARTING task: {task_id}")
    try:
        result = subprocess.run(["python", "src/trajectory_extraction.py",
            "--image_folder", f"{mannequin_challenge_input_root}/{scene}/images/",
            "--depth_folder", f"{mannequin_challenge_input_root}/{scene}/depth/",
            "--output_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/warped",
            "--degrees_per_frame", f"{degree}",
            "--camera_motion_mode", f"{motion_mode}",
            "--major_radius", "80",
            "--minor_radius", "70",
            "--num_frames", f"{NUM_FRAMES}",
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
                "--output_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "100",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "trajattn":
            result = subprocess.run(["python", "src/generate_trajattn.py",
                "--output_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/warped",
                "--num_frames", f"{NUM_FRAMES}",
                "--num_inference_steps", "25",
                "--min_guidance_scale", "1.0",
                "--max_guidance_scale", "3.0",
                "--seed", "12345",
                "--gpu", f"{gpu_id}"],
                check=True, capture_output=True, text=True, encoding='utf-8')
        elif method == "mine":
            result = subprocess.run(["python", "src/generate.py",
                "--output_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/generated",
                "--trajectory_folder", f"{mannequin_challenge_output_root}/{scene}/{motion_mode}_{degree}/warped",
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


def run_pixelwise_metrics_calculation(mannequin_challenge_output_root: str):
    PSNR_MODULES = [PeakSignalNoiseRatio(data_range=1.0).eval().to(f"cuda:{i}") for i in range(NUM_GPUS)]
    LPIPS_MODULES = [lpips.LPIPS(net='alex', spatial=True).eval().to(f"cuda:{i}") for i in range(NUM_GPUS)]

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

        # binarize the mask
        mask_tensor_bool = mask_tensor < 0.5
        mask_tensor_float = mask_tensor_bool.float().mean(dim=1, keepdim=True)

        # calculate psnr, lpips (NOTE: image range is [-1, 1] for LPIPS)
        results = {}
        with torch.no_grad():
            psnr_score = PSNR(warped_tensor[mask_tensor_bool], generated_tensor[mask_tensor_bool])
            results["psnr"] = psnr_score.item()

            lpips_full = LPIPS(warped_tensor * 2 - 1, generated_tensor * 2 - 1)
            lpips_score = torch.sum(lpips_full * mask_tensor_float) / torch.sum(mask_tensor_float)
            results["lpips"] = lpips_score.item()

        total_results[data_dir] = results


    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        future_to_task_info = {}
        for idx, scene in enumerate(sorted(os.listdir(mannequin_challenge_output_root))):
            scene_path = os.path.join(mannequin_challenge_output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                data_dir = os.path.join(mannequin_challenge_output_root, scene, motion_degree)
                assert os.path.isdir(data_dir)

                if not os.path.isdir(os.path.join(data_dir, "generated")):
                    print(f"Missing {os.path.join(data_dir, 'generated')}")
                    missing.append(data_dir)
                    continue

                future = executor.submit(run_task, data_dir, idx % NUM_GPUS)
                future_to_task_info[future] = (data_dir, idx % NUM_GPUS)

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(future_to_task_info), desc="Calculating pixelwise metrics"):
            data_dir, gpu_id = future_to_task_info[future]
            task_desc = f"{data_dir=}, {gpu_id=}"
            try:
                result_message = future.result() # This will re-raise exceptions from run_generation_task
                # print(f"Result for {task_desc}: {result_message}")
            except Exception as exc: # Should be caught by try/except in run_generation_task but good to have a fallback here.
                print(f"Main loop caught exception for {task_desc}: {exc}")
                raise exc


    total_psnr_mean = sum([result["psnr"] for result in total_results.values()]) / len(total_results)
    total_lpips_mean = sum([result["lpips"] for result in total_results.values()]) / len(total_results)
    print(f"Total PSNR Mean: {total_psnr_mean}")
    print(f"Total LPIPS Mean: {total_lpips_mean}")
    print(f"Missing dirs: {len(missing)}")
    total_results["total_psnr_mean"] = total_psnr_mean
    total_results["total_lpips_mean"] = total_lpips_mean
    total_results["missing_dirs"] = missing

    return total_results, missing


def run_fid_calculation(
        mannequin_challenge_data_root: str,
        mannequin_challenge_output_root: str,
    ):
    with tempfile.TemporaryDirectory() as td:
        print(f"[run_fid_calculation] Images are copied to {td} for FID calculation.")
        gt_folder = os.path.join(td, "gt")
        generated_folder = os.path.join(td, "generated")
        os.mkdir(gt_folder)
        os.mkdir(generated_folder)

        # Copy GT images
        for scene in os.listdir(mannequin_challenge_data_root):
            for imgpath in glob.glob(os.path.join(mannequin_challenge_data_root, scene, "*.jpg")):
                imgpath = os.path.normpath(imgpath)
                shutil.copy(imgpath, os.path.join(gt_folder, imgpath.replace("/", "_")))
        print(f"[run_fid_calculation] Number of GT images: {len(os.listdir(gt_folder))}")

        # Copy generated images
        for scene in os.listdir(mannequin_challenge_output_root):
            scene_path = os.path.join(mannequin_challenge_output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                for imgpath in glob.glob(os.path.join(mannequin_challenge_output_root, scene, motion_degree, "generated", "*.png")):
                    imgpath = os.path.normpath(imgpath)  # delete the leading ./
                    dst_path = os.path.join(generated_folder, imgpath.replace("/", "_"))
                    shutil.copy(imgpath, dst_path)
        print(f"[run_fid_calculation] Number of generated images: {len(os.listdir(generated_folder))}")

        # Resize images to 1024x576
        subprocess.run(f"ls {os.path.join(gt_folder, '*.jpg')} | xargs -n1 -P32 mogrify -resize 1024x576!", shell=True)

        # calculate FID
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=gt_folder,
            input2=generated_folder,
            cuda=True,
            fid=True,
            verbose=False,
        )
        fid = metrics_dict.get("frechet_inception_distance", None)
        print(f"FID: {fid}")
        return fid


def run_camera_pose_error_calculation(mannequin_challenge_output_root: str, gt_focal_len: float = 260.0):
    ANYCAM_MODULES = [torch.hub.load('Brummi/anycam', 'AnyCam', version="1.0", training_variant="seq8", pretrained=True).eval().to(f"cuda:{i}") for i in range(1)]

    total_results = {}
    missing_videos = []

    def run_task(data_dir: str, gpu_id: int):
        AnyCam = ANYCAM_MODULES[gpu_id]
        frames = load_video(os.path.join(data_dir, "generated/generated.mp4"))
        frames = [np.array(x).astype(np.float32) / 255.0 for x in frames]
        height, width, _ = frames[0].shape
        gt_poses_w2c = [np.load(os.path.join(data_dir, f"warped/{i:04d}_pose.npy")) for i in range(len(frames))]
        gt_poses_c2w = [np.linalg.inv(p) for p in gt_poses_w2c]
        gt_K = np.array([
            [gt_focal_len, 0, width/2],
            [0, gt_focal_len, height/2],
            [0, 0, 1],
        ], dtype=np.float32)

        # Estimate the camera poses and intrinsics from the generated video
        anycam_ret = AnyCam.process_video(frames, ba_refinement=True)
        estimated_poses_c2w = [p.cpu().numpy() for p in anycam_ret["trajectory"]]  # Camera poses
        estimated_K = anycam_ret["projection_matrix"].cpu().numpy()  # Camera intrinsics

        # compute errors
        results, estimated_abs_poses_c2w, gt_abs_poses_c2w = eval_trajectories(estimated_poses_c2w, gt_poses_c2w, estimated_K, gt_K)
        total_results[data_dir] = results


    # NOTE: AnyCam somehow doesn't support multi-GPU inference, so we use only one GPU.
    for idx, scene in enumerate(sorted(os.listdir(mannequin_challenge_output_root))):
        scene_path = os.path.join(mannequin_challenge_output_root, scene)
        if not os.path.isdir(scene_path):
            continue
        for motion_degree in os.listdir(scene_path):
            data_dir = os.path.join(mannequin_challenge_output_root, scene, motion_degree)
            assert os.path.isdir(data_dir)

            # Load the video frames and ground truth poses
            if not os.path.isfile(os.path.join(data_dir, "generated/generated.mp4")):
                print(f"Missing video for {data_dir}")
                missing_videos.append(data_dir)
                continue

            run_task(data_dir, 0)

    total_ape_mean = sum([result["ape_mean"] for result in total_results.values()]) / len(total_results)
    total_rre_mean = sum([result["rre_mean"] for result in total_results.values()]) / len(total_results)
    total_rte_mean = sum([result["rte_mean"] for result in total_results.values()]) / len(total_results)
    print(f"Total APE Mean: {total_ape_mean}")
    print(f"Total RRE Mean: {total_rre_mean}")
    print(f"Total RTE Mean: {total_rte_mean}")
    print(f"Missing videos: {len(missing_videos)}")
    total_results["total_ape_mean"] = total_ape_mean
    total_results["total_rre_mean"] = total_rre_mean
    total_results["total_rte_mean"] = total_rte_mean
    total_results["missing_videos"] = missing_videos

    return total_results, missing_videos


def run_sed_calculation(mannequin_challenge_output_root: str):
    missing = []
    total_results = {}
    for scene in tqdm(os.listdir(mannequin_challenge_output_root), desc="Calculating SED"):
        scene_path = os.path.join(mannequin_challenge_output_root, scene)
        if not os.path.isdir(scene_path):
            continue
        for motion_degree in os.listdir(scene_path):
            data_dir = os.path.join(mannequin_challenge_output_root, scene, motion_degree)
            assert os.path.isdir(data_dir)
            video_path = os.path.join(data_dir, "generated/generated.mp4")
            camera_paths = os.path.join(data_dir, "warped/*_pose.npy")
            poses = [np.load(p) for p in sorted(glob.glob(camera_paths))]

            if not os.path.isfile(video_path):
                missing.append(data_dir)
                continue

            with tempfile.TemporaryDirectory() as colmap_root:
                consistent_ratios, sed_summary = eval_sed(
                    colmap_root=colmap_root,
                    video_path=video_path,
                    poses=poses,
                    gt_focal_len=260.0,
                    save_sed_graph_to=None,
                )
                total_results[data_dir] = consistent_ratios

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
    parser = argparse.ArgumentParser(description="Mannequin Challenge Evaluation")
    parser.add_argument("--data_root", type=str, default="/home/ryotaro/data/MannequinChallengeHQ/validation_frames")
    parser.add_argument("--scratch", action="store_true", help="If set, all the images, depth, and trajectories are re-organized and re-generated.")
    parser.add_argument("--method", type=str, default="mine", choices=["nvssolver", "trajattn", "mine"], help="Method to use for generation. 'nvssolver' uses NVS-Solver, 'trajattn' uses Trajectory Attention, and 'mine' uses the custom method.")
    args = parser.parse_args()

    # 1. Organize RGB images & Depth estimation
    if args.scratch:
        # reorganize_frames(mannequin_challenge_data_root=os.path.dirname(args.data_root))
        organize_images_and_depth(mannequin_challenge_data_root=args.data_root)

        scene_motion_degree_pairs = []
        for i, scene in enumerate(sorted(os.listdir(mannequin_challenge_input_root))):
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
                    result_message = future.result() # This will re-raise exceptions from run_generation_task
                    # print(f"Result for {task_desc}: {result_message}")
                except Exception as exc: # Should be caught by try/except in run_generation_task but good to have a fallback here.
                    print(f"Main loop caught exception for {task_desc}: {exc}")

    else:
        assert os.path.isdir(mannequin_challenge_output_root)

        scene_motion_degree_pairs = []
        for scene in sorted(os.listdir(mannequin_challenge_output_root)):
            scene_path = os.path.join(mannequin_challenge_output_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                assert os.path.isdir(os.path.join(mannequin_challenge_output_root, scene, motion_degree, "warped"))
                if os.path.isdir(os.path.join(mannequin_challenge_output_root, scene, motion_degree, "generated")):
                    continue
                motion_mode, degree = motion_degree.split("_")
                scene_motion_degree_pairs.append((scene, motion_mode, degree))

    print(f"Total number of tasks to run: {len(scene_motion_degree_pairs)}")

    try:
        # 3. Generation
        job_idx_for_gpu_assignment = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            future_to_task_info = {}
            for scene, motion, degree in scene_motion_degree_pairs:
                gpu_id_for_task = job_idx_for_gpu_assignment % NUM_GPUS
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
        pixelwise_results, _ = run_pixelwise_metrics_calculation(mannequin_challenge_output_root)
        with open(os.path.join(mannequin_challenge_output_root, "pixelwise_results.txt"), "w") as f:
            json.dump(pixelwise_results, f, indent=4)

        # 5. FID calculation
        fid = run_fid_calculation(
            mannequin_challenge_data_root=args.data_root,
            mannequin_challenge_output_root=mannequin_challenge_output_root,
        )
        with open(os.path.join(mannequin_challenge_output_root, "fid.txt"), "w") as f:
            f.write(str(fid))

        # 6. Camera pose error calculation
        camera_pose_results, _ = run_camera_pose_error_calculation(mannequin_challenge_output_root)
        with open(os.path.join(mannequin_challenge_output_root, "camera_pose_results.txt"), "w") as f:
            json.dump(camera_pose_results, f, indent=4)

        # 7. SED calculation
        sed_results = run_sed_calculation(mannequin_challenge_output_root)
        with open(os.path.join(mannequin_challenge_output_root, "sed.txt"), "w") as f:
            json.dump(sed_results, f, indent=4)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, shutting down.")
    finally:
        print("Program finished.")
