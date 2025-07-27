import argparse
import os
import sys
import numpy as np
import torch
from diffusers.utils import load_image, export_to_video

sys.path.append(__file__.rsplit('/', 2)[0])  # Adjust path to include the parent directory
sys.path.append(os.path.join(__file__.rsplit('/', 2)[0], 'tools', 'TrajectoryAttention'))  # Adjust path to include the TrajectoryAttention
from tools.TrajectoryAttention.models.unet import UNetSpatioTemporalConditionModel
from tools.TrajectoryAttention.models.my_svd_pipeline_clean import My_SVD as StableVideoDiffusionPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_folder",
        type=str,
    )

    parser.add_argument(
        "--trajectory_folder",
        type=str,
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25
    )

    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tools/TrajectoryAttention/checkpoints/trajattn_temp.pth",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gpu_memory_limit",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # limit GPU memory
    if args.gpu_memory_limit is not None:
        total_mem_gb = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
        fraction = args.gpu_memory_limit / total_mem_gb
        torch.cuda.set_per_process_memory_fraction(fraction, args.gpu)
        print(f"GPU memory upper limit was set to {args.gpu_memory_limit:.2f}GB ({fraction:.2%})")

    # load the pipeline
    svd_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        svd_path,
        subfolder='unet',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        device_map=None,
        use_safetensors=True,
        using_traj_attn=True,
    )

    it = 0
    param_list = torch.load(args.checkpoint)
    for name, para in unet.named_parameters():
        if 'trajectory' in name:
            para.requires_grad_(False)
            para.copy_(param_list[it])
            it += 1

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        svd_path, torch_dtype=torch.float16, variant="fp16",
        unet=unet,
        use_nvs_solver=False,
    )

    pipeline.enable_model_cpu_offload(device=device)

    # load images
    image = load_image(os.path.join(args.trajectory_folder, "0000.png"))
    trajectory = torch.from_numpy(np.load(os.path.join(args.trajectory_folder, "trans_coordinates.npy"))).reshape(args.num_frames,-1,2)
    masks = torch.from_numpy(np.load(os.path.join(args.trajectory_folder, "trans_valid.npy"))).reshape(args.num_frames,-1)
    pipeline.set_flow_path(
        trajectory=trajectory,
        occ_mask=masks,
        height=576,
        width=1024,
    )

    # inference
    frames = pipeline(
        image,
        temp_cond=None,
        mask=None,
        lambda_ts=None,
        ########
        height=576,
        width=1024,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        min_guidance_scale=args.min_guidance_scale,
        max_guidance_scale=args.max_guidance_scale,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        decode_chunk_size=8,
        generator=torch.manual_seed(args.seed),
    ).frames[0]

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
