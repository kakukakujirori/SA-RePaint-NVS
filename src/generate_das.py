import argparse
import gc
import os
import sys
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.utils import export_to_video
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from gpu_memory_monitor import GPUMemoryMonitor

sys.path.append(__file__.rsplit('/', 2)[0])  # Adjust path to include the parent directory
sys.path.append(os.path.join(__file__.rsplit('/', 2)[0], 'tools', 'DiffusionAsShader'))  # Adjust path to include the DiffusionAsShader
from tools.DiffusionAsShader.models.pipelines import DiffusionAsShaderPipeline


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
        default=50
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Diffusion-As-Shader",
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

    # monitor = GPUMemoryMonitor(gpu_id=args.gpu)
    # monitor.start()

    # load the pipeline
    pipeline = DiffusionAsShaderPipeline(gpu_id=args.gpu, output_dir=args.output_folder)

    # load the captionar
    blip_path = "Salesforce/blip2-opt-2.7b"
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    ).to(device)

    # load images
    first_frame = PIL.Image.open(os.path.join(args.trajectory_folder, f"0000.png"))
    tracking_tensor = torch.from_numpy(np.load(os.path.join(args.trajectory_folder, "trans_coordinates_rgb.npy"))).to(device)

    # get caption
    captioner_inputs = caption_processor(images=first_frame, return_tensors="pt").to(device, torch.float16)
    generated_ids = captioner.generate(**captioner_inputs)
    generated_text = caption_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    prompt = generated_text + ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    del caption_processor
    del captioner

    # pack to tensors and resize
    height, width = 480, 720
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor()
    ])
    image_tensor = transform(first_frame)
    tracking_tensor = F.interpolate(tracking_tensor, size=(height, width), mode="bilinear", align_corners=False)

    # expand frames to 49 (DaS requires 49 frames, otherwise the results are terrible)
    assert args.num_frames <= 49, "The number of frames must be at most 49"
    assert tracking_tensor.shape == (args.num_frames, 3, height, width), f"{tracking_tensor.shape=}"
    start_frame = (49 - args.num_frames) // 2
    end_frame = start_frame + args.num_frames
    tracking_tensor = torch.cat([
        tracking_tensor[0:1].expand(start_frame, -1, -1, -1),
        tracking_tensor,
        tracking_tensor[-1:].expand(49 - end_frame, -1, -1, -1)
    ], dim=0)

    # inference
    gc.collect()
    torch.cuda.empty_cache()
    frames = pipeline._infer(
        prompt=prompt,
        model_path=args.checkpoint,
        tracking_tensor=tracking_tensor,
        image_tensor=image_tensor,
        output_path=None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        dtype=torch.bfloat16,
        fps=-1,
        seed=args.seed,
    )

    # monitor.stop()
    # print(f"Peak GPU memory usage: {monitor.get_max_memory():.2f} GB")

    # extract the middle args.num_frames
    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames[start_frame:end_frame]):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
