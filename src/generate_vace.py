# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import numpy as np
import torch
import PIL.Image
from diffusers import WanVACEPipeline
from diffusers.utils import export_to_video, load_image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from autoencoder_kl_wan import AutoencoderKLWan


def main():
    parser = argparse.ArgumentParser(description="Generate video using Wan2.1 VACE 14B model.")

    parser.add_argument(
        "--output_folder",
        type=str,
        help="Folder to save generated frames and video.",
    )

    parser.add_argument(
        "--trajectory_folder",
        type=str,
        help="Folder containing warped images and masks.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for generation.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use.",
    )

    parser.add_argument(
        "--gpu_memory_limit",
        type=float,
        default=None,
        help="Limit GPU memory in GB.",
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Limit GPU memory if specified
    if args.gpu_memory_limit is not None and torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
        fraction = args.gpu_memory_limit / total_mem_gb
        torch.cuda.set_per_process_memory_fraction(fraction, args.gpu)
        print(f"GPU memory upper limit was set to {args.gpu_memory_limit:.2f}GB ({fraction:.2%})")

    # 1. Load WanVACEPipeline
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

    # FIX: text encoder.encoder.embed_tokens.weight | MISSING
    pipe.text_encoder.encoder.embed_tokens.weight = pipe.text_encoder.shared.weight

    pipe.enable_model_cpu_offload(device=device)

    # 2. Load warped images and masks
    warped_images = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}.png")) for i in range(args.num_frames)]
    warped_masks = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}_mask.png")) for i in range(args.num_frames)]

    # 3. Generate caption for the first frame
    blip_path = "Salesforce/blip2-opt-2.7b"
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    ).to(device)

    inputs = caption_processor(images=warped_images[0], return_tensors="pt").to(device, torch.float16)
    generated_ids = captioner.generate(**inputs)
    generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    prompt = f"{generated_text}. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    print(f"Generated prompt: {prompt}")

    # Cleanup captioner to save memory
    del caption_processor
    del captioner
    torch.cuda.empty_cache()

    # pack to tensors and resize
    height, width = 720, 1280  # NOTE: 1280x720 results in better outcomes than 832x480
    warped_images = [x.resize((width, height), PIL.Image.Resampling.LANCZOS) for x in warped_images]
    warped_masks = [x.resize((width, height), PIL.Image.Resampling.NEAREST) for x in warped_masks]

    # 4. Inference
    frames = pipe(
        video=warped_images,
        mask=warped_masks,
        prompt=prompt,
        negative_prompt="Bright tones, overexposed, blurred details, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, messy background, three legs, many people in the background, walking backwards",
        reference_images=[warped_images[0]], # Use first frame as reference
        height=height,
        width=width,
        num_frames=len(warped_images),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device).manual_seed(args.seed),
        output_type="pil",
    ).frames[0]

    os.makedirs(args.output_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))

if __name__ == "__main__":
    main()
