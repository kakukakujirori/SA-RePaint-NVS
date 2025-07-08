import argparse
import gc
import os
import sys
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.utils import export_to_video
from transformers import AutoProcessor, Blip2ForConditionalGeneration, T5EncoderModel

sys.path.append(__file__.rsplit('/', 2)[0])  # Adjust path to include the parent directory
sys.path.append(os.path.join(__file__.rsplit('/', 2)[0], 'tools', 'TrajectoryCrafter'))  # Adjust path to include the TrajectoryCrafter
from tools.TrajectoryCrafter.models.crosstransformer3d import CrossTransformer3DModel
from tools.TrajectoryCrafter.models.autoencoder_magvit import AutoencoderKLCogVideoX
from tools.TrajectoryCrafter.models.pipeline_trajectorycrafter import TrajCrafter_Pipeline


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
        "--gpu",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # load the pipeline
    model_name = 'alibaba-pai/CogVideoX-Fun-V1.1-5b-InP'
    weight_dtype = torch.bfloat16
    pipeline = TrajCrafter_Pipeline.from_pretrained(
        model_name,
        vae=AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder="vae").to(weight_dtype),
        text_encoder=T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype),
        transformer=CrossTransformer3DModel.from_pretrained("TrajectoryCrafter/TrajectoryCrafter").to(weight_dtype),
        scheduler=DDIMScheduler.from_pretrained(model_name, subfolder="scheduler"),
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload(device=device)

    # load the captionar
    blip_path = "Salesforce/blip2-opt-2.7b"
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    ).to(device)

    # load images
    warped_images = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}.png")) for i in range(args.num_frames)]
    warped_masks = [PIL.Image.open(os.path.join(args.trajectory_folder, f"{i:04d}_mask.png")) for i in range(args.num_frames)]

    # get caption
    captioner_inputs = caption_processor(images=warped_images[0], return_tensors="pt").to(device, torch.float16)
    generated_ids = captioner.generate(**captioner_inputs)
    generated_text = caption_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    prompt = generated_text + ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    del caption_processor
    del captioner

    # pack to tensors and resize
    height, width = 384, 672
    cond_video = torch.stack([torch.from_numpy(np.array(x)).permute(2,0,1).float() / 255.0 for x in warped_images])
    cond_masks = torch.stack([torch.from_numpy(np.array(x)).permute(2,0,1).float() / 255.0 for x in warped_masks])
    cond_masks = cond_masks.mean(dim=1, keepdim=True)
    cond_video = F.interpolate(cond_video, size=(height, width), mode='bilinear', align_corners=False)
    cond_masks = F.interpolate(cond_masks, size=(height, width), mode='nearest')
    cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
    cond_masks = cond_masks.permute(1, 0, 2, 3).unsqueeze(0) * 255.0

    # inference
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        frames = pipeline(
            prompt,
            num_frames=args.num_frames,
            negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
            height=height,
            width=width,
            generator=torch.manual_seed(args.seed),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            video=cond_video,
            mask_video=cond_masks,
            reference=cond_video[:, :, 0:1, :, :].expand(-1, -1, 10, -1, -1),
        ).videos[0].permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C)

    frames = [PIL.Image.fromarray(np.clip(fr * 255, 0, 255).astype(np.uint8)) for fr in frames]  # Convert to PIL images

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(frames):
        fr.save(os.path.join(args.output_folder, f"{i:04d}.png"))
    export_to_video(frames, os.path.join(args.output_folder, "generated.mp4"))
