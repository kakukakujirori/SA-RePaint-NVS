import argparse
import glob
import os
import sys

import numpy as np
import skimage
import torch
import viser
import viser.transforms as tf
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils.export_utils import export_to_video
from PIL import Image
from prior_depth_anything import PriorDepthAnything
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras
from tqdm.auto import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

sys.path.append(__file__.rsplit('/', 2)[0])  # Adjust path to include the parent directory
sys.path.append(os.path.join(__file__.rsplit('/', 2)[0], 'tools', 'invisible_stitch'))  # Adjust path to include the InvisibleStitch
from tools.invisible_stitch.utils.models import get_sd_pipeline
from tools.invisible_stitch.utils.ops import snap_high_gradients_to_nn, get_pointcloud, merge_pointclouds, nearest_neighbor_fill
from tools.invisible_stitch.utils.render import PointsRasterizationSettings, render_with_settings


def render(cameras, point_cloud, fill_point_cloud_holes: bool = False, radius: float | None = None, antialiasing: int = 1):
    if fill_point_cloud_holes:
        coarse_raster_settings = PointsRasterizationSettings(
            image_size=(int(cameras.image_size[0, 0]), int(cameras.image_size[0, 1])),
            radius = 1e-2,
            points_per_pixel = 1
        )

        _, coarse_mask, _ = render_with_settings(cameras, point_cloud, coarse_raster_settings)

        eroded_coarse_mask = torch.from_numpy(skimage.morphology.binary_erosion(coarse_mask[0].cpu().numpy(), footprint=skimage.morphology.disk(2)))

        raster_settings = PointsRasterizationSettings(
            image_size=(int(cameras.image_size[0, 0]), int(cameras.image_size[0, 1])),
            radius = (1 / float(max(cameras.image_size[0, 0], cameras.image_size[0, 1])) * 2.0) if radius is None else radius,
            points_per_pixel = 16
        )

        # Render the scene
        images, masks, depths = render_with_settings(cameras, point_cloud, raster_settings)

        holes_in_rendering = masks[0].cpu() ^ eroded_coarse_mask

        images[0] = nearest_neighbor_fill(images[0], ~holes_in_rendering, 0)
        depths[0] = nearest_neighbor_fill(depths[0], ~holes_in_rendering, 0)

        return images, eroded_coarse_mask.unsqueeze(0).to(masks.device), depths

    else:
        raster_settings = PointsRasterizationSettings(
            image_size=(int(cameras.image_size[0, 0]), int(cameras.image_size[0, 1])),
            radius = (1 / float(max(cameras.image_size[0, 0], cameras.image_size[0, 1])) * 2.0) if radius is None else radius,
            points_per_pixel = 16
        )

        # Render the scene
        return render_with_settings(cameras, point_cloud, raster_settings)


def project_points(cameras, depth, use_pixel_centers=True):
    if len(cameras) > 1:
        import warnings
        warnings.warn("project_points assumes only a single camera is used")

    depth_t = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth
    depth_t = depth_t.to(cameras.device)

    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = cameras.focal_length[0]
    cx, cy = cameras.principal_point[0]

    # cameras.image_size is (height, width)
    i, j = torch.meshgrid(
        torch.arange(cameras.image_size[0][1], dtype=torch.float32, device=cameras.device) + pixel_center,
        torch.arange(cameras.image_size[0][0], dtype=torch.float32, device=cameras.device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth_t / fx, -(j - cy) * depth_t / fy, depth_t], -1
    )

    xy_depth_world = cameras.get_world_to_view_transform().inverse().transform_points(directions.view(-1, 3)).unsqueeze(0)

    return xy_depth_world


def initialize_point_cloud(initial_image: Image.Image, depth: np.ndarray, camera: PerspectiveCameras, device: str):
    # snap high gradients to nearest neighbor, which eliminates noodle artifacts
    depth = torch.from_numpy(depth).to(device)
    depth = snap_high_gradients_to_nn(depth, threshold=12)
    xy_depth_world = project_points(camera, depth)

    rgb = (torch.from_numpy(np.asarray(initial_image).copy()).reshape(-1, 3).float() / 255).to(device)
    point_cloud = get_pointcloud(xy_depth_world[0], device=device, features=rgb)

    return point_cloud


def outpaint_with_depth_estimation(
    image: torch.Tensor,
    mask: torch.Tensor,
    previous_depth: torch.Tensor,
    h: int,
    w: int,
    pipe: StableDiffusionInpaintPipeline,
    prior_da: PriorDepthAnything,
    prompt: str,
    camera: PerspectiveCameras,
    dilation_size: int = 2,
    depth_scaling: float = 1,
    generator = None,
):
    img_input = Image.fromarray((255*image[..., :3].cpu().numpy()).astype(np.uint8))

    # we slightly dilate the mask as aliasing might cause us to receive a too small mask from pytorch3d
    img_mask = Image.fromarray((255*skimage.morphology.isotropic_dilation(((~mask).cpu().numpy()), radius=dilation_size)).astype(np.uint8))#footprint=skimage.morphology.disk(dilation_size)))

    out_image = pipe(prompt=prompt, image=img_input, mask_image=img_mask, height=h, width=w, generator=generator).images[0]

    # depth completion
    out_depth = prior_da.infer_one_sample(
        image=np.array(out_image),
        prior=previous_depth.cpu().numpy(),
    )

    return out_image, out_depth


def extrapolate_point_cloud(
    pipe: StableDiffusionInpaintPipeline,
    prior_da: PriorDepthAnything,
    prompt: str,
    point_cloud: Pointclouds,
    camera_list: list[PerspectiveCameras],
    dry_run: bool = False,
    discard_mask: bool = False,
    initial_image: Image.Image | None = None,
    depth_scaling: float = 1,
    seed: int = 0,
    **render_kwargs,
):
    w, h = initial_image.size
    optimization_bundle_frames = []

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    for camera in tqdm(camera_list):
        images, masks, depths = render(camera, point_cloud, **render_kwargs)

        if not dry_run:
            eroded_mask = skimage.morphology.binary_erosion((depths[0] > 0).cpu().numpy(), footprint=None)
            eroded_depth = depths[0].clone()
            eroded_depth[torch.from_numpy(eroded_mask).to(depths.device) <= 0] = 0

            outpainted_img, aligned_depth = outpaint_with_depth_estimation(
                images[0],
                masks[0],
                eroded_depth,
                h,
                w,
                pipe,
                prior_da,
                prompt,
                camera,
                dilation_size=2,
                depth_scaling=depth_scaling,
                generator=generator)

        else:
            # in a dry run, we do not actually outpaint the image
            outpainted_img = Image.fromarray((255*images[0].cpu().numpy()).astype(np.uint8))

        if not dry_run:
            # snap high gradients to nearest neighbor, which eliminates noodle artifacts
            aligned_depth = snap_high_gradients_to_nn(aligned_depth.to(device), threshold=12).cpu()
            xy_depth_world = project_points(camera, aligned_depth)

        w2c = camera.get_world_to_view_transform().get_matrix()[0]

        optimization_bundle_frames.append({
            "rendered": Image.fromarray((255*images[0].cpu().numpy()).astype(np.uint8)),
            "image": outpainted_img,
            "mask": masks[0].cpu().numpy(),
            "transform_matrix": w2c.tolist(),
        })

        if discard_mask:
            optimization_bundle_frames[-1].pop("mask")

        if not dry_run:
            optimization_bundle_frames[-1]["center_point"] = xy_depth_world[0].mean(dim=0).tolist()
            optimization_bundle_frames[-1]["depth"] = aligned_depth.cpu().numpy()
            optimization_bundle_frames[-1]["mean_depth"] = aligned_depth.mean().item()

        else:
            # in a dry run, we do not modify the point cloud
            continue

        rgb = (torch.from_numpy(np.asarray(outpainted_img).copy()).reshape(-1, 3).float() / 255).to(device)

        # pytorch 3d's mask might be slightly too big (subpixels), so we erode it a little to avoid seams
        # in theory, 1 pixel is sufficient but we use 2 to be safe
        masks[0] = torch.from_numpy(skimage.morphology.binary_erosion(masks[0].cpu().numpy(), footprint=skimage.morphology.disk(2))).to(device)
        if torch.any(~masks[0]):
            partial_outpainted_point_cloud = get_pointcloud(xy_depth_world[0][~masks[0].view(-1)], device=device, features=rgb[~masks[0].view(-1)])

            point_cloud = merge_pointclouds([point_cloud, partial_outpainted_point_cloud])

    return optimization_bundle_frames, point_cloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_folder",
        type=str,
    )

    parser.add_argument(
        "--invert_depth",
        action="store_true",
    )

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
        "--outpaint_frame_interval",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345
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

    parser.add_argument(
        "--viser",
        action="store_true",
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    # limit GPU memory
    if args.gpu_memory_limit is not None:
        total_mem_gb = torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3)
        fraction = args.gpu_memory_limit / total_mem_gb
        torch.cuda.set_per_process_memory_fraction(fraction, args.gpu)
        print(f"GPU memory upper limit was set to {args.gpu_memory_limit:.2f}GB ({fraction:.2%})")

    # load the captionar
    blip_path = "Salesforce/blip2-opt-2.7b"
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    ).to(device)

    # load images
    image_path = sorted(glob.glob(os.path.join(args.input_folder, "images/*g")))[0]  # .jpg or .png
    img = Image.open(image_path).convert("RGB")
    assert img.width % 8 == 0 and img.height % 8 == 0, "Image dimensions must be multiples of 8"

    # load depth
    basename = os.path.splitext(os.path.basename(image_path))[0]
    depth_path = os.path.join(args.input_folder, f"depths/{basename}.npy")
    depth = np.load(depth_path)
    if args.invert_depth:
        depth = 10000./depth.clip(1e-5, None)
        depth = np.clip(depth, 0.0001, 10000)
    assert depth.shape == (img.height, img.width)

    # load cameras
    if os.path.isdir(os.path.join(args.input_folder, "cameras")):
        extrinsics_list = [np.load(os.path.join(args.input_folder, f"cameras/{i:04d}_extr.npy")) for i in range(args.num_frames)]
        intrinsics = np.load(os.path.join(args.input_folder, "cameras/intrinsics.npy"))
    else:  # older impl (TODO: Update `eval_data_i2v.py` and change this line accordingly)
        extrinsics_list = [np.load(os.path.join(args.trajectory_folder, f"{i:04d}_pose.npy")) for i in range(args.num_frames)]
        focal_len = 260
        intrinsics = np.array([[focal_len, 0, img.width/2], [0, focal_len, img.height/2], [0, 0, 1]], dtype=np.float32)

    # opencv numpy -> pytorch3d tensor
    extrinsics_list = [torch.from_numpy(np.diag([-1, -1, 1, 1]) @ e).unsqueeze(0) for e in extrinsics_list]
    camera_list = [
        PerspectiveCameras(
            focal_length=(np.diag(intrinsics)[:2],),
            principal_point=(intrinsics[:2, 2],),
            R=extr[:,:3,:3].permute(0,2,1),
            T=extr[:,:3,3],
            image_size=((img.height, img.width),),
            device=device,
            in_ndc=False,
        ) for extr in extrinsics_list]

    # get caption
    captioner_inputs = caption_processor(images=img, return_tensors="pt").to(device, torch.float16)
    generated_ids = captioner.generate(**captioner_inputs)
    prompt = caption_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    # print(f"\n\n\n{prompt=}\n\n\n", flush=True)
    del caption_processor
    del captioner

    # load pipeline
    prior_da = PriorDepthAnything(device=device, version='1.0')  # NOTE: version='1.1' shows artifacts
    pipe = get_sd_pipeline(device)

    # inference
    # monitor = GPUMemoryMonitor(gpu_id=args.gpu)
    # monitor.start()

    point_cloud = initialize_point_cloud(img, depth, camera_list[0], device)
    bundle_frames, point_cloud = extrapolate_point_cloud(
        pipe,
        prior_da,
        prompt,
        point_cloud,
        camera_list[args.outpaint_frame_interval::args.outpaint_frame_interval],
        dry_run=False,
        discard_mask=True,
        initial_image=img,
        depth_scaling=0.5,
        seed=args.seed,
        fill_point_cloud_holes=True,
    )

    # monitor.stop()
    # print(f"Peak GPU memory usage: {monitor.get_max_memory():.2f} GB")

    # rendered_images = [bundle_frames[i]["rendered"] for i in range(len(bundle_frames))]
    generated_images = [bundle_frames[i]["image"] for i in range(len(bundle_frames))]

    os.makedirs(args.output_folder, exist_ok=True)
    for i,fr in enumerate(generated_images):
        frame_id = (i + 1) * args.outpaint_frame_interval
        fr.save(os.path.join(args.output_folder, f"{frame_id:04d}.png"))
    export_to_video(generated_images, os.path.join(args.output_folder, "generated.mp4"))

    if args.viser:
        server = viser.ViserServer(port=8080, log_level="CRITICAL", verbose=False)
        server.gui.configure_theme(dark_mode=True)

        server.scene.reset()

        for i, extr in enumerate(extrinsics_list):
            extr = torch.from_numpy(np.diag([-1., -1., 1., 1.])) @ extr.squeeze()
            c2w = torch.linalg.inv(extr).cpu().numpy()
            server.scene.add_camera_frustum(
                f"/frames/frustum{i}",
                fov=2*np.arctan(0.5 * img.width / intrinsics[0, 0]),
                aspect=img.width/float(img.height),
                scale=0.1,
                color=(255, 255-i*10, 255-i*10),
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )


        server.scene.add_point_cloud(
            name="/point_cloud",
            points=point_cloud.points_packed().cpu().numpy().reshape(-1, 3),
            colors=np.array(img).reshape(-1,3),
            point_size=0.01,
        )

        import time
        time.sleep(300)
