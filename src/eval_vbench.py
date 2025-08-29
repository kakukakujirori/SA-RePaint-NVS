"""
To run this script, you must create another python environment with the following requirements:
- Python: 3.10
- PyTorch: 2.4.1
- CUDA: 11.8
"""
import argparse, os, shutil, tempfile
from tqdm import tqdm

import torch
from diffusers.utils import load_image
from transformers import AutoProcessor, Blip2ForConditionalGeneration, set_seed
from vbench import VBench

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VBench evaluation")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--cuda_home", type=str, default="/usr/local/cuda-11.8", help="Path to CUDA 11.8 installation")
    args = parser.parse_args()

    # set up CUDA environment variables
    assert os.path.isdir(args.cuda_home)
    os.environ["CUDA_HOME"] = args.cuda_home

    existing_path = os.environ.get('PATH', '')
    cuda_bin_path = os.path.join(args.cuda_home, "bin")
    os.environ['PATH'] = f"{cuda_bin_path}{os.pathsep}{existing_path}"

    existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    cuda_lib_path = os.path.join(args.cuda_home, "lib64")
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}{os.pathsep}{existing_ld_path}"

    # set up BLIP2
    blip_path = "Salesforce/blip2-opt-2.7b"
    caption_processor = AutoProcessor.from_pretrained(blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(blip_path, torch_dtype=torch.float16).to("cuda:0")

    # set up VBench
    device = f"cuda:{args.gpu}"
    my_VBench = VBench(device, "", "vbench_results")

    with tempfile.TemporaryDirectory() as td:
        # copy generated images
        for scene in tqdm(os.listdir(args.output_dir), desc="Captioning videos"):
            scene_path = os.path.join(args.output_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                videopath = os.path.join(args.output_dir, scene, motion_degree, "generated/generated.mp4")
                videopath = os.path.normpath(videopath)  # delete the leading ./

                # captioning
                imagepath = os.path.join(args.output_dir, scene, motion_degree, "warped/0000.png")
                image = load_image(imagepath)
                with torch.no_grad():
                    set_seed(42)
                    captioner_inputs = caption_processor(images=image, return_tensors="pt").to(device, torch.float16)
                    generated_ids = captioner.generate(**captioner_inputs)
                    generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                # video name is refereed as a text query for Overall Consistency evaluation
                dst_path = os.path.join(td, f"{generated_text}.mp4")
                shutil.copy(videopath, dst_path)

        # run
        my_VBench.evaluate(
            videos_path=td,
            name=os.path.basename(os.path.normpath(args.output_dir)),
            dimension_list=[
                'subject_consistency',
                'background_consistency',
                'temporal_flickering',
                'motion_smoothness',
                'overall_consistency',
                'aesthetic_quality',
                'imaging_quality',
            ],
            mode="custom_input",
        )
