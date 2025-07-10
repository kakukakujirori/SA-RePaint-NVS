"""
To run this script, you must create another python environment with the following requirements:
- Python: 3.10
- PyTorch: 2.4.1
- CUDA: 11.8
"""
import argparse, glob, os, shutil, tempfile
from vbench import VBench

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VBench evaluation")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--cuda_home", type=str, default="/home/ryotaro/cuda-11.8", help="Path to CUDA 11.8 installation")
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

    # set up VBench
    device = f"cuda:{args.gpu}"
    my_VBench = VBench(device, "", "vbench_results")

    with tempfile.TemporaryDirectory() as td:
        print(f"Video are copied to {td} for VBench evaluation.")

        # copy generated images
        for scene in os.listdir(args.output_dir):
            scene_path = os.path.join(args.output_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            for motion_degree in os.listdir(scene_path):
                videopath = os.path.join(args.output_dir, scene, motion_degree, "generated/generated.mp4")
                videopath = os.path.normpath(videopath)  # delete the leading ./
                dst_path = os.path.join(td, videopath.replace("/", "_"))
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
