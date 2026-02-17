import argparse
import glob
import os
import shutil
import subprocess
import time
import tempfile

import imagesize
import numpy as np
import pycolmap
import torch
import viser
import viser.transforms as tf
from diffusers.utils import load_image
from evo.core import metrics
from evo.core.geometry import GeometryException
from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie
from jaxtyping import Float
from scipy.spatial.transform import Rotation
from tabulate import tabulate


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


class RTE(metrics.RPE):
    @staticmethod
    def rpe_base(Q_i: np.ndarray, Q_i_delta: np.ndarray, P_i: np.ndarray,
                 P_i_delta: np.ndarray) -> np.ndarray:
        """
        Computes the relative SE(3) error pose for a single pose pair
        following the notation of the TUM RGB-D paper.
        :param Q_i: reference SE(3) pose at i
        :param Q_i_delta: reference SE(3) pose at i+delta
        :param P_i: estimated SE(3) pose at i
        :param P_i_delta: estimated SE(3) pose at i+delta
        :return: the RPE matrix E_i in SE(3)
        """
        Q_rel = lie.relative_se3(Q_i, Q_i_delta)
        P_rel = lie.relative_se3(P_i, P_i_delta)
        t_Q = Q_rel[:3, 3]
        t_P = P_rel[:3, 3]

        E = translation_angle(torch.tensor(t_Q[None, ]), torch.tensor(t_P[None, ])).numpy()
        E = np.array([[0, 0, 0, E[0]]])
        return E


def align_linear_trajectories(traj_query_posepath3d: PosePath3D, traj_ref_posepath3d: PosePath3D):
    """
    Aligns two linear camera trajectories (traj_query to traj_ref).

    Args:
        traj_query (list): A list of N 4x4 numpy arrays for the query trajectory.
        traj_ref   (list): A list of N 4x4 numpy arrays for the reference trajectory.

    Returns:
        tuple: A tuple (s, R, t) for the similarity transformation.
               s: scale, R: 3x3 rotation matrix, t: 3x1 translation vector.
    """
    traj_query: list[Float[np.ndarray, "4 4"]] = traj_query_posepath3d.poses_se3
    traj_ref: list[Float[np.ndarray, "4 4"]] = traj_ref_posepath3d.poses_se3

    # --- Step 1: Parameter extraction ---
    # translation vectors
    t_query_list = np.array([m[:3, 3] for m in traj_query])
    t_ref_list = np.array([m[:3, 3] for m in traj_ref])

    # origin
    p_query_start = t_query_list[0]
    p_ref_start = t_ref_list[0]

    # direction
    dir_query = t_query_list[-1] - p_query_start
    dir_ref = t_ref_list[-1] - p_ref_start

    # scale
    len_query = np.linalg.norm(dir_query)
    len_ref = np.linalg.norm(dir_ref)
    if len_query == 0:
        raise GeometryException("traj_query has zero length.")
    scale = len_ref / len_query

    dir_query_norm = dir_query / len_query
    dir_ref_norm = dir_ref / len_ref

    # --- Step 2: R_motion to align the motion direction ---
    dot_product = np.dot(dir_query_norm, dir_ref_norm).clip(-1.0, 1.0)

    if np.allclose(dot_product, 1.0):  # same direction
        R_motion = np.identity(3)
    elif np.allclose(dot_product, -1.0):  # opposite direction
        # 回転軸はdir_ref_normに直交する任意のベクトルでよい
        # 頑健な方法として、単位ベクトルとの外積で直交ベクトルを求める
        axis_vec = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(np.dot(axis_vec, dir_ref_norm)), 1.0):
            axis_vec = np.array([0.0, 1.0, 0.0])
        rot_axis = np.cross(dir_ref_norm, axis_vec)
        rot_axis /= np.linalg.norm(rot_axis)
        R_motion = Rotation.from_rotvec(np.pi * rot_axis).as_matrix()
    else:
        rot_axis = np.cross(dir_query_norm, dir_ref_norm)
        rot_angle = np.arccos(dot_product)
        R_motion = Rotation.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)).as_matrix()

    # --- Step 3: resolve unknown roll by the camera up direction ---
    R_query_start = traj_query[0][:3, :3]
    R_ref_start = traj_ref[0][:3, :3]

    R_query_aligned = R_motion @ R_query_start

    # get "up" vectors
    up_ref = R_ref_start[:, 1]
    up_query_aligned = R_query_aligned[:, 1]

    # project these "up" vectors to a plane orthogonal to the moving direction
    proj_up_ref = up_ref - np.dot(up_ref, dir_ref_norm) * dir_ref_norm
    proj_up_query = up_query_aligned - np.dot(up_query_aligned, dir_ref_norm) * dir_ref_norm

    proj_up_ref /= np.linalg.norm(proj_up_ref)
    proj_up_query /= np.linalg.norm(proj_up_query)

    # get the roll angle
    dot_product_roll = np.dot(proj_up_query, proj_up_ref).clip(-1.0, 1.0)
    roll_angle = np.arccos(dot_product_roll)

    # get roll direction
    cross_product_roll = np.cross(proj_up_query, proj_up_ref)
    if np.dot(cross_product_roll, dir_ref_norm) < 0:
        roll_angle = -roll_angle

    # rotation matrix for roll alignment
    R_roll = Rotation.from_rotvec(roll_angle * dir_ref_norm).as_matrix()

    # --- Step 4: Synthesize the final transformation (scale, R, t) ---
    R = R_roll @ R_motion
    t = p_ref_start - scale * (R @ p_query_start)

    traj_query_posepath3d.scale(scale)
    traj_query_posepath3d.transform(lie.se3(R, t))

    return traj_query_posepath3d


def eval_trajectories(
        pred_traj: list[Float[np.ndarray, "4 4"]],  # MUST BE c2w
        gt_traj: list[Float[np.ndarray, "4 4"]],  # MUST BE c2w
        pred_proj: Float[np.ndarray, "3 3"],
        gt_proj: Float[np.ndarray, "3 3"]):
    assert len(pred_traj) == len(gt_traj), "The number of poses in the predicted and ground truth trajectories must be the same."
    pred_traj = np.stack(pred_traj).astype(np.float64)
    gt_traj = np.stack(gt_traj).astype(np.float64)

    pred_traj: PosePath3D = PosePath3D(poses_se3=pred_traj)
    gt_traj: PosePath3D = PosePath3D(poses_se3=gt_traj)

    ape = metrics.APE()
    rre = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg)
    # rte = RTE(pose_relation=metrics.PoseRelation.translation_part) # This computes the relative angle between the translations rather than the translation error
    rte = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part)

    rre_all = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg, all_pairs=True)
    # rte_all = RTE(pose_relation=metrics.PoseRelation.translation_part, all_pairs=True) # This computes the relative angle between the translations rather than the translation error
    rte_all = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part, all_pairs=True)

    try:
        pred_traj.align(gt_traj, correct_scale=True)
    except GeometryException:
        try:
            # print("[eval_trajectories] Umeyama alignment failed. Assuming that the trajectories are straight lines.")
            pred_traj = align_linear_trajectories(pred_traj, gt_traj)
        except GeometryException as e:
            print(f"`align_linear_trajectories` failed: {e}")
            return None, pred_traj, gt_traj
    except Exception as e:
        raise e

    ape.process_data((pred_traj, gt_traj))
    rre.process_data((pred_traj, gt_traj))
    rte.process_data((pred_traj, gt_traj))
    rre_all.process_data((pred_traj, gt_traj))
    rte_all.process_data((pred_traj, gt_traj))

    ape_errors = ape.error
    rre_errors = rre.error
    rte_errors = rte.error

    all_rre_errors = rre_all.error
    all_rte_errors = rte_all.error

    ape_result = float(ape_errors.mean())
    rre_result = float(rre_errors.mean())
    rte_result = float(rte_errors.mean())

    auc_05 = (0.5 - np.minimum(rre_errors, rte_errors).clip(0, 0.5)).mean() / 0.5
    auc_1 = (1 - np.minimum(rre_errors, rte_errors).clip(0, 1)).mean()
    auc_3 = (3 - np.minimum(rre_errors, rte_errors).clip(0, 3)).mean() / 3
    auc_10 = (10 - np.minimum(rre_errors, rte_errors).clip(0, 10)).mean() / 10
    rre_0_01 = (rre_errors < 0.01).mean()
    rte_0_01 = (rte_errors < 0.01).mean()

    rre_0_1 = (rre_errors < 0.1).mean()
    rte_0_1 = (rte_errors < 0.1).mean()

    rre_1 = (rre_errors < 1).mean()
    rte_1 = (rte_errors < 1).mean()

    rre_5 = (rre_errors < 5).mean()
    rte_5 = (rte_errors < 5).mean()

    all_auc_3 = (3 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 3)).mean() / 3
    all_auc_5 = (5 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 5)).mean() / 5
    all_auc_10 = (10 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 10)).mean() / 10
    all_auc_30 = (30 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 30)).mean() / 30

    all_rre_1 = (all_rre_errors < 1).mean()
    all_rte_1 = (all_rte_errors < 1).mean()
    all_rre_5 = (all_rre_errors < 5).mean()
    all_rte_5 = (all_rte_errors < 5).mean()

    all_rre_15 = (all_rre_errors < 15).mean()
    all_rte_15 = (all_rte_errors < 15).mean()

    pred_fx = pred_proj[..., 0, 0].mean().item()
    pred_fy = pred_proj[..., 1, 1].mean().item()
    pred_cx = pred_proj[..., 0, 2].mean().item()
    pred_cy = pred_proj[..., 1, 2].mean().item()

    gt_fx = gt_proj[..., 0, 0].mean().item()
    gt_fy = gt_proj[..., 1, 1].mean().item()
    gt_cx = gt_proj[..., 0, 2].mean().item()
    gt_cy = gt_proj[..., 1, 2].mean().item()

    mean_fx_error = np.abs(pred_fx - gt_fx)
    mean_fy_error = np.abs(pred_fy - gt_fy)

    mean_cx_error = np.abs(pred_cx - gt_cx)
    mean_cy_error = np.abs(pred_cy - gt_cy)

    fx_below_10 = np.abs(pred_fx - gt_fx) < 10
    fy_below_10 = np.abs(pred_fy - gt_fy) < 10

    fx_below_40 = np.abs(pred_fx - gt_fx) < 40
    fy_below_40 = np.abs(pred_fy - gt_fy) < 40

    rel_fx_error = np.abs(pred_fx - gt_fx) / gt_fx
    rel_fy_error = np.abs(pred_fy - gt_fy) / gt_fy

    ape_errors = ape_errors.tolist()
    rre_errors = rre_errors.tolist()
    rte_errors = rte_errors.tolist()

    all_rre_errors = all_rre_errors.tolist()
    all_rte_errors = all_rte_errors.tolist()

    result = {
        "ape_mean": ape_result,
        "rre_mean": rre_result,
        "rte_mean": rte_result,
        "ape_errors": ape_errors,
        "rre_errors": rre_errors,
        "rte_errors": rte_errors,
        "all_rre_errors": all_rre_errors,
        "all_rte_errors": all_rte_errors,
        "auc_05": float(auc_05),
        "auc_1": float(auc_1),
        "auc_3": float(auc_3),
        "auc_10": float(auc_10),
        "rre_0_01": float(rre_0_01),
        "rte_0_01": float(rte_0_01),
        "rre_0_1": float(rre_0_1),
        "rte_0_1": float(rte_0_1),
        "rre_1": float(rre_1),
        "rte_1": float(rte_1),
        "rre_5": float(rre_5),
        "rte_5": float(rte_5),
        "all_auc_3": float(all_auc_3),
        "all_auc_5": float(all_auc_5),
        "all_auc_10": float(all_auc_10),
        "all_auc_30": float(all_auc_30),
        "all_rre_1": float(all_rre_1),
        "all_rte_1": float(all_rte_1),
        "all_rre_5": float(all_rre_5),
        "all_rte_5": float(all_rte_5),
        "all_rre_15": float(all_rre_15),
        "all_rte_15": float(all_rte_15),
        "mean_fx_error": float(mean_fx_error),
        "mean_fy_error": float(mean_fy_error),
        "mean_cx_error": float(mean_cx_error),
        "mean_cy_error": float(mean_cy_error),
        "fx_below_10": float(fx_below_10),
        "fy_below_10": float(fy_below_10),
        "fx_below_40": float(fx_below_40),
        "fy_below_40": float(fy_below_40),
        "rel_fx_error": float(rel_fx_error),
        "rel_fy_error": float(rel_fy_error),
        "traj_len": pred_traj.num_poses,
        }

    return result, pred_traj, gt_traj  # the latter two has the same scale


def run_anycam(image_paths: list[str], gpu_id: int = 0):
    assert gpu_id == 0, "AnyCam only supports GPU 0 somehow."
    AnyCam = torch.hub.load('Brummi/anycam', 'AnyCam', version="1.0", training_variant="seq8", pretrained=True).t0(f"cuda:{gpu_id}")
    frames = [np.array(load_image(x)).astype(np.float32) / 255.0 for x in image_paths]
    anycam_ret = AnyCam.process_video(frames, ba_refinement=True)
    estimated_poses_c2w = [p.cpu().numpy() for p in anycam_ret["trajectory"]]  # Camera poses
    estimated_K = anycam_ret["projection_matrix"].cpu().numpy()  # Camera intrinsics
    return estimated_poses_c2w, estimated_K


def run_glomap(image_paths: list[str], gt_width: int, gt_height: int, gt_focal_len: float, gpu_id: int = 0, verbose: bool = False):
    with tempfile.TemporaryDirectory() as td:
        img_dir = os.path.join(td, "images")
        out_dir = os.path.join(td, "sparse")
        os.mkdir(img_dir)
        os.mkdir(out_dir)

        # copy images
        for p in image_paths:
            shutil.copy(p, img_dir)
        subprocess.run(["mogrify", "-resize", f"{gt_width}x{gt_height}!", os.path.join(img_dir, "*")])

        # run glomap (NOTE: intrinsic parameters are set to the ground truth values)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            _ = subprocess.run(["colmap", "feature_extractor",
                                    "--image_path", img_dir,
                                    "--database_path", os.path.join(td, "database.db"),
                                    "--ImageReader.single_camera", "1",
                                    "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                                    "--ImageReader.camera_params", f"{gt_focal_len},{gt_width/2},{gt_height/2}",
                                    ],
                                    check=True, capture_output=not verbose, text=True, encoding='utf-8', env=env)
            _ = subprocess.run(["colmap", "exhaustive_matcher",
                                    "--database_path", os.path.join(td, "database.db"),
                                    ],
                                    check=True, capture_output=not verbose, text=True, encoding='utf-8', env=env)

            # Check if 'global_mapper' is available in colmap
            colmap_help = subprocess.run(["colmap", "help"], capture_output=True, text=True, check=False).stdout
            has_global_mapper = "global_mapper" in colmap_help

            if has_global_mapper:
                _ = subprocess.run(["colmap", "global_mapper",
                                    "--database_path", os.path.join(td, "database.db"),
                                    "--image_path", img_dir,
                                    "--output_path", os.path.join(td, "sparse"),
                                    "--GlobalMapper.ba_refine_focal_length", "0",
                                    ],
                                    check=True, capture_output=not verbose, text=True, encoding='utf-8', env=env)
            else:
                _ = subprocess.run(["glomap", "mapper",
                                    "--database_path", os.path.join(td, "database.db"),
                                    "--image_path", img_dir,
                                    "--output_path", os.path.join(td, "sparse"),
                                    "--BundleAdjustment.optimize_intrinsics", "0",
                                    "--skip_view_graph_calibration", "1",
                                    ],
                                    check=True, capture_output=not verbose, text=True, encoding='utf-8', env=env)

        except subprocess.CalledProcessError as e:
            print(f"\n❌ command failed: {e.cmd}")
            print(f"📤 stdout:\n{e.stdout}")
            print(f"📥 stderr:\n{e.stderr}")
            return [None for _ in range(len(image_paths))]

        # read estimated poses (NOTE: intrinsic parameters are set to the ground truth values)
        estimated_poses_c2w = [None for _ in range(len(image_paths))]
        recon = pycolmap.Reconstruction(os.path.join(td, "sparse/0"))
        for img in recon.images.values():
            rig: pycolmap.Rigid3d = img.frame.rig_from_world
            c2w = rig.inverse().matrix()
            c2w = np.concatenate([c2w, np.array([[0,0,0,1]], dtype=c2w.dtype)], axis=0)
            estimated_poses_c2w[int(img.name.split(".")[0])] = c2w
        # assert all([x.shape == (4,4) for x in estimated_poses_c2w])  # NOTE: In some cases, the pose may not be estimated for some frames, so we cannot assert this.

        return estimated_poses_c2w


def run_vggt(image_paths: list[str], gpu_id: int = 0):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = f"cuda:{gpu_id}"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images
    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extrinsic, intrinsic = extrinsic[0], intrinsic[0]
        extrinsic = torch.cat([extrinsic, torch.tensor([[[0, 0, 0, 1]]]).expand(extrinsic.shape[0], 1, 4).to(extrinsic)], dim=-2)

    # convert to c2w
    estimated_poses_c2w =torch.linalg.inv(extrinsic).cpu().numpy()
    estimated_K = intrinsic.cpu().numpy()

    return estimated_poses_c2w, estimated_K


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing the 'warped' folder and the 'generated' folder")
    parser.add_argument("--gt_focal_len", type=float, default=260.0, help="Focal length of the ground truth camera")
    parser.add_argument("--model", type=str, default="glomap", choices=["anycam", "glomap", "vggt"], help="Model to use for pose estimation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--visualize", action="store_true", help="Visualize the trajectories using viser")
    args = parser.parse_args()

    if args.visualize:
        server = viser.ViserServer(port=8080)
        server.gui.configure_theme(dark_mode=True)

    # Load the generated frames
    image_names = sorted(glob.glob(os.path.join(args.data_dir, f"generated/0*.png")))
    num_frames = len(image_names)

    # Load the gt image size
    gt_width, gt_height = imagesize.get(os.path.join(args.data_dir, f"warped/0000.png"))

    # Load the camera poses
    gt_poses_w2c = [np.load(os.path.join(args.data_dir, f"warped/{i:04d}_pose.npy")) for i in range(num_frames)]
    gt_poses_c2w = [np.linalg.inv(p) for p in gt_poses_w2c]
    gt_K = np.array([
        [args.gt_focal_len, 0, gt_width/2],
        [0, args.gt_focal_len, gt_height/2],
        [0, 0, 1],
    ], dtype=np.float32)

    # Estimate the camera poses and intrinsics from the generated video
    if args.model.lower() == "anycam":
        estimated_poses_c2w, estimated_K = run_anycam(image_names, gpu_id=args.gpu)

    elif args.model.lower() == "glomap":
        # NOTE: intrinsic parameters are set to the ground truth values
        estimated_poses_c2w = run_glomap(image_names, gt_width=gt_width, gt_height=gt_height, gt_focal_len=args.gt_focal_len, gpu_id=args.gpu, verbose=True)
        estimated_K = gt_K

    elif args.model.lower() == "vggt":
        estimated_poses_c2w, estimated_K = run_vggt(image_names, gpu_id=args.gpu)

    else:
        raise NotImplementedError(f"Model {args.model} is not implemented.")

    # Filter out missing poses
    missing_estimated_poses_ids = [i for (i, pose) in enumerate(estimated_poses_c2w) if pose is None]
    estimated_poses_c2w = [pose for (i, pose) in enumerate(estimated_poses_c2w) if i not in missing_estimated_poses_ids]
    gt_poses_c2w = [pose for (i, pose) in enumerate(gt_poses_c2w) if i not in missing_estimated_poses_ids]
    if len(estimated_K) == num_frames:
        estimated_K = np.array([estimated_K[i] for i in range(num_frames) if i not in missing_estimated_poses_ids])

    # compute errors
    results, estimated_abs_poses_c2w, gt_abs_poses_c2w = eval_trajectories(estimated_poses_c2w, gt_poses_c2w, estimated_K, gt_K)
    print(tabulate(results.items()))

    # visualize
    if args.visualize:

        for i, c2w in enumerate(gt_abs_poses_c2w.poses_se3):
            server.scene.add_camera_frustum(
                f"/frames/gt{i}/frustum",
                fov=2*np.arctan((gt_height / 2.0) / gt_K[1, 1]),
                aspect=gt_width/float(gt_height),
                scale=0.5,
                color=(255, 255, 255),
                #image=np.array(warped_frames[i]),
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )

        for i, c2w in enumerate(estimated_abs_poses_c2w.poses_se3):
            server.scene.add_camera_frustum(
                f"/frames/estimated{i}/frustum",
                fov=2*np.arctan((gt_height / 2.0) / estimated_K[1, 1]),
                aspect=gt_width/float(gt_height),
                scale=0.5,
                color=(255, 0, 0),
                #image=np.array(warped_frames[i]),
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )

        print("Visualizing trajectories in Viser. Open http://localhost:8080 in your browser.")
        time.sleep(100000)
