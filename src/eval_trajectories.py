import argparse
import os
import time

import numpy as np
import torch
import viser
import viser.transforms as tf
from diffusers.utils import load_video
from evo.core import metrics
from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie
from jaxtyping import Float
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


def eval_trajectories(
        pred_traj: list[Float[np.ndarray, "4 4"]],  # MUST BE c2w
        gt_traj: list[Float[np.ndarray, "4 4"]],  # MUST BE c2w
        pred_proj: Float[np.ndarray, "3 3"],
        gt_proj: Float[np.ndarray, "3 3"]):
    pred_traj = np.stack(pred_traj).astype(np.float64)
    gt_traj = np.stack(gt_traj).astype(np.float64)

    pred_traj: PosePath3D = PosePath3D(poses_se3=pred_traj)
    gt_traj: PosePath3D = PosePath3D(poses_se3=gt_traj)

    ape = metrics.APE()
    rre = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg)
    # rte = RTE(pose_relation=metrics.PoseRelation.translation_part) # This computes the relative angle between the translations rather than the translation error
    rte = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part)

    rre_all = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg, all_pairs=True)
    rte_all = RTE(pose_relation=metrics.PoseRelation.translation_part, all_pairs=True)

    try:
        pred_traj.align(gt_traj, correct_scale=True)
    except:
        print("Could not align trajectories")

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

    pred_fx = pred_proj[0, 0].item()
    pred_fy = pred_proj[1, 1].item()
    pred_cx = pred_proj[0, 2].item()
    pred_cy = pred_proj[1, 2].item()

    gt_fx = gt_proj[0, 0].item()
    gt_fy = gt_proj[1, 1].item()
    gt_cx = gt_proj[0, 2].item()
    gt_cy = gt_proj[1, 2].item()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing the 'warped' folder and the 'generated' folder")
    parser.add_argument("--gt_focal_len", type=float, default=260.0, help="Focal length of the ground truth camera")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--visualize", action="store_true", help="Visualize the trajectories using viser")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.visualize:
        server = viser.ViserServer(port=8080)
        server.gui.configure_theme(dark_mode=True)

    AnyCam = torch.hub.load('Brummi/anycam', 'AnyCam', version="1.0", training_variant="seq8", pretrained=True).cuda()

    # Load the video frames and ground truth poses
    frames = load_video(os.path.join(args.data_dir, "generated/generated.mp4"))
    frames = [np.array(x).astype(np.float32) / 255.0 for x in frames]
    height, width, _ = frames[0].shape
    gt_poses_w2c = [np.load(os.path.join(args.data_dir, f"warped/{i:04d}_pose.npy")) for i in range(len(frames))]
    gt_poses_c2w = [np.linalg.inv(p) for p in gt_poses_w2c]
    gt_K = np.array([
        [args.gt_focal_len, 0, width/2],
        [0, args.gt_focal_len, height/2],
        [0, 0, 1],
    ], dtype=np.float32)

    # Estimate the camera poses and intrinsics from the generated video
    anycam_ret = AnyCam.process_video(frames, ba_refinement=True)
    estimated_poses_c2w = [p.cpu().numpy() for p in anycam_ret["trajectory"]]  # Camera poses
    estimated_K = anycam_ret["projection_matrix"].cpu().numpy()  # Camera intrinsics

    # compute errors
    results, estimated_abs_poses_c2w, gt_abs_poses_c2w = eval_trajectories(estimated_poses_c2w, gt_poses_c2w, estimated_K, gt_K)
    print(tabulate(results.items()))

    # visualize
    if args.visualize:

        for i, c2w in enumerate(gt_abs_poses_c2w.poses_se3):
            server.scene.add_camera_frustum(
                f"/frames/gt{i}/frustum",
                fov=2*np.arctan((height / 2.0) / gt_K[1, 1]),
                aspect=width/float(height),
                scale=0.5,
                color=(255, 255, 255),
                #image=np.array(warped_frames[i]),
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )

        for i, c2w in enumerate(estimated_abs_poses_c2w.poses_se3):
            server.scene.add_camera_frustum(
                f"/frames/estimated{i}/frustum",
                fov=2*np.arctan((height / 2.0) / estimated_K[1, 1]),
                aspect=width/float(height),
                scale=0.5,
                color=(255, 0, 0),
                #image=np.array(warped_frames[i]),
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )

        print("Visualizing trajectories in Viser. Open http://localhost:8080 in your browser.")
        time.sleep(100000)
