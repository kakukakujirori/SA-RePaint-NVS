from typing import Optional
import argparse, glob, os, struct, subprocess, tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import torch


class DB_reader:
    def __init__(self,db_path):
        con = sqlite3.connect(db_path)
        self.cursor = con.cursor()

    def print_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        data = self.cursor.fetchall()
        print('Available Tables:')
        for row in data:
            print(row)

    def get_image_fn(self,image_id):
        image_data = self.get_table_data('images')
        out_fn = None
        for id,fn in zip(image_data['image_id'],image_data['name']):
            if id == image_id: out_fn = fn
        return out_fn

    def get_image_id_from_fn(self,im_fn):
        image_data = self.get_table_data('images')
        out_id = None
        for id,fn in zip(image_data['image_id'],image_data['name']):
            if fn == im_fn: out_id = id
        return out_id

    def get_column_info(self,table_name):
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        data = self.cursor.fetchall()
        out = []
        for row in data:
            out.append({'name':row[1],'dtype':row[2]})
        return out

    def get_table_data(self,table_name):
        column_names = self.get_column_info(table_name)
        out = {}
        for col in column_names:
            out[col['name']] = []
        self.cursor.execute(f"SELECT * FROM {table_name};")
        data = self.cursor.fetchall()
        for row in data:
            for col_idx,col in enumerate(row):
                out[column_names[col_idx]['name']].append(col)
        return out

    def print_table_by_name(self,table_name):
        def _print_header(col_data):
            # get headers
            for col in col_data:
                print(f"{col['name'].center(16)} |",end='')
            print('')
            print('-'*18*len(col_data))

        def _print_table(col_data, data, no_blob=True):
            _print_header(col_data)

            # print data
            keys = list(data.keys())
            n_rows = len(data[keys[0]])
            for row_idx in range(n_rows):
                for col in col_data:
                    if no_blob and col['dtype'] == 'BLOB':
                        print(f"BLOB".center(16),'|',end='')
                    else:
                        print(f"{data[col['name']][row_idx]}".center(16),'|',end='')
                print()

        _print_table(self.get_column_info(table_name),self.get_table_data(table_name))


def extract_keypoints(table_data):
    n_images = len(table_data['image_id'])
    out = {}
    for image_id,n_rows,n_cols,data in zip(table_data['image_id'],table_data['rows'],table_data['cols'],table_data['data']):
        n_floats = n_rows*n_cols
        if n_floats > 0:
            float_array = struct.unpack(f'{n_floats}f',data)
            float_table = np.array(float_array).reshape(n_rows,n_cols)
            out[image_id] = float_table
        else:
            out[image_id] = np.zeros((n_rows,n_cols))
    return out


def extract_matches(match_data):
    out = []
    for pair_id, n_rows, n_cols, data in zip(match_data['pair_id'],match_data['rows'],match_data['cols'],match_data['data']):
        im2_id = image_id2 = pair_id % 2147483647
        im1_id = (pair_id - im2_id) // 2147483647
        n_ids = n_rows * n_cols
        if n_ids > 0:
            int32_array = struct.unpack(f'{n_ids}i',data)
            int32_table = np.array(int32_array).reshape(n_rows,n_cols)
        else:
            int32_table = np.zeros((n_rows,n_cols))
        out.append([[im1_id,im2_id],int32_table])
    return out


def get_matches_for_pair(match_data: dict, id1: int, id2: int) -> list | np.ndarray:
    for matches in match_data:
        if matches[0] == [id1, id2]:
            return matches[1]
        elif matches[0] == [id2, id1]:
            return np.flip(matches[1], axis=1)
    return []


def get_fundamental_matrix(Rt1, Rt2, K1, K2):
    """Compute the essential matrix between two camera matrices.

    Args:
        Rt1 (4x4 np.ndarray): Camera 1 extrinsic matrix
        Rt2 (4x4 np.ndarray): Camera 2 extrinsic matrix
        K1  (3x3 np.ndarray): Camera 1 intrinsic matrix
        K2  (3x3 np.ndarray): Camera 2 intrinsic matrix

    Note:
        F = [e']_x @ P2 @ P1^+ (Hartley-Zisserman eq.(9.1))
        P1^+ : the pseudo-inverse of P1
        e'   : the epipole in the second camera frame (i.e. the projection of the first camera centre)
    """
    assert Rt1.shape == (4, 4) and np.allclose(Rt1[3, :], np.array([0, 0, 0, 1]))
    assert Rt2.shape == (4, 4) and np.allclose(Rt2[3, :], np.array([0, 0, 0, 1]))
    assert K1.shape == (3, 3) and np.allclose(K1[2, :], np.array([0, 0, 1]))
    assert K2.shape == (3, 3) and np.allclose(K2[2, :], np.array([0, 0, 1]))

    zero3 = np.zeros((3, 1))
    P1 = np.concatenate([K1, zero3], axis=-1) @ Rt1
    P2 = np.concatenate([K2, zero3], axis=-1) @ Rt2

    P1_inv = np.linalg.pinv(P1)
    C1 = -Rt1[:3, :3].T @ Rt1[:3, 3:4]
    e2 = (P2 @ np.concatenate([C1, torch.ones(1,1)], axis=0)).ravel()
    e2 = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])
    F = e2 @ P2 @ P1_inv
    return F


def get_essential_matrix(Rt1, Rt2):
    """Compute the essential matrix between two camera matrices.

    Args:
        Rt1 (4x4 np.ndarray): Camera 1 extrinsic matrix
        Rt2 (4x4 np.ndarray): Camera 2 extrinsic matrix

    Returns:
        3x3 np.ndarray: Essential matrix E (x2^T @ E @ x1 = 0)

    Note:
        E = [t]_x R if P1=I and P2=[R|t] (Harley-Zisserman, the line before Definition 9.16.)
        Also, fundamental matrices are projective invariant, so we can normalize Rt1 to I by applying inverse of Rt1.
    """
    assert Rt1.shape == (4, 4) and np.allclose(Rt1[3, :], np.array([0, 0, 0, 1]))
    assert Rt2.shape == (4, 4) and np.allclose(Rt2[3, :], np.array([0, 0, 0, 1]))

    R1, t1 = Rt1[:3, :3], Rt1[:3, 3:4]
    R2, t2 = Rt2[:3, :3], Rt2[:3, 3:4]

    # Bring everything under cam1 coordinate system (computing P2 @ inv(P1))
    R_rel = R2 @ R1.T
    t_rel = (t2 - R_rel @ t1).ravel()

    # E = [t]_x R if P1=I and P2=[R|t]
    t_x = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])

    # Essential matrix: [t]_x R
    E = t_x @ R_rel

    return E


def get_min_dist(p1: np.ndarray, E12: np.ndarray, p2: np.ndarray):
    """Compute the epipolar error (distance from p2 to the epipolar line E12 @ p1)

    Args:
        p1 (1x3 np.ndarray): A point in the image 1
        E12 (3x3 np.ndarray): Essential matrix
        p2 (1x3 np.ndarray): A point in image 2

    Returns:
        float: Distance from p2 to the epipolar line E12 @ p1
    """
    assert p1.shape == (1, 3) and p1[0, -1] == 1, f"{p1=}"
    assert p2.shape == (1, 3) and p2[0, -1] == 1, f"{p2=}"
    assert E12.shape == (3, 3)
    line = (E12 @ p1.reshape(3, 1)).ravel()
    dist = np.abs(p2 @ line) / np.linalg.norm(line[:2])
    return dist


def compute_consistency(
    median_seds: list[dict[str, float]],
    intra_scene_sed_threshold: float = 2,
    intra_scene_min_consistent: float = 10,
) -> float:
    n_pairs = len(median_seds)
    n_consistent_pairs = 0
    for raw_data in median_seds:
        median = raw_data['median']
        n_matches = raw_data['n_matches']
        if n_matches >= intra_scene_min_consistent and median < intra_scene_sed_threshold:
            n_consistent_pairs += 1
    return n_consistent_pairs / n_pairs


def eval_sed(
        colmap_root: str,
        video_path: str,
        poses: list[np.ndarray],
        gt_focal_len: float = 260.0,
        save_sed_graph_to: Optional[str] = None,
    ):
    colmap_image_dir = os.path.join(colmap_root, "images")
    colmap_database_path = os.path.join(colmap_root, "colmap.db")

    os.makedirs(colmap_root, exist_ok=True)
    os.makedirs(colmap_image_dir, exist_ok=True)

    # Read video frames -> save for COLMAP
    assert os.path.isfile(video_path), f"Video file not found: {video_path}"
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    num_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (1024, 576))
        cv2.imwrite(os.path.join(colmap_image_dir, f"{num_frames:04d}.png"), resized_frame)
        num_frames += 1
        video_frames.append(resized_frame[:,:,::-1])

    # Define camera matrices
    assert len(poses) == num_frames, f"{len(poses)=}, {num_frames=}"
    intrinsic = np.array([
        [gt_focal_len, 0, 512],
        [0, gt_focal_len, 288],
        [0, 0, 1],
    ], dtype=np.float32)

    # feature extraction
    cmd = f'colmap feature_extractor --SiftExtraction.use_gpu 1 --SiftExtraction.edge_threshold 30 --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --database_path {colmap_database_path} --image_path {colmap_image_dir}'
    subprocess.run(cmd, shell=True, capture_output=True)

    # feature matcher
    cmd = f'colmap exhaustive_matcher --database_path {colmap_database_path}'
    subprocess.run(cmd, shell=True, capture_output=True)

    # Read COLMAP database
    reader = DB_reader(colmap_database_path)
    keypoint_data = extract_keypoints(reader.get_table_data('keypoints'))
    match_data = extract_matches(reader.get_table_data('matches'))

    # Compute sed for neighboring images
    seds_summary = []
    for idx_1 in range(num_frames-1):
        idx_2 = idx_1 + 1

        # find im_ids
        vis_pair = [f"{idx_1:04d}.png", f"{idx_2:04d}.png"]
        id1 = reader.get_image_id_from_fn(vis_pair[0])
        id2 = reader.get_image_id_from_fn(vis_pair[1])

        # load match info
        cur_matches = get_matches_for_pair(match_data, id1, id2)

        # load pose/camera data
        pose_1 = poses[idx_1]
        pose_2 = poses[idx_2]
        F_12 = get_fundamental_matrix(pose_1, pose_2, intrinsic, intrinsic)
        F_21 = get_fundamental_matrix(pose_2, pose_1, intrinsic, intrinsic)

        seds = []
        for match in cur_matches:
            kp1 = keypoint_data[id1][match[0]]
            kp2 = keypoint_data[id2][match[1]]

            # Extract positions of SIFT keypoints (x, y, scale, orientation, response/score, octave/layer)
            # NOTE: SIFT features (128-dim) are stored in a different file, which we don't use here.
            p1 = np.array([ kp1[:2].tolist()+[1] ])
            p2 = np.array([ kp2[:2].tolist()+[1] ])

            # project 3d normal to get normal of 2d line
            ed_2 = get_min_dist(p1, F_12, p2)
            ed_1 = get_min_dist(p2, F_21, p1)

            sed = 0.5*(abs(ed_2) + abs(ed_1))
            seds.append(sed)
        n_matches = len(seds)
        mean = 99999999999 if n_matches == 0 else np.mean(seds)
        median = 99999999999 if n_matches == 0 else np.median(seds)
        seds_summary.append({'pair':vis_pair, 'mean': mean, 'median':median, 'n_matches':n_matches})

    # Thresholds for SED
    sed_thresholds = np.linspace(0, 1, 101)
    consistent_ratios = {}
    for intra_scene_sed_threshold in sed_thresholds:
        ratio = compute_consistency(seds_summary, intra_scene_sed_threshold, 10)
        consistent_ratios[intra_scene_sed_threshold] = ratio

    # Draw sed graph
    if save_sed_graph_to is not None:
        plt.plot(sed_thresholds, list(consistent_ratios.values()))
        plt.title(f'SED visualization\n{video_path}')
        plt.xlabel('sed_threshold')
        plt.ylabel('Percentage of consistent frame pairs')
        plt.savefig(os.path.join(save_sed_graph_to, "SED_graph.png"))

    return consistent_ratios, seds_summary


if __name__ == '__main__':
    """ Example usage:
    python eval_sed.py --output_root ../output/basketball/horizontal_1.0 --save_sed_graph_to ../output/basketball/horizontal_1.0
    """
    parser = argparse.ArgumentParser(description="Evaluate SED for a video")
    parser.add_argument("--output_root", type=str, help="NVS-Solver's output directory for each scene")
    parser.add_argument("--gt_focal_len", type=float, default=260.0, help="Ground truth focal length for the camera")
    parser.add_argument("--save_sed_graph_to", type=str, default=None, help="Path to save the SED graph (default: don't save)")
    args = parser.parse_args()

    video_path = os.path.join(args.output_root, "generated/generated.mp4")
    camera_paths = os.path.join(args.output_root, "warped/*_pose.npy")
    poses = [np.load(p) for p in sorted(glob.glob(camera_paths))]

    with tempfile.TemporaryDirectory() as colmap_root:
        consistent_ratios, sed_summary = eval_sed(
            colmap_root=colmap_root,
            video_path=video_path,
            poses=poses,
            gt_focal_len=args.gt_focal_len,
            save_sed_graph_to=args.save_sed_graph_to,
        )
