import argparse
import os

import cv2
import numpy as np
import OpenEXR
import pyvista as pv
import utils3d
from numba import njit
from PIL import Image


def create_grid(h:int, w: int):
    x_1d = np.arange(0, w)[None]
    y_1d = np.arange(0, h)[:, None]
    x_2d = np.repeat(x_1d, repeats=h, axis=0)
    y_2d = np.repeat(y_1d, repeats=w, axis=1)
    grid = np.stack([x_2d, y_2d], axis=2)
    return grid


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def look_at_matrix(camera_position, target, up):

    # Camera's forward vector (z-axis)
    forward = normalize(target - camera_position)
    # Camera's right vector (x-axis)
    right = normalize(np.cross(up, forward))
    # Camera's up vector (y-axis), ensure it is orthogonal to the other two axes
    up = np.cross(forward, right)

    # Create the rotation matrix by combining the camera axes to form a basis
    rotation = np.array([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1]
    ])

    # Create the translation matrix
    translation = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])

    # The view matrix is the inverse of the camera's transformation matrix
    # Here we assume the rotation matrix is orthogonal (i.e., rotation.T == rotation^-1)
    view_matrix = rotation.T @ translation

    return view_matrix


def generate_camera_poses(num_poses, angle_step, major_radius, minor_radius, camera_motion_mode, inverse=False):
    """
    Generate a camera pose that rotates around the origin, forming an elliptical trajectory.
    """
    poses = []

    for i in range(num_poses):
        angle = np.deg2rad(angle_step * i if not inverse else 360 - angle_step * i)

        if camera_motion_mode == 'horizontal':
            cam_x = major_radius * np.sin(angle)
            cam_y = 0
            cam_z =  minor_radius * np.cos(angle)
        elif camera_motion_mode == 'vertical':
            cam_y = major_radius * np.sin(angle)
            cam_x = 0
            cam_z =  minor_radius * np.cos(angle)
        elif camera_motion_mode == 'zoomin':
            cam_x = 0
            cam_y = 0
            cam_z =  minor_radius * np.cos(angle)
        elif camera_motion_mode == 'zoomout':
            cam_x = 0
            cam_y = 0
            cam_z =  minor_radius * (1+np.sin(angle))
        else:
            raise NotImplementedError

        look_at = np.array([0, 0, 0])
        camera_position = np.array([cam_x, cam_y, cam_z])
        up_direction = np.array([0, 1, 0])

        pose_matrix = look_at_matrix(camera_position, look_at, up_direction)
        poses.append(pose_matrix)
    return poses


################################################################


def compute_transformed_points(
        depth1: np.ndarray,
        transformation1: np.ndarray,
        transformation2: np.ndarray,
        intrinsic1: np.ndarray,
        intrinsic2: np.ndarray | None):
    """
    Computes transformed position for each pixel location
    """
    h, w = depth1.shape
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)

    transformation = np.matmul(transformation2, np.linalg.inv(transformation1))

    y1d = np.array(range(h))
    x1d = np.array(range(w))
    x2d, y2d = np.meshgrid(x1d, y1d)
    ones_2d = np.ones(shape=(h, w))
    ones_4d = ones_2d[:, :, None, None]
    pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

    intrinsic1_inv = np.linalg.inv(intrinsic1)
    intrinsic1_inv_4d = intrinsic1_inv[None, None]
    intrinsic2_4d = intrinsic2[None, None]
    depth_4d = depth1[:, :, None, None]
    trans_4d = transformation[None, None]

    unnormalized_pos = np.matmul(intrinsic1_inv_4d, pos_vectors_homo)
    world_points = depth_4d * unnormalized_pos
    world_points_homo = np.concatenate([world_points, ones_4d], axis=2)
    trans_world_homo = np.matmul(trans_4d, world_points_homo)
    trans_world = trans_world_homo[:, :, :3]
    trans_norm_points = np.matmul(intrinsic2_4d, trans_world)

    return trans_norm_points, world_points


@njit
def mask_occlusion_traj(
        depth: np.ndarray,  # (h, w)
        trans_depth: np.ndarray,  # (h, w)
        trans_coordinates: np.ndarray,  # (h, w, 2)
        trans_valid: np.ndarray,  # (h, w)
    ) -> np.ndarray:
    depth_new = np.full_like(trans_depth, np.inf)
    h, w = depth.shape

    for hh in range(1,h):
        for ww in range(1, w):
            x1, y1 = trans_coordinates[hh-1,ww-1]
            x2, y2 = trans_coordinates[hh,ww]
            x1, x2, y1, y2 = int(round(x1)), int(round(x2) + 1), int(round(y1)), int(round(y2) + 1)
            if 0 <= x1 < w and 0 <= x2 <= w and 0 <= y1 < h and 0 <= y2 <= h and \
                abs(depth[hh,ww] - depth[hh-1,ww-1]) < 0.1 * min(depth[hh,ww], depth[hh-1,ww-1]):
                    value_array = np.full(depth_new[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)].shape, trans_depth[hh,ww])
                    depth_new[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = np.minimum(value_array, depth_new[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)])

    for hh in range(1,h):
        for ww in range(1, w):
            x1, y1 = trans_coordinates[hh,ww]
            x1 = int(round(x1))
            y1 = int(round(y1))

            if (0 <= x1 < w and 0 <= y1 < h) and (trans_depth[hh, ww] > depth_new[y1, x1] * 1.1):
                    trans_valid[hh, ww] = False

    return trans_valid


def bilinear_splatting(
        frame1: np.ndarray,
        mask1: np.ndarray | None,
        depth1: np.ndarray,
        flow12: np.ndarray,
        flow12_mask: np.ndarray | None,
        is_image: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Using inverse bilinear interpolation based splatting
    :param frame1: (h, w, c)
    :param mask1: (h, w): True if known and False if unknown. Optional
    :param depth1: (h, w)
    :param flow12: (h, w, 2)
    :param flow12_mask: (h, w): True if valid and False if invalid. Optional
    :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
    :return: warped_frame2: (h, w, c)
                mask2: (h, w): True if known and False if unknown
    """
    h, w, c = frame1.shape
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if flow12_mask is None:
        flow12_mask = np.ones(shape=(h, w), dtype=bool)
    grid = create_grid(h, w)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = np.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
    trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_ceil[:, :, 0] = np.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_ceil[:, :, 1] = np.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

    prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

    sat_depth1 = np.clip(depth1, a_min=0, a_max=5000)
    log_depth1 = np.log(1 + sat_depth1)
    depth_weights = np.exp(log_depth1 / log_depth1.max() * 50)

    weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
    weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
    weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
    weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]

    warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
    warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)

    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]

    mask = cropped_weights > 0
    mask2 = cropped_weights <= 0.6
    mask = mask * mask2
    with np.errstate(invalid='ignore'):
        warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

    if is_image:
        assert np.min(warped_frame2) >= 0
        assert np.max(warped_frame2) <= 256
        clipped_image = np.clip(warped_frame2, a_min=0, a_max=255)
        warped_frame2 = np.round(clipped_image).astype('uint8')
    return warped_frame2, mask


################################################################


def forward_warp(
        frame1: np.ndarray,
        mask1: np.ndarray | None,
        depth1: np.ndarray,
        transformation1: np.ndarray,
        transformation2: np.ndarray,
        intrinsic1: np.ndarray,
        intrinsic2: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
    bilinear splatting.
    :param frame1: (h, w, 3) uint8 np array
    :param mask1: (h, w) bool np array. Wherever mask1 is False, those pixels are ignored while warping. Optional
    :param depth1: (h, w) float np array.
    :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
    :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
    :param intrinsic1: (3, 3) camera intrinsic matrix
    :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
    """
    h, w = frame1.shape[:2]
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    assert frame1.shape == (h, w, 3)
    assert mask1.shape == (h, w)
    assert depth1.shape == (h, w)
    assert transformation1.shape == (4, 4)
    assert transformation2.shape == (4, 4)
    assert intrinsic1.shape == (3, 3)
    assert intrinsic2.shape == (3, 3)

    trans_points1, world_points = compute_transformed_points(
        depth1,
        transformation1,
        transformation2,
        intrinsic1,
        intrinsic2)

    trans_coordinates = trans_points1[:, :, :2, 0] / (trans_points1[:, :, 2:3, 0])

    trans_depth1 = trans_points1[:, :, 2, 0]

    grid = create_grid(h, w)
    flow12 = trans_coordinates - grid
    flow12 = flow12.reshape(-1,2)
    flow12[trans_depth1.reshape(-1) < 0] = 100000 # important
    flow12 = flow12.reshape(h, w, 2)

    trans_coordinates = trans_coordinates.reshape(-1,2)
    trans_coordinates[trans_depth1.reshape(-1)<0] = 100000
    trans_coordinates = trans_coordinates.reshape(h, w, 2)

    trans_valid = (trans_depth1 > 0)
    trans_valid = mask_occlusion_traj(depth1, trans_depth1, trans_coordinates, trans_valid)

    warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)

    # make trans_coordiante 3D (for Diffusion As Shader)
    trans_coordinates_3D = np.concatenate([trans_coordinates, trans_depth1.reshape(h,w,1)], axis=-1)

    return warped_frame2, mask2, trans_coordinates_3D, trans_valid


def render_mesh(
        frame1: np.ndarray,
        depth1: np.ndarray,
        transformation1: np.ndarray,
        transformation2: np.ndarray,
        intrinsic1: np.ndarray,
        intrinsic2: np.ndarray | None,
        mask_deocclusion: bool = True,
        mask_postprocess_opening_kernel_size: int = 7,
    ) -> np.ndarray:
    """
    Given a frame1 and global transformations, warps frame1 to next view using
    pyvista with unlit rendering, and adds a 4th channel as a validity mask.
    """
    h, w = frame1.shape[:2]
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    assert frame1.shape == (h, w, 3)
    assert depth1.shape == (h, w)
    assert transformation1.shape == (4, 4)
    assert transformation2.shape == (4, 4)
    assert intrinsic1.shape == (3, 3)
    assert intrinsic2.shape == (3, 3)

    # make pyvista run background
    pv.set_plot_theme("document")
    pv.global_theme.lighting = False
    pv.global_theme.show_edges = False
    pv.global_theme.background = 'black'

    trans_points1, world_points = compute_transformed_points(
        depth1,
        transformation1,
        transformation2,
        intrinsic1,
        intrinsic2)

    trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
    trans_depth1 = trans_points1[:, :, 2, 0]
    trans_coordinates = trans_coordinates.reshape(-1,2)
    trans_coordinates[trans_depth1.reshape(-1)<0] = 100000
    trans_coordinates = trans_coordinates.reshape(h, w, 2)

    trans_valid = (trans_depth1 > 0)
    trans_valid = mask_occlusion_traj(depth1, trans_depth1, trans_coordinates, trans_valid)

    # make trans_coordiante 3D (for Diffusion As Shader)
    trans_coordinates_3D = np.concatenate([trans_coordinates, trans_depth1.reshape(h,w,1)], axis=-1)

    # build mesh components
    world_points = world_points.reshape(h, w, 3)
    valid_mask = ~utils3d.numpy.depth_edge(depth1, rtol=0.04)
    faces, vertices, vertex_colors_float, vertex_uvs = utils3d.numpy.image_mesh(
        world_points,
        frame1.astype(np.float32),
        utils3d.numpy.image_uv(width=w, height=h),
        mask=None if mask_deocclusion else valid_mask,
        tri=True
    )

    # pyvista mesh
    num_faces = len(faces)
    pv_faces = np.c_[np.full(num_faces, 3), faces].ravel()
    mesh = pv.PolyData(vertices, pv_faces)

    # camera and Plotter
    plotter = pv.Plotter(off_screen=True, window_size=[w, h])
    plotter.enable_anti_aliasing("ssaa")  # NOTE: IMPORTANT!!! Without it, rendered images contain noticeable salt-pepper noise
    relative_pose_cv = transformation1 @ np.linalg.inv(transformation2)
    R_c2w = relative_pose_cv[:3, :3]
    t_c2w = relative_pose_cv[:3, 3]
    position = t_c2w
    focal_point = position + R_c2w @ np.array([0, 0, 1])
    viewup = -R_c2w @ np.array([0, 1, 0])
    fov_y = np.degrees(2 * np.arctan(h / (2 * intrinsic2[1, 1])))
    plotter.camera.position = position
    plotter.camera.focal_point = focal_point
    plotter.camera.up = viewup
    plotter.camera.view_angle = fov_y
    plotter.camera.clipping_range = (
        plotter.camera.clipping_range[0],
        plotter.camera.clipping_range[1] + depth1.max() * 10,  # NOTE: empirically set
    )

    # 1. RGB rendering
    mesh['RGB'] = vertex_colors_float.astype(np.uint8)
    actor = plotter.add_mesh(mesh, scalars='RGB', rgb=True)
    prop = actor.GetProperty()
    prop.SetLighting(False)

    rgb_image = plotter.screenshot(transparent_background=False, return_img=True)
    if rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]

    # 2. Mask rendering
    if mask_deocclusion:
        mask_vertex_colors = np.zeros_like(vertex_colors_float, dtype=np.uint8)
        pixel_coords = (vertex_uvs * np.array([w - 1, h - 1])).round().astype(int)
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, w - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, h - 1)
        valid_vertex_indices = np.where(valid_mask[pixel_coords[:, 1], pixel_coords[:, 0]] == True)[0]
        mask_vertex_colors[valid_vertex_indices] = 255
        mesh['RGB'] = mask_vertex_colors
    else:
        mesh['RGB'] = np.full_like(vertex_colors_float, 255, dtype=np.uint8)

    # update plotter
    plotter.render()
    mask_image_rgb = plotter.screenshot(transparent_background=False, return_img=True)

    plotter.close()

    # Mask postprocessing (remove thin masks mis-reacted at depth edges)
    if mask_postprocess_opening_kernel_size > 1:
        assert mask_postprocess_opening_kernel_size % 2 == 1, f"{mask_postprocess_opening_kernel_size=} must be odd"
        morph_kernel = np.ones((mask_postprocess_opening_kernel_size, mask_postprocess_opening_kernel_size), np.uint8)
        mask_image_rgb_padded = cv2.copyMakeBorder(
            mask_image_rgb,
            mask_postprocess_opening_kernel_size,  # top
            mask_postprocess_opening_kernel_size,  # bottom
            mask_postprocess_opening_kernel_size,  # left
            mask_postprocess_opening_kernel_size,  # right
            cv2.BORDER_CONSTANT, None, value=[0, 0, 0])
        mask_image_rgb_padded = cv2.dilate(mask_image_rgb_padded, morph_kernel, iterations=1)
        mask_image_rgb_padded = cv2.erode(mask_image_rgb_padded, morph_kernel, iterations=1)
        mask_image_rgb = mask_image_rgb_padded[
            mask_postprocess_opening_kernel_size:-mask_postprocess_opening_kernel_size,
            mask_postprocess_opening_kernel_size:-mask_postprocess_opening_kernel_size,
        ]

    return rgb_image, (255 - mask_image_rgb), trans_coordinates_3D, trans_valid


def save_images(
        save_path: str,
        images_lists: list[np.ndarray],
        depth_lists: list[np.ndarray],
        num_frames: int = 25,
        focal_len: float = 260,
        degrees_per_frame: float = 1.0,
        major_radius: float = 80,
        minor_radius: float = 70,
        camera_motion_mode: str = "horizontal",
        no_occlusion_revealing: bool = False,
        use_mesh: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    The images will be center cropped after each warp.
    """
    if no_occlusion_revealing and (not use_mesh):
        import warnings
        warnings.warn("[save_image] 'no_occlusion_revealing' without using a mesh is inaccurate. It is advisable to set 'use_mesh=True'.")

    height, width, _ = images_lists[0].shape
    poses = generate_camera_poses(num_frames, degrees_per_frame,major_radius, minor_radius,camera_motion_mode)

    near = 0.0001
    far = 10000.
    K = np.eye(3)
    K[0,0] = focal_len; K[1,1] = focal_len; K[0,2] = width/2.0; K[1,2] = height/2.0

    pose_s = poses[0]
    never_occluded = np.ones((height, width), dtype=bool)

    trans_coordinates_3D_list = []
    trans_valid_list = []

    for i, pose_t in enumerate(poses):
        np.save(os.path.join(save_path, str(i).zfill(4)+"_pose.npy"), pose_t)

        image = images_lists[i]
        assert image.shape[:2] == (height, width), f"{image.shape=}, {height=}, {width=}"

        depth = depth_lists[i].astype(np.float32)
        depth = np.clip(depth, near, far)
        assert depth.shape == (height, width), f"{image.shape=} {depth.shape=}"

        if use_mesh:
            warped_frame2, mask2, trans_coordinates_3D, trans_valid = render_mesh(
                image,
                depth,
                pose_s,
                pose_t,
                K,
                None,
                mask_deocclusion=no_occlusion_revealing,
            )

            if i == 0:
                warped_frame2 = image
                mask2 = np.zeros_like(mask2)

            # binarize masks
            mask2 = np.where(mask2 > 0, 255, 0).astype(np.uint8)

            # save images
            warped_frame2[mask2 > 0] = 0
            Image.fromarray(mask2).save(os.path.join(save_path, str(i).zfill(4)+"_mask.png"))
            Image.fromarray(warped_frame2).save(os.path.join(save_path, str(i).zfill(4)+".png"))

        else:
            warped_frame2, mask2, trans_coordinates_3D, trans_valid = forward_warp(
                image,
                never_occluded,
                depth,
                pose_s,
                pose_t,
                K,
                None,
            )
            if no_occlusion_revealing:
                never_occluded *= trans_valid

            # save images
            mask = 1 - mask2
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = np.repeat(mask[:,:,np.newaxis]*255., repeats=3, axis=2)

            kernel = np.ones((5,5), np.uint8)
            mask_erosion = cv2.dilate(np.array(mask), kernel)
            mask_erosion = Image.fromarray(np.uint8(mask_erosion))
            mask_erosion.save(os.path.join(save_path, str(i).zfill(4)+"_mask.png"))

            mask_erosion_ = np.array(mask_erosion)/255.
            mask_erosion_[mask_erosion_ < 0.5] = 0
            mask_erosion_[mask_erosion_ >= 0.5] = 1
            warped_frame2 = Image.fromarray(np.uint8(warped_frame2 * (1-mask_erosion_)))
            warped_frame2.save(os.path.join(save_path, str(i).zfill(4)+".png"))

        trans_coordinates_3D_list.append(trans_coordinates_3D)
        trans_valid_list.append(trans_valid)

    return trans_coordinates_3D_list, trans_valid_list


def save_trajectories(
    save_path: str,
    save_trajectory_type: str | None,
    trans_coordinates_3D_list: list[np.ndarray],
    trans_valid_list: list[np.ndarray],
):
    height, width, _ = trans_coordinates_3D_list[0].shape

    # overwrite the first frame trans coordinates and valid (just in case)
    trans_coordinates_list = [coords[:, :, :2] for coords in trans_coordinates_3D_list]
    trans_coordinates_list[0] = create_grid(height, width)
    trans_valid_list[0] = trans_valid_list[0] + True

    if save_trajectory_type == "2d_npy":
        trans_coordinates = np.stack(trans_coordinates_list, axis=0)
        trans_valid = np.stack(trans_valid_list, axis=0)
        np.save(os.path.join(save_path, 'trans_coordinates.npy'), trans_coordinates)
        np.save(os.path.join(save_path, 'trans_valid.npy'), trans_valid)

    elif save_trajectory_type == "3d_rgb":
        import sys
        sys.path.append(__file__.rsplit('/', 2)[0])  # Adjust path to include the parent directory
        sys.path.append(os.path.join(__file__.rsplit('/', 2)[0], 'tools', 'DiffusionAsShader'))  # Adjust path to include the DiffusionAsShader
        from tools.DiffusionAsShader.models.pipelines import DiffusionAsShaderPipeline

        trans_coordinates_3D = np.stack(trans_coordinates_3D_list, axis=0)
        trans_coordinates_3D[:, :, :, 0] /= width
        trans_coordinates_3D[:, :, :, 1] /= height

        das = DiffusionAsShaderPipeline(gpu_id=0, output_dir=save_path)
        _, tracking_tensor = das.visualize_tracking_moge(
            points=trans_coordinates_3D,
            mask=None,
            save_tracking=False,  # TODO: turn off during evaluation
        )
        np.save(os.path.join(save_path, 'trans_coordinates_rgb.npy'), tracking_tensor.cpu().numpy())

    elif save_trajectory_type == "2d_homography":

        def spatially_normalize(M: np.ndarray | None):
            if M is None:
                return M

            # convert to kornia format ([0, H/W] -> [-1, 1])
            import torch
            from kornia.geometry.conversions import normalize_homography
            M_norm = normalize_homography(
                torch.from_numpy(M).reshape(1, 3, 3),
                (height, width), # Source size
                (height, width)  # Dest size
            )
            return M_norm.cpu().numpy().reshape(3, 3)

        # homography fitting
        from tqdm import tqdm
        homographies = []
        for trans_coord, valid_area in tqdm(zip(trans_coordinates_list, trans_valid_list), desc="Fitting homography", total=len(trans_valid_list)):
            src_points = create_grid(height, width)[valid_area].reshape(-1, 2)
            tgt_points = trans_coord[valid_area].reshape(-1, 2)

            fallback = homographies[-1] if len(homographies) > 0 else np.eye(3)

            if len(src_points) >= 4:
                M, _ = cv2.findHomography(src_points, tgt_points, cv2.LMEDS)
                M = spatially_normalize(M)
                det = np.linalg.det(M) if M is not None else 0
                if M is None or det <= 0 or det > 1e5 or abs(M[2, 2]) < 1e-6:
                    M = None
                else:
                    M = M / M[2, 2]  # NOTE: cv2.findHomography outputs are already M[2,2]=1?
            else:
                M = None

            homographies.append(M)

        # homography interpolation
        for idx_current, M in enumerate(homographies):
            if M is not None: continue

            idx_pre, M_pre = idx_current - 1, homographies[idx_current - 1]

            idx_post, M_post = -1, None
            for j in range(idx_current + 1, len(homographies)):
                if homographies[j] is not None:
                    idx_post = j
                    M_post = homographies[j]
                    break

            if M_post is None:
                M_interpolated = M_pre
            else:
                M_interpolated = ((idx_post - idx_current) * M_pre + (idx_current - idx_pre) * M_post) / (idx_post - idx_pre)
            homographies[idx_current] = M_interpolated

        homographies = np.stack(homographies, axis=0)

        # homography smoothing (element-wise)
        from scipy.ndimage import median_filter
        from scipy.signal import savgol_filter

        N = homographies.shape[0]
        window_length = max(3, N // 2)
        window_length = window_length if window_length % 2 == 1 else window_length + 1  # make it odd

        smoothed_homographies = np.zeros_like(homographies)
        for i in range(3):
            for j in range(3):
                cleaned = median_filter(homographies[:, i, j], size=5)
                smoothed_homographies[:, i, j] = savgol_filter(cleaned, window_length=window_length, polyorder=2)

        smoothed_homographies = smoothed_homographies / smoothed_homographies[:, 2:3, 2:3]

        # ensure that the starting homography is an identity
        blend_weight = np.linspace(1, 0, N).reshape(-1, 1, 1)
        blend_weight = blend_weight * np.linalg.inv(smoothed_homographies[0]) + (1 - blend_weight) * np.eye(3).reshape(1, 3, 3)
        smoothed_homographies = blend_weight @ smoothed_homographies

        homographies = smoothed_homographies

        np.save(os.path.join(save_path, 'trans_coordinates_homography.npy'), homographies)

    elif save_trajectory_type is None:
        pass
    else:
        raise NotImplementedError(f"Invalid save_trajectory_type: {save_trajectory_type}")

    return None


def read_exr(ext_file_path: str) -> np.ndarray:
    exr_file = OpenEXR.InputFile(ext_file_path)
    header = exr_file.header()
    data = np.frombuffer(exr_file.channel('Y'), dtype=np.float32)
    width = header['displayWindow'].max.x - header['displayWindow'].min.x + 1
    height = header['displayWindow'].max.y - header['displayWindow'].min.y + 1
    image = data.reshape((height, width))
    return image


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str
    )
    parser.add_argument(
        "--depth_folder",
        type=str
    )

    parser.add_argument(
        "--output_folder",
        type=str
    )

    parser.add_argument(
        "--depth_format",
        type=str,
        choices=["npy", "npz", "exr"],
        default="npy",
    )

    parser.add_argument(
        "--invert_depth",
        action="store_true",
        help="Set if the depth estimator returns inverse depth."
    )

    parser.add_argument(
        "--focal_len",
        type=float,
        default=260,
    )

    parser.add_argument(
        "--degrees_per_frame",
        type=float
    )

    parser.add_argument(
        "--camera_motion_mode",
        type=str,
        choices=['horizontal', 'vertical', 'zoomin', 'zoomout'],
    )

    parser.add_argument(
        "--major_radius",
        type=int,
        default=80
    )

    parser.add_argument(
        "--minor_radius",
        type=int,
        default=70
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=25
    )

    parser.add_argument(
        "--control_mode",
        type=str,
        choices=['image', 'video'],
        default='image'
    )

    parser.add_argument(
        "--no_occlusion_revealing",
        action="store_true",
        help="If set, the pixels once occluded in the past frames never show up in the latter frames."
    )

    parser.add_argument(
        "--use_mesh",
        action="store_true",
        help="If set, the trajectory will be extracted using a pyvista mesh renderer. "
             "Otherwise, it will be extracted using forward warping."
    )

    parser.add_argument(
        "--save_trajectory_type",
        type=str,
        choices=[None, "2d_npy", "3d_rgb", "2d_homography"],
        default=None,
        help="If set, the trajectory will be saved as a numpy file. (Use it for TrajectoryAttention or DiffusionAsShader)"
    )

    args = parser.parse_args()

    image_path = [os.path.join(args.image_folder, ip) for ip in sorted(os.listdir(args.image_folder))]
    image_list = [np.array(Image.open(ip)) for ip in image_path]

    if args.depth_format == "npy":
        depth_path_npy = [os.path.join(args.depth_folder, dp) for dp in sorted(os.listdir(args.depth_folder)) if dp.endswith('.npy')]
        depth_list = [np.load(dp) for dp in depth_path_npy]
    elif args.depth_format == "npz":
        depth_path_npz = [os.path.join(args.depth_folder, dp) for dp in sorted(os.listdir(args.depth_folder)) if dp.endswith('.npz')]
        assert len(depth_path_npz) == 1
        depth_list = np.load(depth_path_npz[0])['depths']
        depth_list = [depth_list[i] for i in range(depth_list.shape[0])]
    elif args.depth_format == "exr":
        depth_path_exr = [os.path.join(args.depth_folder, dp) for dp in sorted(os.listdir(args.depth_folder)) if dp.endswith('.exr')]
        depth_list = [read_exr(dp) for dp in depth_path_exr]
    else:
        raise NotImplementedError(f"Depth format '{args.depth_format}' is not yet implemented.")

    assert len(image_list) == len(depth_list), f"{len(image_list)=}, {len(depth_list)=}"

    if args.invert_depth:
        depth_list = [10000./depth.clip(1e-5, None) for depth in depth_list]

    if args.control_mode == 'image':
        image_list = image_list[0:1]
        depth_list = depth_list[0:1]

    # for image camera control
    if len(image_list) == 1:
        image_list = image_list * args.num_frames
        depth_list = depth_list * args.num_frames

    os.makedirs(args.output_folder, exist_ok=True)

    trans_coordinates_3D_list, trans_valid_list = save_images(
        args.output_folder,
        image_list,
        depth_list,
        args.num_frames,
        args.focal_len,
        args.degrees_per_frame,
        args.major_radius,
        args.minor_radius,
        args.camera_motion_mode,
        args.no_occlusion_revealing,
        args.use_mesh,
    )
    save_trajectories(
        args.output_folder,
        args.save_trajectory_type,
        trans_coordinates_3D_list,
        trans_valid_list,
    )
    print(f"Trajectory extraction finished, saved to {args.output_folder}")
