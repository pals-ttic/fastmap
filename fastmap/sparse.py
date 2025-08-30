import typing
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F

from fastmap.utils import to_homogeneous
from fastmap.container import Tracks, PointPairs, Cameras, Images, Points3D


def projection(
    pts3d: torch.Tensor,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    K: torch.Tensor,
):
    """Project 3D points to 2D.
    Args:
        pts3d: torch.Tensor (..., 3) 3D points
        R_w2c: torch.Tensor (..., 3, 3)
        t_w2c: torch.Tensor (..., 3)
        K: torch.Tensor (..., 3, 3)
    Returns:
        xy: torch.Tensor (..., 2) projected 2D points
        depth: torch.Tensor (...,) depth of the 3D points in the camera space
    """
    pts3d_cam = torch.einsum("...ij,...j->...i", R_w2c, pts3d) + t_w2c  # (..., 3)
    depth = pts3d_cam[..., 2]  # (...,)
    proj = torch.einsum("...ij,...j->...i", K, pts3d_cam)  # (..., 3)
    xy = proj[..., :2] / (proj[..., 2:] + 1e-8)  # (..., 2)
    return xy, depth


def triangulate(
    R_w2c1: torch.Tensor,
    t_w2c1: torch.Tensor,
    R_w2c2: torch.Tensor,
    t_w2c2: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    xy1: torch.Tensor,
    xy2: torch.Tensor,
):
    """Triangulate 3D points from 2D correspondences and camera poses. Also return angles between rays.
    Args:
        R_w2c1: torch.Tensor (..., 3, 3)
        t_w2c1: torch.Tensor (..., 3)
        R_w2c2: torch.Tensor (..., 3, 3)
        t_w2c2: torch.Tensor (..., 3)
        K1: torch.Tensor (..., 3, 3)
        K2: torch.Tensor (..., 3, 3)
        xy1: torch.Tensor (..., 2)
        xy2: torch.Tensor (..., 2)
    Returns:
        pts3d: torch.Tensor (..., 3)
        angles: torch.Tensor (...,) in degrees
    """
    ##### Get the device and dtype #####
    device = R_w2c1.device
    dtype = R_w2c1.dtype

    ##### Calibrate the 2D points #####
    xy_homo1 = to_homogeneous(xy1)  # (..., 3)
    xy_homo2 = to_homogeneous(xy2)  # (..., 3)
    xy_homo1 = torch.einsum("...ij,...j->...i", torch.inverse(K1), xy_homo1)  # (..., 3)
    xy_homo2 = torch.einsum("...ij,...j->...i", torch.inverse(K2), xy_homo2)  # (..., 3)
    xy1 = xy_homo1[..., :2] / xy_homo1[..., 2:]  # (..., 2)
    xy2 = xy_homo2[..., :2] / xy_homo2[..., 2:]  # (..., 2)
    del xy_homo1, xy_homo2

    ##### Get c2w rotations and translations #####
    R_c2w1 = torch.einsum("...ij->...ji", R_w2c1)  # (..., 3, 3)
    t_c2w1 = torch.einsum("...ij,...j->...i", R_c2w1, -t_w2c1)  # (..., 3)
    R_c2w2 = torch.einsum("...ij->...ji", R_w2c2)  # (..., 3, 3)
    t_c2w2 = torch.einsum("...ij,...j->...i", R_c2w2, -t_w2c2)  # (..., 3)

    ##### Get the camera centers in the world space #####
    o1 = t_c2w1  # (..., 3)
    o2 = t_c2w2  # (..., 3)

    ##### Get the normalized ray directions in the world space #####
    d1 = F.normalize(to_homogeneous(xy1), dim=-1)  # (..., 3)
    d1 = torch.einsum("...ij,...j->...i", R_c2w1, d1)  # (..., 3)
    d2 = F.normalize(to_homogeneous(xy2), dim=-1)  # (..., 3)
    d2 = torch.einsum("...ij,...j->...i", R_c2w2, d2)  # (..., 3)

    ##### Prevent misusing the variables #####
    del R_c2w1, R_c2w2, t_c2w1, t_c2w2

    ##### Get the equations #####
    I = torch.eye(3, device=device, dtype=dtype)  # (3, 3)
    I = I.expand_as(R_w2c1)  # (..., 3, 3)
    # camera 1
    A1 = I - torch.einsum("...i,...j->...ij", d1, d1)  # (..., 3, 3)
    b1 = torch.einsum("...ij,...j->...i", A1, o1)  # (..., 3)
    # camera 2
    A2 = I - torch.einsum("...i,...j->...ij", d2, d2)  # (..., 3, 3)
    b2 = torch.einsum("...ij,...j->...i", A2, o2)  # (..., 3)
    # equations
    AT_A1 = torch.einsum("...ij,...ik->...jk", A1, A1)  # (..., 3, 3)
    AT_A2 = torch.einsum("...ij,...ik->...jk", A2, A2)  # (..., 3, 3)
    AT_b1 = torch.einsum("...ij,...i->...j", A1, b1)  # (..., 3)
    AT_b2 = torch.einsum("...ij,...i->...j", A2, b2)  # (..., 3)
    # solve
    AT_A = (
        AT_A1 + AT_A2 + 1e-6 * torch.eye(3, device=device, dtype=dtype).expand_as(AT_A1)
    )  # (..., 3, 3) add a small value to the diagonal to make the matrix non-singular
    AT_b = AT_b1 + AT_b2  # (..., 3)
    pts3d = torch.linalg.solve(AT_A, AT_b)  # (..., 3)

    ##### compute angles between rays #####
    cos_angle = torch.einsum("...i,...i->...", d1, d2)  # (...,)
    cos_angle = cos_angle.clamp(-1.0 + 1e-8, 1.0 - 1e-8)  # (...,)
    angles = torch.acos(cos_angle) * 180 / np.pi  # (...,)

    ##### return #####
    return pts3d, angles


def sparse_reconstruction(
    tracks: Tracks,
    point_pairs: PointPairs,
    point_pair_mask: torch.Tensor,
    cameras: Cameras,
    images: Images,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    color2d: torch.Tensor | None = None,
    reproj_err_thr: float = 15.0,
    min_ray_angle: float = 2.0,
    batch_size: int = 4096 * 16,
):
    """Sparse reconstruction from tracks and point pairs.
    Args:
        tracks: Tracks container
        point_pairs: PointPairs container
        point_pair_mask: torch.Tensor float (num_point_pairs,), mask for valid point pairs
        cameras: Cameras container
        R_w2c: torch.Tensor (3, 3), rotation matrix from world to camera
        t_w2c: torch.Tensor (3,), translation vector from world to camera
        color2d: torch.Tensor (num_points2d, 3), color for the 2D points; will use all 0 (black) if None
        reproj_err_thr: float, threshold for re-projection error in pixels
        min_ray_angle: float, minimum triangulation angle between rays in degrees
        batch_size: int, batch size for computation
    Returns:
        container: Points3D container
    """
    ##### Get information #####
    device, dtype = R_w2c.device, R_w2c.dtype
    num_points2d = tracks.track_idx.shape[0]
    if color2d is None:
        color2d = torch.zeros(
            num_points2d, 3, device=device, dtype=torch.uint8
        )  # (num_points2d, 3)
        logger.info("No color provided, using black color for all points.")
    assert color2d.shape == (num_points2d, 3)
    assert color2d.dtype == torch.uint8
    logger.info(
        f"Re-projection error threshold: {reproj_err_thr} pixels for mean image size {images.widths.float().mean().item()} x {images.heights.float().mean().item()}"
    )

    ##### Initialize the buffers #####
    xyz = torch.zeros(
        tracks.num_tracks, 3, device=device, dtype=dtype
    )  # (num_tracks, 3)
    weights = torch.zeros(
        tracks.num_tracks, device=device, dtype=dtype
    )  # (num_tracks,)
    max_angle = -torch.inf + torch.zeros(
        tracks.num_tracks, device=device, dtype=dtype
    )  # (num_tracks,)

    ##### Triangulate 3D points #####
    # loop over batches
    logger.info("Triangulating 3D points...")
    for batch_data in point_pairs.query():
        # get data
        batch_R_w2c1 = R_w2c[batch_data.image_idx1]  # (B, 3, 3)
        batch_R_w2c2 = R_w2c[batch_data.image_idx2]  # (B, 3, 3)
        batch_t_w2c1 = t_w2c[batch_data.image_idx1]  # (B, 3)
        batch_t_w2c2 = t_w2c[batch_data.image_idx2]  # (B, 3)
        batch_K1 = cameras.K[cameras.camera_idx[batch_data.image_idx1]]  # (B, 3, 3)
        batch_K2 = cameras.K[cameras.camera_idx[batch_data.image_idx2]]  # (B, 3, 3)
        batch_xy1_pixels = batch_data.xy1_pixels  # (B, 2)
        batch_xy2_pixels = batch_data.xy2_pixels  # (B, 2)
        batch_track_idx = batch_data.track_idx  # (B,)
        batch_point_pair_mask = point_pair_mask[batch_data.point_pair_idx]  # (B,)

        # triangulate
        batch_points3d, batch_angles = triangulate(
            R_w2c1=batch_R_w2c1,
            t_w2c1=batch_t_w2c1,
            R_w2c2=batch_R_w2c2,
            t_w2c2=batch_t_w2c2,
            K1=batch_K1,
            K2=batch_K2,
            xy1=batch_xy1_pixels,
            xy2=batch_xy2_pixels,
        )

        # accumulate weights
        batch_weights = batch_point_pair_mask.to(dtype=dtype)  # (B,)
        weights.scatter_reduce_(
            dim=0,
            index=batch_track_idx,
            src=batch_weights,
            reduce="sum",
            include_self=True,
        )

        # accumulate 3D points
        batch_pts3d = batch_points3d * batch_weights.unsqueeze(-1)  # (B, 3)
        xyz.scatter_reduce_(
            dim=0,
            index=batch_track_idx[:, None].expand_as(batch_pts3d),
            src=batch_pts3d,
            reduce="sum",
            include_self=True,
        )

        # accumulate max angle
        if not batch_point_pair_mask.any():
            continue
        max_angle.scatter_reduce_(
            dim=0,
            index=batch_track_idx,
            src=batch_angles,
            reduce="max",
            include_self=True,
        )
    logger.info("Triangulation done.")

    # average the 3D points (ignore that some slots contain invalid values; we will check the re-projection error later)
    xyz /= weights.unsqueeze(-1) + 1e-8  # (num_tracks, 3)
    del weights

    ##### Compute the re-projection error for each 2D point #####
    logger.info("Computing re-projection error...")
    num_batches = (num_points2d + batch_size - 1) // batch_size
    error2d = torch.zeros(num_points2d, device=device, dtype=dtype)  # (num_points2d,)
    for i in range(num_batches):
        # get batch indices
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_points2d)

        # get data
        batch_track_idx = tracks.track_idx[start_idx:end_idx]  # (B,)
        batch_image_idx = tracks.image_idx[start_idx:end_idx]  # (B,)
        batch_R_w2c = R_w2c[batch_image_idx]  # (B, 3, 3)
        batch_t_w2c = t_w2c[batch_image_idx]  # (B, 3)
        batch_K = cameras.K[cameras.camera_idx[batch_image_idx]]  # (B, 3, 3)
        batch_xy_pixels = tracks.xy_pixels[start_idx:end_idx]  # (B, 2)
        batch_xyz = xyz[batch_track_idx]  # (B, 3)

        # project 3D points to 2D
        batch_xy_proj, batch_depth = projection(
            pts3d=batch_xyz,
            R_w2c=batch_R_w2c,
            t_w2c=batch_t_w2c,
            K=batch_K,
        )  # (B, 2), (B,)

        # compute the re-projection error
        batch_error2d = torch.linalg.norm(
            batch_xy_pixels - batch_xy_proj, dim=-1
        )  # (B,)

        # write back
        error2d[start_idx:end_idx] = batch_error2d
    logger.info("Re-projection error computed.")

    ##### Check the re-projection error #####
    valid_points2d_mask = error2d < reproj_err_thr  # (num_points2d,)

    ##### Filter out 3D points with less than 2 valid 2D points #####
    valid_points2d_count = torch.zeros(
        xyz.shape[0], device=device, dtype=torch.long
    )  # (num_tracks,)
    valid_points2d_count.scatter_reduce_(
        dim=0,
        index=tracks.track_idx,
        src=valid_points2d_mask.long(),
        reduce="sum",
    )  # (num_tracks,)
    valid_points3d_mask = valid_points2d_count >= 2  # (num_tracks,)
    del valid_points2d_count

    ##### Filter out 3D points with large angles between rays #####
    valid_points3d_mask &= max_angle > min_ray_angle  # (num_tracks,)
    del max_angle

    ##### Compute the mean re-projection error for each 3D point #####
    error3d = torch.zeros(
        tracks.num_tracks, device=device, dtype=dtype
    )  # (num_tracks,)
    error3d.scatter_reduce_(
        dim=0,
        index=tracks.track_idx[valid_points2d_mask],
        src=error2d[valid_points2d_mask],
        reduce="mean",
        include_self=False,
    )  # (num_tracks,)
    error3d[~valid_points3d_mask] = torch.nan
    del error2d

    ##### Accumulate the color of each 3D point #####
    color3d = torch.zeros(
        tracks.num_tracks, 3, device=device, dtype=dtype
    )  # (num_tracks, 3)
    color3d.scatter_reduce_(
        dim=0,
        index=tracks.track_idx[valid_points2d_mask][:, None].expand(-1, 3),
        src=color2d[valid_points2d_mask].to(dtype=dtype),
        reduce="mean",
        include_self=False,
    )  # (num_tracks, 3)
    assert torch.all(color3d >= 0.0) and torch.all(color3d <= 255.0)
    color3d = color3d.to(dtype=torch.uint8)  # (num_tracks, 3)
    del color2d

    ##### Build the Points3D container #####
    logger.info("Building the Points3D container...")
    # build the container with place holders for lists
    container = Points3D(
        xyz=xyz[valid_points3d_mask],
        rgb=color3d[valid_points3d_mask],
        error=error3d[valid_points3d_mask],
        image_idx_lists=[],
        xy_pixels_lists=[],
    )
    del xyz, color3d, error3d
    assert container.xyz.dtype == dtype
    assert container.rgb.dtype == torch.uint8
    assert container.error.dtype == dtype
    assert not torch.any(torch.isnan(container.xyz))
    assert not torch.any(torch.isnan(container.error))
    assert not torch.any(torch.isinf(container.xyz))
    assert not torch.any(torch.isinf(container.error))

    # fetch the keypoint mask, image idx and keypoint idx for each 2D point from GPU to lists on CPU
    tracks_image_idx_list = tracks.image_idx.tolist()  # List[int] (num_points2d,)
    tracks_xy_pixels_list = (
        tracks.xy_pixels.tolist()
    )  # List[List[float]] (num_points2d, 2)
    valid_points2d_mask_list = (
        valid_points2d_mask.tolist()
    )  # List[bool] (num_points2d,)
    del valid_points2d_mask
    assert (
        len(tracks_image_idx_list)
        == len(tracks_xy_pixels_list)
        == len(valid_points2d_mask_list)
        == num_points2d
    )

    # fill the lists
    for i in range(tracks.num_tracks):
        if not valid_points3d_mask[i]:
            continue

        # get track start and end indices
        start_idx = tracks.track_start[i].item()
        end_idx = start_idx + tracks.track_size[i].item()

        # get the mask of valid 2D points
        mask_list = valid_points2d_mask_list[
            start_idx:end_idx
        ]  # List[bool] (num_points2d_for_track,)

        # get the image indices and keypoint indices
        image_idx_list = [
            x
            for x, mask_value in zip(
                tracks_image_idx_list[start_idx:end_idx], mask_list
            )
            if mask_value
        ]
        xy_pixels_list = [
            tuple(x)
            for x, mask_value in zip(
                tracks_xy_pixels_list[start_idx:end_idx], mask_list
            )
            if mask_value
        ]
        assert len(image_idx_list) == len(xy_pixels_list)
        assert len(image_idx_list) > 0

        # add to lists
        container.image_idx_lists.append(image_idx_list)
        container.xy_pixels_lists.append(xy_pixels_list)

    # make sure the results are consistent
    num_valid_points3d = valid_points3d_mask.long().sum().item()
    num_valid_points3d = typing.cast(int, num_valid_points3d)
    assert container.num_points == num_valid_points3d
    assert container.xyz.shape == (container.num_points, 3)
    assert container.rgb.shape == (container.num_points, 3)
    assert container.error.shape == (container.num_points,)
    assert container.num_points == len(container.image_idx_lists)
    assert container.num_points == len(container.xy_pixels_lists)
    logger.info("Points3D container built.")

    ##### Log #####
    logger.info(
        f"Number of 3D points: {container.num_points} from {tracks.num_tracks} tracks"
    )
    logger.info(f"Mean re-projection error: {container.error.mean().item()} pixels")
    logger.info(f"Max re-projection error: {container.error.max().item()} pixels")
    logger.info(
        f"Mean track length: {sum(len(x) for x in container.image_idx_lists) / container.num_points:.2f}"
    )
    logger.info(f"Min track length: {min(len(x) for x in container.image_idx_lists)}")
    logger.info(f"Max track length: {max(len(x) for x in container.image_idx_lists)}")

    ##### Return #####
    return container
