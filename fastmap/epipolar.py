from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmap.timer import timer
from fastmap.container import PointPairs, Cameras, Images
from fastmap.utils import (
    to_homogeneous,
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
    normalize_matrix,
    vector_to_skew_symmetric_matrix,
)
from fastmap.utils import ConvergenceManager


class EpipolarAdjustmentParameters(nn.Module):
    """Module for storing all the global epipolar adjustment parameters"""

    def __init__(
        self,
        R_w2c: torch.Tensor,
        t_w2c: torch.Tensor,
        focal_scale: torch.Tensor,
        precision: torch.dtype = torch.float32,
    ) -> None:
        """
        Args:
            R_w2c (torch.Tensor): (num_images, 3, 3), the w2c camera rotations
            t_w2c (torch.Tensor): (num_images, 3), the w2c camera translations
            focal (torch.Tensor): (num_cameras,), the scale factor on focal lengths
            precision (torch.dtype): the precision for the parameters
        """
        super().__init__()
        self.rot6d_w2c = nn.Parameter(
            rotation_matrix_to_6d(R_w2c.clone().to(precision)), requires_grad=True
        )  # (num_images, 6)
        self.t_w2c = nn.Parameter(
            t_w2c.clone().to(precision), requires_grad=True
        )  # (num_images, 3)
        self.inv_focal_scale = nn.Parameter(
            (1.0 / focal_scale).to(precision), requires_grad=True
        )  # (num_cameras,)

    def forward(self):
        """Return the parameters after some processing"""
        # get the parameters
        rot6d_w2c = self.rot6d_w2c  # (num_images, 6)
        t_w2c = self.t_w2c  # (num_images, 3)
        inv_focal_scale = self.inv_focal_scale  # (num_cameras,)
        focal_scale = 1.0 / inv_focal_scale  # (num_cameras,)

        # convert the rotation to matrix
        R_w2c = rotation_6d_to_matrix(rot6d_w2c)  # (num_images, 3, 3)

        # return the results
        return R_w2c, t_w2c, focal_scale, inv_focal_scale


@torch.no_grad()
def quadratic_form(
    num_image_pairs: int,
    point_pairs: PointPairs,
    image_pair_idx: torch.Tensor,
    prev_fundamental: torch.Tensor,
    point_pair_mask: torch.Tensor,
    precision: torch.dtype = torch.float32,
):
    """Compute the quadratic form of weighted L2 loss for epipolar adjustment.

    Args:
        num_image_pairs: int, the number of image pairs
        point_pairs: PointPairs container
        image_pair_idx: torch.Tensor long (num_point_pairs,), the image pair idx for each point pair
        prev_fundamental: torch.Tensor float (num_image_pairs, 3, 3), the previous fundamental matrix (for weighting)
        point_pair_mask: torch.Tensor bool (num_point_pairs,), the mask indicating the inlier point pairs
        precision: torch.dtype, the precision to return (but always use float64 for accumulation)

    Returns:
        W: torch.Tensor float (num_image_pairs, 9, 9), the quadratic form of the weighted L2 loss
    """
    ##### Get some information #####
    device = point_pairs.device
    num_point_pairs = point_pairs.num_point_pairs
    assert image_pair_idx.shape == (num_point_pairs,)
    assert precision in [torch.float32, torch.float64]

    ##### Accumulate the quadratic form #####
    # initialize the quadratic form (always use float64 for accumulation)
    W = torch.zeros(
        num_image_pairs, 9, 9, device=device, dtype=torch.float64
    )  # (num_image_pairs, 9, 9)

    for batch_data in point_pairs.query():
        # get the data
        batch_point_pair_mask = point_pair_mask[
            batch_data.point_pair_idx
        ]  # (num_point_pairs_in_batch,)
        if not torch.any(batch_point_pair_mask):
            continue
        num_valid_point_pairs_in_batch = batch_point_pair_mask.long().sum().item()
        batch_image_pair_idx = image_pair_idx[
            batch_data.point_pair_idx[batch_point_pair_mask]
        ]  # (num_valid_point_pairs_in_batch,)
        batch_fundamental = prev_fundamental[
            batch_image_pair_idx
        ]  # (num_valid_point_pairs_in_batch, 3, 3)
        batch_xy_homo1 = F.normalize(
            to_homogeneous(batch_data.xy1[batch_point_pair_mask]), p=2, dim=-1
        )  # (num_valid_point_pairs_in_batch, 3)
        batch_xy_homo2 = F.normalize(
            to_homogeneous(batch_data.xy2[batch_point_pair_mask]), p=2, dim=-1
        )  # (num_valid_point_pairs_in_batch, 3)
        assert batch_image_pair_idx.shape == (num_valid_point_pairs_in_batch,)
        assert batch_fundamental.shape == (num_valid_point_pairs_in_batch, 3, 3)
        assert batch_xy_homo1.shape == (num_valid_point_pairs_in_batch, 3)
        assert batch_xy_homo2.shape == (num_valid_point_pairs_in_batch, 3)
        del batch_data, batch_point_pair_mask, num_valid_point_pairs_in_batch

        # compute the linear coefficients
        batch_w = torch.einsum(
            "bi,bj->bij", batch_xy_homo2, batch_xy_homo1
        )  # (num_valid_point_pairs_in_batch, 3, 3)
        batch_w = batch_w.reshape(-1, 9)  # (num_valid_point_pairs_in_batch, 9)

        # compute w w^T
        batch_W = torch.einsum(
            "bi,bj->bij", batch_w, batch_w
        )  # (num_valid_point_pairs_in_batch, 9, 9)

        # weight
        batch_error = (
            (batch_w * batch_fundamental.view(-1, 9)).sum(dim=-1).abs()
        )  # (num_valid_point_pairs_in_batch,)
        epsilon = 1e-4
        batch_weights = 1.0 / (
            batch_error + epsilon
        )  # (num_valid_point_pairs_in_batch,)
        batch_W = (
            batch_W * batch_weights[..., None, None]
        )  # (num_valid_point_pairs_in_batch, 9, 9)
        del batch_fundamental, batch_error, batch_weights, epsilon

        # convert to float64
        batch_W = batch_W.to(torch.float64)  # (num_valid_point_pairs_in_batch, 9)

        # accumulate
        W.scatter_reduce_(
            dim=0,
            index=batch_image_pair_idx[:, None, None].expand(-1, 9, 9),
            src=batch_W,
            reduce="sum",
        )  # (num_image_pairs, 9, 9)

    # scale by the number of point pairs
    W /= point_pair_mask.to(W).sum()  # (num_image_pairs, 9, 9)

    # convert to the desired precision
    W = W.to(precision)  # (num_image_pairs, 9, 9)

    ##### Return #####
    return W


# @triton.jit
# def triton_kernel(
#     out_R_rel_ptr,
#     image_camera_indices_ptr,
#     R_w2c_ptr,
#     t_w2c_ptr,
#     inv_focal_scale_ptr,
#     num_rows,
# ):
#     # assume all the tensors to be contiguous
#     # starting row of the program
#     row_start = tl.program_id(0)
#     row_step = tl.num_programs(0)
#     for row_idx in tl.range(row_start, num_rows, row_step, num_stages=0):  # type: ignore
#         # load image and camera indices
#         idx_row_start_ptr = image_camera_indices_ptr + row_idx * 4  # long ptr
#         image_idx1_ptr = idx_row_start_ptr + 0  # long ptr
#         image_idx2_ptr = idx_row_start_ptr + 1  # long ptr
#         camera_idx1_ptr = idx_row_start_ptr + 2  # long ptr
#         camera_idx2_ptr = idx_row_start_ptr + 3  # long ptr
#         image_idx1 = tl.load(image_idx1_ptr)  # long
#         image_idx2 = tl.load(image_idx2_ptr)  # long
#         camera_idx1 = tl.load(camera_idx1_ptr)  # long
#         camera_idx2 = tl.load(camera_idx2_ptr)  # long
#
#         # load R_w2c, t_w2c (note that arange only works for power of 2)
#         R_row_offset = tl.arange(0, 4)[:, None]  # long (4, 1)
#         R_col_offset = tl.arange(0, 4)[None, :]  # long (1, 4)
#         R_offset = R_row_offset * 3 + R_col_offset  # long (4, 4)
#         R_mask = (R_col_offset < 3) & (R_row_offset < 3)  # long (4, 4)
#         t_offset = tl.arange(0, 4)  # long (4,)
#         t_mask = t_offset < 3  # long (4,)
#         R1_ptrs = R_w2c_ptr + image_idx1 * 9 + R_offset  # float (4, 4)
#         R2_ptrs = R_w2c_ptr + image_idx2 * 9 + R_offset  # float (4, 4)
#         t1_ptrs = t_w2c_ptr + image_idx1 * 3 + t_offset  # float (4,)
#         t2_ptrs = t_w2c_ptr + image_idx2 * 3 + t_offset  # float (4,)
#         R1 = tl.load(R1_ptrs, mask=R_mask, other=0.0)  # float (4, 4)
#         R2 = tl.load(R2_ptrs, mask=R_mask, other=0.0)  # float (4, 4)
#         t1 = tl.load(t1_ptrs, mask=t_mask, other=0.0)  # float (4,)
#         t2 = tl.load(t2_ptrs, mask=t_mask, other=0.0)  # float (4,)
#
#         # compute relative rotation
#         R_rel = tl.dot(R2, tl.trans(R1, 0, 1))  # float (4, 4)
#
#         # store
#         R_rel_ptrs = out_R_rel_ptr + row_idx * 9 + R_offset  # float (4, 4)
#         tl.store(R_rel_ptrs, R_rel, mask=R_mask)
#
#
# def triton_wrapper(
#     image_camera_indices: torch.Tensor,
#     R_w2c: torch.Tensor,
#     t_w2c: torch.Tensor,
#     inv_focal_scale: torch.Tensor,
#     W: torch.Tensor,
# ):
#     # get information
#     device, dtype = R_w2c.device, R_w2c.dtype
#     num_image_pairs = image_camera_indices.shape[0]
#     num_rows = num_image_pairs
#
#     # allocate output tensors
#     R_rel = torch.zeros(
#         (num_image_pairs, 3, 3), device=device, dtype=dtype
#     )  # (num_image_pairs, 3, 3)
#
#     # hyperparameters to tune
#     num_warps = 1  # debug: where is this used?
#     num_programs = min(num_rows, 4096)  # number of programs to launch
#
#     # launch
#     triton_kernel[(num_programs, 1, 1)](
#         R_rel,
#         image_camera_indices,
#         R_w2c,
#         t_w2c,
#         inv_focal_scale,
#         num_rows,
#     )
#
#     # return the result
#     return R_rel


def compute_gradients(
    image_camera_indices: torch.Tensor,  # long (B, 4)
    R_w2c: torch.Tensor,  # float (N, 3, 3)
    t_w2c: torch.Tensor,  # float (N, 3)
    inv_focal_scale: torch.Tensor,  # float (C,)
    W: torch.Tensor,  # float (B, 9, 9)
):
    # make sure everything is contiguous
    assert image_camera_indices.is_contiguous()
    assert R_w2c.is_contiguous()
    assert t_w2c.is_contiguous()
    assert inv_focal_scale.is_contiguous()
    assert W.is_contiguous()

    # unbind image and camera indices
    image_idx1, image_idx2, camera_idx1, camera_idx2 = image_camera_indices.unbind(
        -1
    )  # (B,), (B,), (B,), (B,)

    # debug: try triton kernel
    # R_rel = triton_wrapper(
    #     image_camera_indices=image_camera_indices,  # (B, 4)
    #     R_w2c=R_w2c,  # (N, 3, 3)
    #     t_w2c=t_w2c,  # (N, 3)
    #     inv_focal_scale=inv_focal_scale,  # (C,)
    #     W=W,  # (B, 9, 9)
    # )  # (B, 3, 3)

    # ------------------------------------------------------------------ #
    # Layer-1: gather poses & relative rotation
    # ------------------------------------------------------------------ #
    R1 = R_w2c.index_select(0, image_idx1)  # (B,3,3)
    R2 = R_w2c.index_select(0, image_idx2)  # (B,3,3)
    t1 = t_w2c.index_select(0, image_idx1)  # (B,3)
    t2 = t_w2c.index_select(0, image_idx2)  # (B,3)
    R_rel = R2 @ R1.transpose(-1, -2)  # (B,3,3)

    # ------------------------------------------------------------------ #
    # Layer-2: essential matrix
    # ------------------------------------------------------------------ #
    t1_x = vector_to_skew_symmetric_matrix(t1)  # (B,3,3)
    t2_x = vector_to_skew_symmetric_matrix(t2)  # (B,3,3)
    essential = R_rel @ t1_x - t2_x @ R_rel  # (B,3,3)

    # ------------------------------------------------------------------ #
    # Layer-3: fundamental matrix (unnormalised)
    # ------------------------------------------------------------------ #
    f1_inv = inv_focal_scale[camera_idx1]  # (B,)
    f2_inv = inv_focal_scale[camera_idx2]  # (B,)
    K1_inv = torch.stack((f1_inv, f1_inv, torch.ones_like(f1_inv)), dim=-1)  # (B,3)
    K2_inv = torch.stack((f2_inv, f2_inv, torch.ones_like(f2_inv)), dim=-1)  # (B,3)
    fundamental = K2_inv[:, :, None] * essential * K1_inv[:, None, :]  # (B,3,3)

    # ------------------------------------------------------------------ #
    # Layer-4: ℓ2-normalise the 9-vector
    # ------------------------------------------------------------------ #
    F_flat = fundamental.reshape(-1, 9)  # (B,9)
    F_norm = F_flat.norm(dim=-1, keepdim=True) + 1e-8  # (B,1)
    F_normalised = F_flat / F_norm  # (B,9)

    # ------------------------------------------------------------------ #
    # Layer-5: quadratic loss
    # ------------------------------------------------------------------ #
    W_vec = torch.einsum("bij,bj->bi", W, F_normalised)  # (B,9)
    loss = 0.5 * (F_normalised * W_vec).sum()  # scalar

    # -------------------------------------------------------------- #
    # ⇢ Layer-5
    # -------------------------------------------------------------- #
    d_vec = W_vec  # (B,9)

    # -------------------------------------------------------------- #
    # ⇢ Layer-4
    # -------------------------------------------------------------- #
    d_F_flat = (
        d_vec - (F_normalised * d_vec).sum(dim=-1, keepdim=True) * F_normalised
    ) / F_norm  # (B,9)
    d_F = d_F_flat.view(-1, 3, 3)  # (B,3,3)

    # -------------------------------------------------------------- #
    # ⇢ Layer-3
    # -------------------------------------------------------------- #
    d_E = d_F * K2_inv[:, :, None] * K1_inv[:, None, :]  # (B,3,3)
    d_K1_inv = d_F * essential * K2_inv[:, :, None]  # (B,3,3)
    d_K2_inv = d_F * essential * K1_inv[:, None, :]  # (B,3,3)

    d_f1_inv = d_K1_inv[:, :, :2].sum((-1, -2))  # (B,)
    d_f2_inv = d_K2_inv[:, :2, :].sum((-1, -2))  # (B,)

    num_cam = inv_focal_scale.shape[0]
    d_f_inv = torch.zeros(
        (num_cam,), device=inv_focal_scale.device, dtype=inv_focal_scale.dtype
    )  # (C,)
    d_f_inv.scatter_reduce_(0, camera_idx1, d_f1_inv, reduce="sum", include_self=True)
    d_f_inv.scatter_reduce_(0, camera_idx2, d_f2_inv, reduce="sum", include_self=True)

    # -------------------------------------------------------------- #
    # ⇢ Layer-2
    # -------------------------------------------------------------- #
    d_R_rel = d_E @ t1_x.transpose(-1, -2) - t2_x.transpose(-1, -2) @ d_E  # (B,3,3)
    d_t1_x = R_rel.transpose(-1, -2) @ d_E  # (B,3,3)
    d_t2_x = -d_E @ R_rel.transpose(-1, -2)  # (B,3,3)

    d_t1 = torch.stack(  # (B,3)
        (
            d_t1_x[:, 2, 1] - d_t1_x[:, 1, 2],
            d_t1_x[:, 0, 2] - d_t1_x[:, 2, 0],
            d_t1_x[:, 1, 0] - d_t1_x[:, 0, 1],
        ),
        dim=-1,
    )
    d_t2 = torch.stack(  # (B,3)
        (
            d_t2_x[:, 2, 1] - d_t2_x[:, 1, 2],
            d_t2_x[:, 0, 2] - d_t2_x[:, 2, 0],
            d_t2_x[:, 1, 0] - d_t2_x[:, 0, 1],
        ),
        dim=-1,
    )

    # -------------------------------------------------------------- #
    # ⇢ Layer-1
    # -------------------------------------------------------------- #
    d_R1 = d_R_rel.transpose(-1, -2) @ R2  # (B,3,3)
    d_R2 = d_R_rel @ R1  # (B,3,3)

    N = len(R_w2c)  # number of images
    d_R_w2c = torch.zeros((N, 3, 3), device=R_w2c.device, dtype=R_w2c.dtype)  # (N,3,3)
    d_t_w2c = torch.zeros((N, 3), device=t_w2c.device, dtype=t_w2c.dtype)  # (N,3)

    d_R_w2c.scatter_reduce_(
        0,
        image_idx1[:, None, None].expand(-1, 3, 3),
        d_R1,
        reduce="sum",
        include_self=True,
    )
    d_R_w2c.scatter_reduce_(
        0,
        image_idx2[:, None, None].expand(-1, 3, 3),
        d_R2,
        reduce="sum",
        include_self=True,
    )
    d_t_w2c.scatter_reduce_(
        0,
        image_idx1[:, None].expand(-1, 3),
        d_t1,
        reduce="sum",
        include_self=True,
    )
    d_t_w2c.scatter_reduce_(
        0,
        image_idx2[:, None].expand(-1, 3),
        d_t2,
        reduce="sum",
        include_self=True,
    )

    # -------------------------------------------------------------- #
    # Return grads in input order
    # -------------------------------------------------------------- #
    return (
        loss,
        d_R_w2c,  # R_w2c
        d_t_w2c,  # t_w2c
        d_f_inv,  # inv_focal_scale
    )


def _compute_fundamental_matrix(
    image_idx1: torch.Tensor,
    image_idx2: torch.Tensor,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    focal_scale: torch.Tensor,
    camera_idx: torch.Tensor,
):
    """Compute the fundamental matrix for all image pairs.
    Args:
        image_idx1: torch.Tensor long (num_image_pairs,), the first image idx for each pair
        image_idx2: torch.Tensor long (num_image_pairs,), the second image idx for each pair
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation matrices for each image
        t_w2c: torch.Tensor float (num_images, 3), w2c global translation vectors for each image
        focal_scale: torch.Tensor float (num_cameras,), the scale factor on focal lengths
        camera_idx: torch.Tensor long (num_images,), the camera idx for each image
    Returns:
        fundamental: torch.Tensor float (num_image_pairs, 3, 3), the fundamental matrix for each image pair
    """
    # compute relative rotation
    R_w2c1 = torch.index_select(
        input=R_w2c, dim=0, index=image_idx1
    )  # (num_image_pairs, 3, 3)
    R_w2c2 = torch.index_select(
        input=R_w2c, dim=0, index=image_idx2
    )  # (num_image_pairs, 3, 3)
    R = R_w2c2 @ R_w2c1.transpose(-1, -2)  # (num_image_pairs, 3, 3)
    del R_w2c, R_w2c1, R_w2c2

    # get relative translation
    t_w2c1 = torch.index_select(
        input=t_w2c, dim=0, index=image_idx1
    )  # (num_image_pairs, 3)
    t_w2c2 = torch.index_select(
        input=t_w2c, dim=0, index=image_idx2
    )  # (num_image_pairs, 3)
    del t_w2c

    # compute essential matrix
    essential = torch.cross(R, t_w2c1[..., None, :], dim=-1) - torch.cross(
        t_w2c2[..., None], R, dim=-2
    )

    # normalize essential matrix
    essential = normalize_matrix(essential)  # (num_image_pairs, 3, 3)

    # compute fundamental matrix
    focal_scale1_inv = 1.0 / focal_scale[camera_idx[image_idx1]]  # (num_image_pairs,)
    focal_scale2_inv = 1.0 / focal_scale[camera_idx[image_idx2]]  # (num_image_pairs,)
    K1_inv_diag = torch.stack(
        [focal_scale1_inv, focal_scale1_inv, torch.ones_like(focal_scale1_inv)],
        dim=-1,
    )  # (num_image_pairs, 3)
    K2_inv_diag = torch.stack(
        [focal_scale2_inv, focal_scale2_inv, torch.ones_like(focal_scale2_inv)],
        dim=-1,
    )  # (num_image_pairs, 3)
    fundamental = (
        K2_inv_diag[:, :, None] * essential * K1_inv_diag[:, None, :]
    )  # (num_image_pairs, 3, 3)

    # return
    return fundamental


@torch.no_grad()
def _all_error(
    point_pairs: PointPairs,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    focal_scale: torch.Tensor,
    camera_idx: torch.Tensor,
    image_mask: torch.Tensor,
):
    """Compute the epipolar error for all point pairs.
    Args:
        point_pairs: PointPairs container
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation matrices for each image
        t_w2c: torch.Tensor float (num_images, 3), w2c global translation vectors for each image
        focal_scale: torch.Tensor float (num_cameras,), the scale factor on focal lengths
        camera_idx: torch.Tensor long (num_images,), the camera idx for each image
        image_mask: torch.Tensor bool (num_images,), the mask indicating the valid images
    Returns:
        error: torch.Tensor float (num_point_pairs,), the epipolar error for each point pair (inf for point pairs involving invalid images)
    """
    # get information
    device, dtype = R_w2c.device, R_w2c.dtype
    num_point_pairs = point_pairs.num_point_pairs

    # initialize error
    error = torch.nan + torch.zeros(
        num_point_pairs, device=device, dtype=dtype
    )  # (num_point_pairs,)

    # loop over batches
    for batch_data in point_pairs.query():
        # compute the fundamental matrix
        batch_fundamental = _compute_fundamental_matrix(
            image_idx1=batch_data.image_idx1,
            image_idx2=batch_data.image_idx2,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            focal_scale=focal_scale,
            camera_idx=camera_idx,
        )  # (B, 3, 3)

        # compute the error
        batch_xy_homo1 = F.normalize(
            to_homogeneous(batch_data.xy1), p=2, dim=-1
        )  # (B, 3)
        batch_xy_homo2 = F.normalize(
            to_homogeneous(batch_data.xy2), p=2, dim=-1
        )  # (B, 3)
        batch_error = torch.einsum(
            "bi,bij,bj->b", batch_xy_homo2, batch_fundamental, batch_xy_homo1
        ).abs()  # (B,)
        error[batch_data.point_pair_idx] = batch_error

    # set errors involving invalid images to inf
    error[~image_mask[point_pairs.image_idx[point_pairs.point_idx1]]] = torch.inf
    error[~image_mask[point_pairs.image_idx[point_pairs.point_idx2]]] = torch.inf

    # make sure there is no nan
    assert not torch.any(torch.isnan(error))

    # return
    return error


@torch.no_grad()
def loop(
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    focal_scale: torch.Tensor,
    point_pairs: PointPairs,
    point_pair_mask: torch.Tensor,
    camera_idx: torch.Tensor,
    lr: float = 0.0001,
    precision: torch.dtype = torch.float32,
    log_interval: int = 500,
):
    """Global epipolar adjustment loop.

    Args:
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation matrices for each image
        t_w2c: torch.Tensor float (num_images, 3), w2c global translation vectors for each image
        focal_scale: torch.Tensor float (num_cameras,), the scale factor on focal lengths
        point_pairs: PointPairs container
        point_pair_mask: torch.Tensor bool (num_point_pairs,), the mask indicating the inlier point pairs
        camera_idx: torch.Tensor long (num_images,), the camera idx for each image
        lr: float, the learning rate for the optimization.
        precision: torch.dtype, the precision for the optimization.
        log_interval: int, the log interval in number of iterations.

    Returns:
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation
        t_w2c: torch.Tensor float (num_images, 3), w2c global translation
        focal_scale: torch.Tensor float (num_cameras,), the optimized scale factor on focal lengths.
    """
    ##### Make sure the precision is valid #####
    assert precision in [torch.float32, torch.float64]

    ##### Get original dtype #####
    orig_dtype = R_w2c.dtype

    ##### Find all image pairs with a non-empty set of inliers #####
    # get point pair idx
    unique_image_idx, _inverse_idx = torch.unique(
        torch.stack(
            [
                point_pairs.image_idx[point_pairs.point_idx1][point_pair_mask],
                point_pairs.image_idx[point_pairs.point_idx2][point_pair_mask],
            ],
            dim=-1,
        ),
        dim=0,
        return_inverse=True,
    )  # (num_image_pairs, 2), (num_valid_point_pairs,)
    image_idx1, image_idx2 = unique_image_idx.unbind(
        -1
    )  # (num_image_pairs,), (num_image_pairs,)
    image_pair_idx = 209347298473 + torch.zeros(
        point_pairs.num_point_pairs, device=point_pairs.device, dtype=torch.long
    )  # (num_point_pairs,) use a large number to indicate invalid
    image_pair_idx[point_pair_mask] = _inverse_idx  # (num_point_pairs,)
    del unique_image_idx, _inverse_idx

    # get number of image pairs
    num_image_pairs = image_idx1.shape[0]

    ##### Compute the quadratic form #####
    # compute the initial fundamental matrix
    initial_fundamental = _compute_fundamental_matrix(
        image_idx1=image_idx1,
        image_idx2=image_idx2,
        R_w2c=R_w2c,
        t_w2c=t_w2c,
        focal_scale=focal_scale,
        camera_idx=camera_idx,
    )  # (num_image_pairs, 3, 3)

    # compute the weighted quadratic form
    W = quadratic_form(
        num_image_pairs=num_image_pairs,
        point_pairs=point_pairs,
        image_pair_idx=image_pair_idx,
        prev_fundamental=initial_fundamental,
        point_pair_mask=point_pair_mask,
        precision=precision,
    )  # (num_image_pairs, 9, 9)

    # prevent misuse
    del initial_fundamental

    ##### Compose image camera indices #####
    camera_idx1 = camera_idx[image_idx1]  # (num_image_pairs,)
    camera_idx2 = camera_idx[image_idx2]  # (num_image_pairs,)
    image_camera_indices = torch.stack(
        [
            image_idx1,
            image_idx2,
            camera_idx1,
            camera_idx2,
        ],
        dim=-1,
    )  # (num_image_pairs, 4)
    del image_idx1, image_idx2, camera_idx1, camera_idx2, camera_idx

    ##### Initialize parameters for optimization #####
    params = EpipolarAdjustmentParameters(
        R_w2c=R_w2c, t_w2c=t_w2c, focal_scale=focal_scale, precision=precision
    )
    del R_w2c, t_w2c, focal_scale

    ##### Optimizer and convergence manager #####
    # optimizer
    optimizer = torch.optim.Adam(params.parameters(), lr=lr)

    # convergence manager
    convergence_manager = ConvergenceManager(
        warmup_steps=10,
        decay=0.0,
        convergence_window=100,
    )
    convergence_manager.start()

    # computation module
    # computation_module = ComputationModule()

    ##### Optimization loop #####
    with torch.enable_grad():
        for iter_idx in range(1000000000):
            (
                R_w2c,
                t_w2c,
                _,  # focal_scale
                inv_focal_scale,
            ) = params()  # (num_images, 3, 3), (num_images, 3), (num_cameras,)

            # compute the loss
            # loss = computation_module(
            #     image_idx1=image_idx1,
            #     image_idx2=image_idx2,
            #     R_w2c=R_w2c,
            #     t_w2c=t_w2c,
            #     focal_scale=focal_scale,
            #     camera_idx=camera_idx,
            #     W=W,
            # )
            loss, d_R_w2c, d_t_w2c, d_inv_focal_scale = compute_gradients(
                image_camera_indices=image_camera_indices,  # (num_image_pairs, 4)
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                inv_focal_scale=inv_focal_scale,  # (num_cameras,)
                W=W,
            )  # scalar, (num_images, 3, 3), (num_images, 3), (num_cameras,)

            # backprop
            optimizer.zero_grad()
            # loss.backward()

            # backward
            torch.autograd.backward(
                tensors=[R_w2c, t_w2c, inv_focal_scale],
                grad_tensors=[d_R_w2c, d_t_w2c, d_inv_focal_scale],
            )

            # step
            optimizer.step()

            # # compute the fundamental matrix
            # fundamental = _compute_fundamental_matrix(
            #     image_idx1=image_idx1,
            #     image_idx2=image_idx2,
            #     R_w2c=R_w2c,
            #     t_w2c=t_w2c,
            #     focal_scale=focal_scale,
            #     camera_idx=camera_idx,
            # )  # (num_image_pairs, 3, 3)
            #
            # # flatten the fundamental matrix
            # fundamental = fundamental.reshape(
            #     num_image_pairs, 9
            # )  # (num_image_pairs, 9)
            #
            # # compute the loss
            # loss = 0.5 * torch.einsum("bi,bij,bj->b", fundamental, W, fundamental).sum()

            # check convergence
            moving_loss, if_converged = convergence_manager.step(
                step=iter_idx, loss=loss
            )
            if if_converged:
                logger.info(
                    f"Converged at iteration {iter_idx+1} with moving loss {moving_loss:.8f}"
                )
                break

            # log
            if iter_idx % log_interval == 0:
                logger.info(
                    f"[Iter {iter_idx} ({precision})] loss={loss.item():.8f}, moving_loss={moving_loss:.8f}"
                )

    ##### Get the results and convert to the original dtype #####
    (
        R_w2c,
        t_w2c,
        focal_scale,
        _,  # inv_focal_scale
    ) = params()  # (num_images, 3, 3), (num_images, 3), (num_cameras,)
    if isinstance(R_w2c, nn.parameter.Parameter):
        R_w2c = R_w2c.data
    if isinstance(t_w2c, nn.parameter.Parameter):
        t_w2c = t_w2c.data
    if isinstance(focal_scale, nn.parameter.Parameter):
        focal_scale = focal_scale.data
    R_w2c = R_w2c.to(orig_dtype)
    t_w2c = t_w2c.to(orig_dtype)
    focal_scale = focal_scale.to(orig_dtype)

    ##### Return #####
    return R_w2c, t_w2c, focal_scale


@torch.no_grad()
def epipolar_adjustment(
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    point_pairs: PointPairs,
    point_pair_mask: torch.Tensor,
    images: Images,
    cameras: Cameras,
    num_irls_steps: int = 3,
    num_prune_steps: int = 3,
    max_thr: float = 0.01,
    min_thr: float = 0.005,
    lr: float = 1e-4,
    lr_decay: float = 0.5,
    log_interval: int = 500,
):
    """Globally optimize the epipolar error for all image pairs.

    Args:
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation matrices for each image
        t_w2c: torch.Tensor float (num_images, 3), w2c global translation vectors for each image
        point_pairs: PointPairs container
        point_pair_mask: torch.Tensor bool (num_point_pairs,), the mask indicating the inlier point pairs
        images: Images container
        cameras: Cameras container
        num_irls_steps: int, the number of IRLS steps for each round
        num_prune_steps: int, the number of pruning steps
        max_thr: float, the maximum threshold for pruning
        min_thr: float, the minimum threshold for pruning
        lr: float, the learning rate for the optimization.
        lr_decay: float, the learning rate decay for each pruning step.
        log_interval: int, the log interval in number of iterations.

    Returns:
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation
        t_w2c: torch.Tensor float (num_images, num_images, 3), w2c global translation
        focal_scale: torch.Tensor float (num_cameras,), the optimized scale factors on focal lengths.
        point_pair_mask: torch.Tensor bool (num_point_pairs,), the final mask indicating the inlier point pairs
    """
    ##### Get some information #####
    dtype = R_w2c.dtype
    device = R_w2c.device
    assert num_prune_steps >= 0

    ##### Make sure the point pair mask only involves valid images #####
    assert torch.all(
        images.mask[point_pairs.image_idx[point_pairs.point_idx1][point_pair_mask]]
    )
    assert torch.all(
        images.mask[point_pairs.image_idx[point_pairs.point_idx2][point_pair_mask]]
    )

    ##### Get all the pruning thresholds #####
    thr_list = torch.linspace(
        min_thr, max_thr, num_prune_steps, device=device
    )  # (num_prune_steps,)
    thr_list = thr_list.flip(0)  # (num_prune_steps,) from large to small
    thr_list = thr_list.tolist()  # list

    ##### Initialize focal length #####
    focal_scale = torch.ones(
        cameras.num_cameras, device=device, dtype=dtype
    )  # (num_cameras,)

    ##### Optimize #####
    for i in range(num_prune_steps + 1):
        with timer(f"Round {i}"):
            # optimization loops
            for j in range(num_irls_steps):

                # IRLS steps
                with timer(f"IRLS Iter {j}"):
                    logger.info(
                        f"[Round {i+1} / {num_prune_steps+1}] Starting IRLS step {j+1} / {num_irls_steps}..."
                    )
                    R_w2c, t_w2c, focal_scale = loop(
                        R_w2c=R_w2c,
                        t_w2c=t_w2c,
                        focal_scale=focal_scale,
                        point_pairs=point_pairs,
                        point_pair_mask=point_pair_mask,
                        camera_idx=cameras.camera_idx,
                        lr=lr,
                        precision=torch.float32,
                        log_interval=log_interval,
                    )

            # prune point pairs
            if i < num_prune_steps:
                thr = thr_list[i]
                error = _all_error(
                    point_pairs=point_pairs,
                    R_w2c=R_w2c,
                    t_w2c=t_w2c,
                    focal_scale=focal_scale,
                    camera_idx=cameras.camera_idx,
                    image_mask=images.mask,
                )  # (num_point_pairs,)
                point_pair_mask = error < thr  # (num_point_pairs,)
                del error

                # make sure only valid images are used
                point_pair_mask &= images.mask[
                    point_pairs.image_idx[point_pairs.point_idx1]
                ]
                point_pair_mask &= images.mask[
                    point_pairs.image_idx[point_pairs.point_idx2]
                ]

                # log
                logger.info(
                    f"Pruned {(~point_pair_mask).long().sum().item()} / {point_pair_mask.shape[0]} point pairs with threshold {thr}"
                )

                # decay learning rate
                lr *= lr_decay
                logger.info(f"Decayed learning rate to {lr}")

    ##### Return #####
    return R_w2c, t_w2c, focal_scale, point_pair_mask
