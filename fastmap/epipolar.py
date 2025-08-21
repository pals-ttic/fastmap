import os
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


class TorchComputeGradientModule(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.set_float32_matmul_precision("high")
        pass

    @torch.no_grad()
    def forward(
        self,
        R1: torch.Tensor,  # float (B, 3, 3)
        R2: torch.Tensor,  # float (B, 3, 3)
        t1: torch.Tensor,  # float (B, 3)
        t2: torch.Tensor,  # float (B, 3)
        f1_inv: torch.Tensor,  # float (B,)
        f2_inv: torch.Tensor,  # float (B,)
        W: torch.Tensor,  # float (B, 9, 9)
    ):
        # make sure everything is contiguous
        assert R1.is_contiguous()
        assert R2.is_contiguous()
        assert t1.is_contiguous()
        assert t2.is_contiguous()
        assert f1_inv.is_contiguous()
        assert f2_inv.is_contiguous()
        assert W.is_contiguous()

        # ------------------------------------------------------------------ #
        # Layer-1: gather poses & relative rotation
        # ------------------------------------------------------------------ #
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

        # -------------------------------------------------------------- #
        # ⇢ Layer-2
        # -------------------------------------------------------------- #
        d_R_rel = (
            d_E @ t1_x.transpose(-1, -2) - t2_x.transpose(-1, -2).contiguous() @ d_E
        )  # (B,3,3)
        d_t1_x = R_rel.transpose(-1, -2).contiguous() @ d_E  # (B,3,3)
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
        d_R1 = d_R_rel.transpose(-1, -2).contiguous() @ R2  # (B,3,3)
        d_R2 = d_R_rel @ R1  # (B,3,3)

        # -------------------------------------------------------------- #
        # Return grads in input order
        # -------------------------------------------------------------- #
        return (
            loss,
            d_R1,  # R_w2c
            d_R2,  # R_w2c
            d_t1,  # t_w2c
            d_t2,  # t_w2c
            d_f1_inv,  # inv_focal_scale
            d_f2_inv,  # inv_focal_scale
        )


class CUDAComputeGradientModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._initialized = False
        from fastmap.cuda import epipolar_gradient

        self.gradient_fn = epipolar_gradient

    @torch.no_grad()
    def forward(
        self,
        R1: torch.Tensor,  # float (B, 3, 3)
        R2: torch.Tensor,  # float (B, 3, 3)
        t1: torch.Tensor,  # float (B, 3)
        t2: torch.Tensor,  # float (B, 3)
        f1_inv: torch.Tensor,  # float (B,)
        f2_inv: torch.Tensor,  # float (B,)
        W: torch.Tensor,  # float (B, 9, 9)
    ):
        if not self._initialized:
            # make sure everything is contiguous
            assert R1.is_contiguous()
            assert R2.is_contiguous()
            assert t1.is_contiguous()
            assert t2.is_contiguous()
            assert f1_inv.is_contiguous()
            assert f2_inv.is_contiguous()
            assert W.is_contiguous()

            # get device and dtype
            device = R1.device
            dtype = R1.dtype

            # initialize the output tensors
            self.loss = torch.zeros((1,), device=device, dtype=dtype)  # scalar
            self.d_R1 = torch.zeros_like(R1)  # (B,3,3)
            self.d_R2 = torch.zeros_like(R2)  # (B,3,3)
            self.d_t1 = torch.zeros_like(t1)  # (B,3)
            self.d_t2 = torch.zeros_like(t2)  # (B,3)
            self.d_f1_inv = torch.zeros_like(f1_inv)  # (B,)
            self.d_f2_inv = torch.zeros_like(f2_inv)  # (B,)

            # initialize the buffers
            self.buffer_R_rel = torch.zeros_like(R1)  # (B,3,3)
            self.buffer_t1_x = torch.zeros_like(R1)  # (B,3,3)
            self.buffer_t2_x = torch.zeros_like(R1)  # (B,3,3)
            self.buffer_essential = torch.zeros_like(R1)  # (B,3,3)
            self.buffer_fundamental = torch.zeros_like(R1)  # (B,3,3)

            # set flag
            self._initialized = True

        self.gradient_fn(
            R1=R1,
            R2=R2,
            t1=t1,
            t2=t2,
            f1_inv=f1_inv,
            f2_inv=f2_inv,
            W=W,
            loss=self.loss,
            d_R1=self.d_R1,
            d_R2=self.d_R2,
            d_t1=self.d_t1,
            d_t2=self.d_t2,
            d_f1_inv=self.d_f1_inv,
            d_f2_inv=self.d_f2_inv,
            buffer_R_rel=self.buffer_R_rel,
            buffer_t1_x=self.buffer_t1_x,
            buffer_t2_x=self.buffer_t2_x,
            buffer_essential=self.buffer_essential,
            buffer_fundamental=self.buffer_fundamental,
        )

        return (
            self.loss,
            self.d_R1,  # R_w2c
            self.d_R2,  # R_w2c
            self.d_t1,  # t_w2c
            self.d_t2,  # t_w2c
            self.d_f1_inv,  # inv_focal_scale
            self.d_f2_inv,  # inv_focal_scale
        )


class ComputeGradient:
    def __init__(self):
        try:
            self.compute_gradients = CUDAComputeGradientModule()
        except ImportError:
            logger.warning(
                "CUDA kernel extension for epipolar adjustment is not available, falling back to the slower PyTorch implementation."
            )
            self.compute_gradients = TorchComputeGradientModule()

    def __call__(
        self,
        image_idx1,
        image_idx2,
        camera_idx1,
        camera_idx2,
        R_w2c,
        t_w2c,
        inv_focal_scale,
        W,
    ):
        R1 = R_w2c.index_select(0, image_idx1)  # (B,3,3)
        R2 = R_w2c.index_select(0, image_idx2)  # (B,3,3)
        t1 = t_w2c.index_select(0, image_idx1)  # (B,3)
        t2 = t_w2c.index_select(0, image_idx2)  # (B,3)
        f1_inv = inv_focal_scale[camera_idx1]  # (B,)
        f2_inv = inv_focal_scale[camera_idx2]  # (B,)

        loss, d_R1, d_R2, d_t1, d_t2, d_f1_inv, d_f2_inv = self.compute_gradients(
            R1=R1,
            R2=R2,
            t1=t1,
            t2=t2,
            f1_inv=f1_inv,
            f2_inv=f2_inv,
            W=W,
        )  # scalar, (num_images, 3, 3), (num_images, 3), (num_cameras,)

        num_cam = inv_focal_scale.shape[0]

        if num_cam == 1:
            # If there is only one camera, we can directly sum the gradients
            d_f_inv = d_f1_inv.sum() + d_f2_inv.sum()
            d_f_inv = d_f_inv.view(1)  # (1,)
        else:
            d_f_inv = torch.zeros(
                (num_cam,),
                device=inv_focal_scale.device,
                dtype=inv_focal_scale.dtype,
            )  # (C,)
            d_f_inv.scatter_reduce_(
                0, camera_idx1, d_f1_inv, reduce="sum", include_self=True
            )
            d_f_inv.scatter_reduce_(
                0, camera_idx2, d_f2_inv, reduce="sum", include_self=True
            )
        d_inv_focal_scale = d_f_inv  # (num_cameras,)

        N = len(R_w2c)  # number of images
        d_R_w2c = torch.zeros(
            (N, 3, 3), device=R_w2c.device, dtype=R_w2c.dtype
        )  # (N,3,3)
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

        # return the gradients in the order of inputs
        return (
            loss,  # scalar
            d_R_w2c,  # R_w2c
            d_t_w2c,  # t_w2c
            d_inv_focal_scale,  # inv_focal_scale
        )  # (num_cameras,)


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
    R_rel = R_w2c2 @ R_w2c1.transpose(-1, -2).contiguous()  # (num_image_pairs, 3, 3)
    del R_w2c, R_w2c1, R_w2c2

    # get relative translation
    t1 = torch.index_select(
        input=t_w2c, dim=0, index=image_idx1
    )  # (num_image_pairs, 3)
    t2 = torch.index_select(
        input=t_w2c, dim=0, index=image_idx2
    )  # (num_image_pairs, 3)
    del t_w2c

    t1_x = vector_to_skew_symmetric_matrix(t1)  # (B,3,3)
    t2_x = vector_to_skew_symmetric_matrix(t2)  # (B,3,3)
    essential = R_rel @ t1_x - t2_x @ R_rel  # (B,3,3)

    camera_idx1 = camera_idx[image_idx1]  # (B,)
    camera_idx2 = camera_idx[image_idx2]  # (B,)
    f1_inv = 1.0 / focal_scale[camera_idx1]  # (B,)
    f2_inv = 1.0 / focal_scale[camera_idx2]  # (B,)
    K1_inv = torch.stack((f1_inv, f1_inv, torch.ones_like(f1_inv)), dim=-1)  # (B,3)
    K2_inv = torch.stack((f2_inv, f2_inv, torch.ones_like(f2_inv)), dim=-1)  # (B,3)
    fundamental = K2_inv[:, :, None] * essential * K1_inv[:, None, :]  # (B,3,3)

    F_flat = fundamental.reshape(-1, 9)  # (B,9)
    F_norm = F_flat.norm(dim=-1, keepdim=True) + 1e-8  # (B,1)
    F_normalised = F_flat / F_norm  # (B,9)

    # reshape to (num_image_pairs, 3, 3)
    fundamental = F_normalised.view(-1, 3, 3)  # (num_image_pairs, 3, 3)

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
    image_idx1: torch.Tensor,
    image_idx2: torch.Tensor,
    image_pair_mask: torch.Tensor,
    point_pairs: PointPairs,
    image_pair_idx: torch.Tensor,
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
        image_idx1: torch.Tensor long (num_image_pairs,), the first image idx for each image pair
        image_idx2: torch.Tensor long (num_image_pairs,), the second image idx for each image pair
        image_pair_mask: torch.Tensor bool (num_image_pairs,), the mask indicating the valid image pairs
        point_pairs: PointPairs container
        image_pair_idx: torch.Tensor long (num_point_pairs,), the image pair index for each point pair
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
        num_image_pairs=len(image_idx1),
        point_pairs=point_pairs,
        image_pair_idx=image_pair_idx,
        prev_fundamental=initial_fundamental,
        point_pair_mask=point_pair_mask,
        precision=precision,
    )  # (num_image_pairs, 9, 9)

    # mask out invalid image pairs
    if not image_pair_mask.all():
        W[~image_pair_mask] = 0.0  # (num_image_pairs, 9, 9)
        del image_pair_mask

    # prevent misuse
    del initial_fundamental

    ##### Compose image camera indices #####
    camera_idx1 = camera_idx[image_idx1]  # (num_image_pairs,)
    camera_idx2 = camera_idx[image_idx2]  # (num_image_pairs,)
    del camera_idx

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
    compute_gradients = ComputeGradient()

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
            loss, d_R_w2c, d_t_w2c, d_inv_focal_scale = compute_gradients(
                image_idx1=image_idx1,
                image_idx2=image_idx2,
                camera_idx1=camera_idx1,
                camera_idx2=camera_idx2,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                inv_focal_scale=inv_focal_scale,  # (num_cameras,)
                W=W,
            )  # scalar, (num_images, 3, 3), (num_images, 3), (num_cameras,)
            if isinstance(loss, torch.Tensor):
                loss = loss.item()  # convert to scalar

            # backprop
            optimizer.zero_grad()

            # backward
            torch.autograd.backward(
                tensors=[R_w2c, t_w2c, inv_focal_scale],
                grad_tensors=[d_R_w2c, d_t_w2c, d_inv_focal_scale],
            )

            # step
            optimizer.step()

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
                    f"[Iter {iter_idx} ({precision})] loss={loss:.8f}, moving_loss={moving_loss:.8f}"
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

    ##### Preserve the original point pair mask #####
    original_point_pair_mask = point_pair_mask.clone()  # (num_point_pairs,)

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

    # initialize image pair mask
    image_pair_mask = torch.ones(
        num_image_pairs, device=device, dtype=torch.bool
    )  # (num_image_pairs,)

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
                        image_idx1=image_idx1,
                        image_idx2=image_idx2,
                        image_pair_mask=image_pair_mask,
                        point_pairs=point_pairs,
                        image_pair_idx=image_pair_idx,
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
                point_pair_mask = original_point_pair_mask & (
                    error < thr
                )  # (num_point_pairs,)
                del error

                # make sure only valid images are used
                point_pair_mask &= images.mask[
                    point_pairs.image_idx[point_pairs.point_idx1]
                ]
                point_pair_mask &= images.mask[
                    point_pairs.image_idx[point_pairs.point_idx2]
                ]

                # update image pair mask
                point_pair_count = torch.zeros(
                    num_image_pairs, device=device, dtype=torch.long
                )  # (num_image_pairs,)
                point_pair_count.scatter_add_(
                    dim=0,
                    index=image_pair_idx[point_pair_mask],
                    src=torch.ones_like(
                        image_pair_idx[point_pair_mask], device=device, dtype=torch.long
                    ),
                )
                image_pair_mask = point_pair_count > 0
                del point_pair_count

                # log
                logger.info(
                    f"Pruned {(~point_pair_mask).long().sum().item()} / {point_pair_mask.shape[0]} point pairs with threshold {thr}"
                )

                # decay learning rate
                lr *= lr_decay
                logger.info(f"Decayed learning rate to {lr}")

    ##### Return #####
    return R_w2c, t_w2c, focal_scale, point_pair_mask
