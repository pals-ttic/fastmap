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
        self.focal_scale = nn.Parameter(
            focal_scale.clone().to(precision), requires_grad=True
        )  # (num_cameras,)

    def rotation_parameters(self):
        return [self.rot6d_w2c]

    def translation_parameters(self):
        return [self.t_w2c]

    def focal_parameters(self):
        return [self.focal]

    def forward(self):
        """Return the parameters after some processing"""
        # get the parameters
        rot6d_w2c = self.rot6d_w2c  # (num_images, 6)
        t_w2c = self.t_w2c  # (num_images, 3)
        focal_scale = self.focal_scale  # (num_cameras,)

        # convert the rotation to matrix
        R_w2c = rotation_6d_to_matrix(rot6d_w2c)  # (num_images, 3, 3)

        # return the results
        return R_w2c, t_w2c, focal_scale


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
    t = torch.einsum("bij,bj->bi", R, -t_w2c1) + t_w2c2  # (B, 3)
    t = F.normalize(t, p=2, dim=-1)  # (B, 3)
    del t_w2c, t_w2c1, t_w2c2

    # compute essential matrix
    essential = torch.cross(t[..., None], R, dim=-2)  # (num_image_pairs, 3, 3)

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

    ##### Initialize parameters for optimization #####
    params = EpipolarAdjustmentParameters(
        R_w2c=R_w2c, t_w2c=t_w2c, focal_scale=focal_scale, precision=precision
    )
    del R_w2c, t_w2c

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

    ##### Optimization loop #####
    with torch.enable_grad():
        for iter_idx in range(1000000000):
            (
                R_w2c,
                t_w2c,
                focal_scale,
            ) = params()  # (num_images, 3, 3), (num_images, 3), (num_cameras,)

            # compute the fundamental matrix
            fundamental = _compute_fundamental_matrix(
                image_idx1=image_idx1,
                image_idx2=image_idx2,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                focal_scale=focal_scale,
                camera_idx=camera_idx,
            )  # (num_image_pairs, 3, 3)

            # flatten the fundamental matrix
            fundamental = fundamental.reshape(
                num_image_pairs, 9
            )  # (num_image_pairs, 9)

            # compute the loss
            loss = 0.5 * torch.einsum("bi,bij,bj->b", fundamental, W, fundamental).sum()

            # backprop
            optimizer.zero_grad()
            loss.backward()
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
                    f"[Iter {iter_idx} ({precision})] loss={loss.item():.8f}, moving_loss={moving_loss:.8f}"
                )

    ##### Get the results and convert to the original dtype #####
    (
        R_w2c,
        t_w2c,
        focal_scale,
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
