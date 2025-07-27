from loguru import logger
import math
import typing
import torch

from fastmap.container import Matches, MatchesData
from fastmap.utils import to_homogeneous, normalize_matrix


@torch.no_grad()
def estimate_fundamental(
    matches_data: MatchesData,
    prev_fundamental: torch.Tensor | None = None,
):
    """Estimate the fundamental matrix using least squares from the xy coordinates of point pairs.

    Args:
        matches_data: MatchesData
        prev_fundamental: torch.Tensor float (num_image_pairs, 3, 3), the previously estimated fundamental matrix (might contain invalid values in slots not related to the image pairs in matches_data)
        batch_size: int, the batch size for accumulation

    Returns:
        fundamental: torch.Tensor float (num_image_pairs, 9, 9), the fundamental matrix. Some slots contain invalid values.
    """
    ##### Get some information #####
    num_image_pairs = len(matches_data.image_pair_mask)
    device, dtype = matches_data.device, matches_data.dtype

    ##### Accumulate the constraints #####

    # convert to homogeneous
    xy_homo1 = to_homogeneous(matches_data.xy1)  # (num_point_pairs, 3)
    xy_homo2 = to_homogeneous(matches_data.xy2)  # (num_point_pairs, 3)

    # compute A
    A = torch.einsum("bi,bj->bij", xy_homo2, xy_homo1)  # (num_point_pairs, 3, 3)
    A = A.reshape(-1, 9)  # (num_point_pairs, 9)

    # weight (use the previous fundamental matrix)
    if prev_fundamental is not None:
        prev_fundamental = normalize_matrix(prev_fundamental)  # (num_image_pairs, 3, 3)
        prev_fundamental = prev_fundamental.reshape(-1, 9)  # (num_image_pairs, 9)
        error = (
            (A * prev_fundamental[matches_data.image_pair_idx]).sum(dim=-1).abs()
        )  # (num_point_pairs,)
        epsilon = 1e-4
        A = A / (error[..., None] ** 0.5 + epsilon)  # (num_point_pairs, 9)
        del prev_fundamental, error, epsilon

    # compute A^T A
    AT_A = torch.einsum("bi,bj->bij", A, A)  # (num_point_pairs, 9, 9)

    # accumulate
    constraint = torch.zeros(
        num_image_pairs,
        9,
        9,
        device=device,
        dtype=dtype,
    )  # (num_image_pairs, 9, 9)
    constraint.scatter_reduce_(
        dim=0,
        index=matches_data.image_pair_idx[:, None, None].expand(-1, 9, 9),
        src=AT_A,
        reduce="sum",
    )  # (num_image_pairs, 9, 9)

    ##### Solve the fundamental matrix #####
    fundamental = torch.zeros(
        (num_image_pairs, 9), device=device, dtype=dtype
    )  # (num_image_pairs, 9)
    fundamental[matches_data.image_pair_mask] = torch.linalg.eigh(
        constraint[matches_data.image_pair_mask]
    ).eigenvectors[..., 0]
    fundamental = fundamental.reshape(-1, 3, 3)  # (num_image_pairs, 3, 3)
    fundamental = normalize_matrix(fundamental)  # (num_image_pairs, 3, 3)

    ##### Set invalid values to nan #####
    fundamental[~matches_data.image_pair_mask] = torch.nan  # (num_image_pairs, 3, 3)

    ##### Return #####
    return fundamental


@torch.no_grad()
def _normalize_xy(xy: torch.Tensor, image_pair_idx: torch.Tensor, T: torch.Tensor):
    """Normalize xy for estimating the fundamental. See https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm

    Args:
        xy: torch.Tensor float (num_point_pairs, 2), the xy coordinates
        image_pair_idx: torch.Tensor long (num_point_pairs,), the image pair idx for each point pair
        T: torch.Tensor float (num_image_pairs, 3, 3), the normalization matrix
    Returns:
        xy: torch.Tensor float (num_point_pairs, 2), the normalized xy coordinates
    """
    xy_homo = to_homogeneous(xy)  # (num_point_pairs, 3)
    xy_homo = torch.einsum(
        "bij,bj->bi", T[image_pair_idx], xy_homo
    )  # (num_point_pairs, 3)
    assert torch.all((xy_homo[:, -1] - 1.0).abs() < 1e-5)  # just to make sure
    xy = xy_homo[..., :-1] / xy_homo[..., -1:]  # (num_point_pairs, 2)
    return xy


@torch.no_grad()
def _compute_normalization_matrix(
    image_pair_mask: torch.Tensor,
    xy: torch.Tensor,
    image_pair_idx: torch.Tensor,
):
    """Compute the normalization matrix for some image pairs.

    Args:
        image_pair_mask: torch.Tensor bool (num_image_pairs,), the mask for the image pairs
        xy: torch.Tensor float (num_point_pairs, 2), the xy coordinates
        image_pair_idx: torch.Tensor long (num_point_pairs,), the image pair idx for each point pair
    Returns:
        T: torch.Tensor float (num_image_pairs, 3, 3), the normalization matrix. Some of the slots contain invalid values.
    """
    # get some information
    num_image_pairs = image_pair_mask.shape[0]
    num_image_pairs = typing.cast(int, num_image_pairs)
    device, dtype = xy.device, xy.dtype

    # compute mean and average norm
    mean = torch.zeros(
        num_image_pairs, 2, device=device, dtype=dtype
    )  # (num_image_pairs, 2)
    avg_norm = torch.zeros(
        num_image_pairs, device=device, dtype=dtype
    )  # (num_image_pairs,)
    mean.scatter_reduce_(
        dim=0,
        index=image_pair_idx[:, None].expand(-1, 2),
        src=xy,
        reduce="mean",
        include_self=False,
    )
    avg_norm.scatter_reduce_(
        dim=0,
        index=image_pair_idx,
        src=(xy - mean[image_pair_idx]).norm(dim=-1),
        reduce="mean",
        include_self=False,
    )

    # compute the normalization matrix
    T = (
        torch.eye(3, device=device, dtype=dtype).expand(num_image_pairs, 3, 3).clone()
    )  # (num_image_pairs, 3, 3)
    for i in range(2):
        T[:, i, i] = math.sqrt(2.0) / avg_norm
        T[:, i, -1] = -mean[:, i] * T[:, i, i]
    T[~image_pair_mask] = torch.nan  # avoid unexpected behavior

    # return
    return T


@torch.no_grad()
def re_estimate_fundamental(
    matches: Matches, num_iters: int = 10, precision: torch.dtype = torch.float64
):
    """Re-esimate the fundamental matrix in the matches container. This function is used to refine the fundamental matrix in the matches container after the undistortion process.

    Args:
        matches: Matches
        num_iters: int, the number of iterations to estimate the fundamental matrix
        precision: torch.dtype, the precision for the computation
    """
    assert num_iters >= 1
    device = matches.device
    dtype = precision

    # make sure each image pair has at least 8 point pairs
    assert torch.all(matches.num_point_pairs >= 8)

    # allocate the memory for the normalization matrix (including slots for homography pairs that will never be used)
    T1 = torch.nan + torch.zeros(
        matches.num_image_pairs, 3, 3, device=device, dtype=dtype
    )  # (num_image_pairs, 3, 3)
    T2 = torch.nan + torch.zeros(
        matches.num_image_pairs, 3, 3, device=device, dtype=dtype
    )  # (num_image_pairs, 3, 3)

    # compute the normalization matrix
    logger.info("Computing normalization matrix...")
    for matches_data in matches.query(matches.is_epipolar):

        # convert to target precision
        matches_data.to_dtype(dtype)

        # compute the normalization matrix
        current_T1 = _compute_normalization_matrix(
            image_pair_mask=matches_data.image_pair_mask,
            xy=matches_data.xy1,
            image_pair_idx=matches_data.image_pair_idx,
        )  # (num_image_pairs, 3, 3)
        current_T2 = _compute_normalization_matrix(
            image_pair_mask=matches_data.image_pair_mask,
            xy=matches_data.xy2,
            image_pair_idx=matches_data.image_pair_idx,
        )  # (num_image_pairs, 3, 3)

        # store the normalization matrix
        T1[matches_data.image_pair_mask] = current_T1[matches_data.image_pair_mask]
        T2[matches_data.image_pair_mask] = current_T2[matches_data.image_pair_mask]
        del current_T1, current_T2
    logger.info("Normalization matrix computed.")

    # allocate the memory for the fundamental matrix (after normalization). It includes slots for homography pairs that will never be used.
    fundamental = (
        T2.inverse().transpose(-1, -2) @ matches.matrix.to(dtype) @ T1.inverse()
    )  # (num_image_pairs, 3, 3)
    fundamental = normalize_matrix(fundamental)  # (num_image_pairs, 3, 3)
    fundamental[~matches.is_epipolar] = torch.nan  # avoid unexpected behavior

    # loop over iterations
    logger.info(f"Iterative re-weighted least squares for {num_iters} iterations...")
    for iter_idx in range(num_iters):
        # loop over batches
        for matches_data in matches.query(matches.is_epipolar):
            # convert to target precision
            matches_data.to_dtype(dtype)

            # normalize xy1 and xy2
            matches_data.xy1 = _normalize_xy(
                xy=matches_data.xy1,
                image_pair_idx=matches_data.image_pair_idx,
                T=T1,
            )  # (num_batch_point_pairs, 2)
            matches_data.xy2 = _normalize_xy(
                xy=matches_data.xy2,
                image_pair_idx=matches_data.image_pair_idx,
                T=T2,
            )  # (num_batch_point_pairs, 2)

            # estimate the fundamental matrix
            current_fundamental = estimate_fundamental(
                matches_data=matches_data,
                prev_fundamental=fundamental,
            )  # (num_image_pairs, 3, 3)

            # store the fundamental
            fundamental[matches_data.image_pair_mask] = current_fundamental[
                matches_data.image_pair_mask
            ]

    # transform the fundamental matrix back
    fundamental = T2.transpose(-1, -2) @ fundamental @ T1  # (num_image_pairs, 3, 3)

    # normalize the fundamental matrix
    fundamental = normalize_matrix(fundamental)  # (num_image_pairs, 3, 3)

    # convert back to the original dtype
    fundamental = fundamental.to(matches.dtype)  # (num_image_pairs, 3, 3)

    # write to the matches container
    matches.matrix[matches.is_epipolar] = fundamental[matches.is_epipolar]
    assert not torch.any(torch.isnan(matches.matrix))
