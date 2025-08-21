from loguru import logger
import torch
import torch.nn.functional as F
import math
import typing

from fastmap.container import Matches, MatchesData
from fastmap.utils import (
    to_homogeneous,
    packed_quantile,
    normalize_matrix,
)


@torch.no_grad()
def decompose_homography(
    H: torch.Tensor, K1: torch.Tensor | None = None, K2: torch.Tensor | None = None
):
    """Decompose homography matrix H into R, t, and n
    It is based on the first method in the paper "Deeper understanding of the homography decomposition for vision-based control": https://inria.hal.science/inria-00174036v2/document
    and the implementation in COLMAP: https://github.com/colmap/colmap/blob/c5c7b9c7cf2f34a7b102de29e356c103db8263be/src/colmap/geometry/homography_matrix.cc#L65
    It will only return the 4 solutions that satisfy the constraint that both cameras are at the same side of the plane. If the solutions do not contain a correct one, it usually means that the homography is generated from an unrealistic configuration where the cameras are at different sides of the plane.
    Args:
        H: torch.Tensor (..., 3, 3), homography matrix
        K1: torch.Tensor (..., 3, 3), intrinsic matrix of camera 1 (H is such that x2 = H @ x1)
        K2: torch.Tensor (..., 3, 3), intrinsic matrix of camera 2 (H is such that x2 = H @ x1)
    Returns:
        R: torch.Tensor (..., 4, 3, 3), rotation matrix
        t: torch.Tensor (..., 4, 3), translation vector
        n: torch.Tensor (..., 4, 3), normal vector
        valid_mask: torch.Tensor (...,), mask of the valid solutions
    """
    # get device
    device = H.device
    input_dtype = H.dtype

    # build K1 and K2 if not provided
    if K1 is None:
        K1 = torch.eye(3, dtype=input_dtype, device=device).expand_as(H)
    if K2 is None:
        K2 = torch.eye(3, dtype=input_dtype, device=device).expand_as(H)

    # switch to float64 for computation to avoid numerical instability
    assert input_dtype == H.dtype == K1.dtype == K2.dtype
    dtype = torch.float64
    H = H.to(dtype)
    K1 = K1.to(dtype)
    K2 = K2.to(dtype)

    # calibrate H
    H_normalized = torch.inverse(K2) @ H @ K1  # (..., 3, 3)
    del H, K1, K2

    # detect homography matrix with low rank and large condition number
    valid_mask = torch.linalg.matrix_rank(H_normalized) == 3  # (...,)
    _sinvals = torch.linalg.svdvals(H_normalized)  # (..., 3)
    valid_mask &= _sinvals[..., 0] / _sinvals[..., 2] < 1e3  # (...,)
    del _sinvals

    # normalize H
    H_normalized /= torch.linalg.svdvals(H_normalized)[
        ..., 1, None, None
    ]  # (..., 3, 3)

    # make sure det(H_normalized) > 0. According to the comment in COLMAP, this also ensures the resulting R is indeed a rotation matrix
    H_normalized *= torch.sign(torch.det(H_normalized))[..., None, None]  # (..., 3, 3)

    # compute S
    S = H_normalized.transpose(-1, -2) @ H_normalized - torch.eye(
        3, dtype=dtype, device=device
    ).expand_as(
        H_normalized
    )  # (..., 3, 3)

    # get the mask of pure rotation
    pure_rotation_mask = (
        torch.abs(S).flatten(start_dim=-2).max(dim=-1).values < 0.005
    )  # (...) note that this is slightly different from the COLMAP implementation

    # get a bunch of values (using the same notations as COLMAP)
    M00 = _opposite_minor(S, 0, 0)  # (...)
    M11 = _opposite_minor(S, 1, 1)  # (...)
    M22 = _opposite_minor(S, 2, 2)  # (...)
    # IMPORTANT NOTE: according to some mysterious math (See Appendix C.1 in the paper), M00, M11, M22 should always be non-negative. If they are not, it is usually due to numerical errors, and they will be very close to zero. So here we clamp them to 0 avoid nan in sqrt. But if something goes wrong, it is better to remove the clamp and expose the problem.
    M00 = torch.clamp(M00, min=0.0)  # (...)
    M11 = torch.clamp(M11, min=0.0)  # (...)
    M22 = torch.clamp(M22, min=0.0)  # (...)

    rtM00 = torch.sqrt(M00)  # (...)
    rtM11 = torch.sqrt(M11)  # (...)
    rtM22 = torch.sqrt(M22)  # (...)

    M01 = _opposite_minor(S, 0, 1)  # (...)
    M12 = _opposite_minor(S, 1, 2)  # (...)
    M02 = _opposite_minor(S, 0, 2)  # (...)

    e12 = torch.where(M12 >= 0.0, 1.0, -1.0)  # (...)
    e02 = torch.where(M02 >= 0.0, 1.0, -1.0)  # (...)
    e01 = torch.where(M01 >= 0.0, 1.0, -1.0)  # (...)

    nS00 = S[..., 0, 0].abs()  # (...)
    nS11 = S[..., 1, 1].abs()  # (...)
    nS22 = S[..., 2, 2].abs()  # (...)

    # compute the normal vector in three cases for convenience
    # case 1: S00 != 0
    np1_S00 = torch.zeros_like(S[..., 0])  # (..., 3)
    np2_S00 = torch.zeros_like(S[..., 0])  # (..., 3)
    np1_S00[..., 0] = S[..., 0, 0]  # (...)
    np2_S00[..., 0] = S[..., 0, 0]  # (...)
    np1_S00[..., 1] = S[..., 0, 1] + rtM22  # (...)
    np2_S00[..., 1] = S[..., 0, 1] - rtM22  # (...)
    np1_S00[..., 2] = S[..., 0, 2] + e12 * rtM11  # (...)
    np2_S00[..., 2] = S[..., 0, 2] - e12 * rtM11  # (...)
    # case 2: S11 != 0
    np1_S11 = torch.zeros_like(S[..., 0])  # (..., 3)
    np2_S11 = torch.zeros_like(S[..., 0])  # (..., 3)
    np1_S11[..., 0] = S[..., 0, 1] + rtM22  # (...)
    np2_S11[..., 0] = S[..., 0, 1] - rtM22  # (...)
    np1_S11[..., 1] = S[..., 1, 1]  # (...)
    np2_S11[..., 1] = S[..., 1, 1]  # (...)
    np1_S11[..., 2] = S[..., 1, 2] - e02 * rtM00  # (...)
    np2_S11[..., 2] = S[..., 1, 2] + e02 * rtM00  # (...)
    # case 3: S22 != 0
    np1_S22 = torch.zeros_like(S[..., 0])  # (..., 3)
    np2_S22 = torch.zeros_like(S[..., 0])  # (..., 3)
    np1_S22[..., 0] = S[..., 0, 2] + e01 * rtM11  # (...)
    np2_S22[..., 0] = S[..., 0, 2] - e01 * rtM11  # (...)
    np1_S22[..., 1] = S[..., 1, 2] + rtM00  # (...)
    np2_S22[..., 1] = S[..., 1, 2] - rtM00  # (...)
    np1_S22[..., 2] = S[..., 2, 2]  # (...)
    np2_S22[..., 2] = S[..., 2, 2]  # (...)

    # pick the correct np1 and np2 based on absoulte values of Sii
    max_ii_idx = torch.stack([nS00, nS11, nS22], dim=-1).argmax(dim=-1)  # (...)
    expanded_max_ii_idx = max_ii_idx[..., None].expand_as(np1_S00)  # (..., 3)
    np1 = torch.where(expanded_max_ii_idx == 0, np1_S00, 0.0)  # (..., 3)
    np1 = torch.where(expanded_max_ii_idx == 1, np1_S11, np1)  # (..., 3)
    np1 = torch.where(expanded_max_ii_idx == 2, np1_S22, np1)  # (..., 3)
    np2 = torch.where(expanded_max_ii_idx == 0, np2_S00, 0.0)  # (..., 3)
    np2 = torch.where(expanded_max_ii_idx == 1, np2_S11, np2)  # (..., 3)
    np2 = torch.where(expanded_max_ii_idx == 2, np2_S22, np2)  # (..., 3)
    del np1_S00, np2_S00, np1_S11, np2_S11, np1_S22, np2_S22

    # compute t (using the same notations as COLMAP)
    traceS = S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]  # (...)
    v = 2.0 * torch.sqrt(1.0 + traceS - M00 - M11 - M22)  # (...)

    Sii = torch.where(max_ii_idx == 0, S[..., 0, 0], 0.0)  # (...)
    Sii = torch.where(max_ii_idx == 1, S[..., 1, 1], Sii)  # (...)
    Sii = torch.where(max_ii_idx == 2, S[..., 2, 2], Sii)  # (...)
    ESii = torch.where(Sii >= 0.0, 1.0, -1.0)  # (...)
    r_2 = 2 + traceS + v  # (...)
    nt_2 = 2 + traceS - v  # (...)

    r = torch.sqrt(r_2)  # (...)
    n_t = torch.sqrt(nt_2)  # (...)

    n1 = F.normalize(np1, p=2, dim=-1)  # (..., 3)
    n2 = F.normalize(np2, p=2, dim=-1)  # (..., 3)

    half_nt = 0.5 * n_t  # (...)
    esii_t_r = ESii * r  # (...)

    t1_star = half_nt[..., None] * (
        esii_t_r[..., None] * n2 - n_t[..., None] * n1
    )  # (..., 3)
    t2_star = half_nt[..., None] * (
        esii_t_r[..., None] * n1 - n_t[..., None] * n2
    )  # (..., 3)

    # add last dimension to make vectors 3x1
    n1 = n1.unsqueeze(-1)  # (..., 3, 1)
    n2 = n2.unsqueeze(-1)  # (..., 3, 1)
    t1_star = t1_star.unsqueeze(-1)  # (..., 3, 1)
    t2_star = t2_star.unsqueeze(-1)  # (..., 3, 1)

    # compute R1, R2, t1, t2
    R1 = H_normalized @ (
        torch.eye(3, dtype=dtype, device=device).expand_as(H_normalized)
        - (2.0 / v[..., None, None]) * t1_star * n1.transpose(-1, -2)
    )  # (..., 3, 3)
    R2 = H_normalized @ (
        torch.eye(3, dtype=dtype, device=device).expand_as(H_normalized)
        - (2.0 / v[..., None, None]) * t2_star * n2.transpose(-1, -2)
    )
    t1 = R1 @ t1_star  # (..., 3, 1)
    t2 = R2 @ t2_star  # (..., 3, 1)

    # deal with pure rotation case
    if torch.any(pure_rotation_mask):
        R1[pure_rotation_mask] = H_normalized[pure_rotation_mask]
        R2[pure_rotation_mask] = H_normalized[pure_rotation_mask]
        t1[pure_rotation_mask] = torch.zeros_like(t1[pure_rotation_mask])
        t2[pure_rotation_mask] = torch.zeros_like(t2[pure_rotation_mask])
        n1[pure_rotation_mask] = torch.zeros_like(n1[pure_rotation_mask])
        n2[pure_rotation_mask] = torch.zeros_like(n2[pure_rotation_mask])

    # compute final results
    R = torch.stack([R1, R1, R2, R2], dim=-3)  # (..., 4, 3, 3)
    t = torch.stack([t1, -t1, t2, -t2], dim=-3)  # (..., 4, 3, 1)
    n = torch.stack([-n1, n1, -n2, n2], dim=-3)  # (..., 4, 3, 1)

    # squeeze the last dimension
    t = t.squeeze(-1)  # (..., 4, 3)
    n = n.squeeze(-1)  # (..., 4, 3)

    # normalize t
    t = F.normalize(t, p=2, dim=-1)  # (..., 4, 3)

    # convert back to the original dtype
    R = R.to(input_dtype)
    t = t.to(input_dtype)
    n = n.to(input_dtype)

    return R, t, n, valid_mask


@torch.no_grad()
def _opposite_minor(A, i, j):
    """Compute the opposite minor of a 3x3 matrix A. Will be used in the homography decomposition (See page 14 of "Deeper understanding of the homography decomposition for vision-based control"
    Args:
        A: torch.Tensor (..., 3, 3), matrix
        i: int, row index starting from 0
        j: int, column index starting from 0
    Returns:
        M: torch.Tensor (...,), opposite minor
    """
    # eliminate the ith row
    A = torch.cat([A[..., :i, :], A[..., i + 1 :, :]], dim=-2)  # (..., 2, 3)

    # eliminate the jth column
    A = torch.cat([A[..., :, :j], A[..., :, j + 1 :]], dim=-1)  # (..., 2, 2)

    # compute the opposite minor
    M = -torch.linalg.det(A)  # (...)

    return M


@torch.no_grad()
def _find_homography_inliers(matches_data: MatchesData, homography: torch.Tensor):
    """Weirdly there are some outliers in the homography image pairs from the COLMAP database. This function finds the inliers with some heuristics.

    Args:
        matches_data: MatchesData. Assume all image pairs are homography pairs
        homography: torch.Tensor float (num_image_pairs, 3, 3), the previously estimated homography matrix. Note that some slots correspond to image pairs not in this matches_data container and will not be used.
    Returns:
        mask: torch.Tensor bool (num_query_point_pairs,), the mask of the inliers for point pairs in matches_data
    """
    # get the total number of image pairs
    num_image_pairs = homography.shape[0]

    # get the homography matrix for each point pair
    homography = normalize_matrix(homography)  # (num_image_pairs, 3, 3)
    homography = homography[
        matches_data.image_pair_idx
    ]  # (num_query_point_pairs, 3, 3)

    # compute residual
    xy_homo1 = to_homogeneous(matches_data.xy1)  # (num_query_point_pairs, 3)
    xy_homo2_pred = torch.einsum(
        "bij,bj->bi", homography, xy_homo1
    )  # (num_query_point_pairs, 3)
    xy2_pred = (
        xy_homo2_pred[:, :2] / xy_homo2_pred[:, 2:3]
    )  # (num_query_point_pairs, 2)
    residual = (xy2_pred - matches_data.xy2).norm(dim=-1)  # (num_query_point_pairs,)
    del xy_homo1, xy_homo2_pred, xy2_pred

    # determine threshold with heuristics (there are slots for image pairs not in this matches_data container)
    thr = 3.0 * packed_quantile(
        values=residual,
        group_idx=matches_data.image_pair_idx,
        q=0.5,
        num_groups=num_image_pairs,
    )  # (num_image_pairs,)
    thr[~matches_data.image_pair_mask] = (
        torch.nan
    )  # (num_image_pairs,) just to make sure

    # get mask
    mask = (
        residual <= thr[matches_data.image_pair_idx] + 1e-8
    )  # (num_homography_point_pairs,)

    # return
    return mask


@torch.no_grad()
def _estimate_homography(
    matches_data: MatchesData,
    prev_homography: torch.Tensor,
):
    """Estimate the homography matrix using least squares from the xy coordinates of point pairs.

    Args:
        matches_data: MatchesData
        prev_homography: torch.Tensor float (num_image_pairs, 3, 3), the previously estimated homography matrix
        batch_size: int, the batch size for accumulation

    Returns:
        homography: torch.Tensor float (num_image_pairs, 9, 9), the homography matrix. Some slots contain invalid values.
    """
    ##### Get some information #####
    num_image_pairs = len(matches_data.image_pair_mask)
    num_point_pairs = matches_data.num_point_pairs
    device, dtype = matches_data.device, matches_data.dtype
    if prev_homography is not None:
        assert prev_homography.device == device
        assert prev_homography.dtype == dtype
        assert prev_homography.shape == (num_image_pairs, 3, 3)

    ##### Accumulate the constraints #####

    # convert to homogeneous
    xy_homo1 = to_homogeneous(matches_data.xy1)  # (num_point_pairs, 3)
    xy_homo2 = to_homogeneous(matches_data.xy2)  # (num_point_pairs, 3)

    # form the constraints
    A = torch.zeros(
        num_point_pairs, 2, 9, device=device, dtype=dtype
    )  # (num_point_pairs, 2, 9)
    A[:, 0, 3:6] = -xy_homo2[:, 2:] * xy_homo1
    A[:, 0, 6:] = xy_homo2[:, 1:2] * xy_homo1
    A[:, 1, :3] = xy_homo2[:, 2:] * xy_homo1
    A[:, 1, 6:] = -xy_homo2[:, 0:1] * xy_homo1

    ##### Weight the constraints #####
    # get the mask of inliers to deal with bug in COLMAP
    inlier_mask = _find_homography_inliers(
        matches_data=matches_data, homography=prev_homography
    )  # (num_point_pairs,)

    # compute error
    prev_homography = normalize_matrix(prev_homography)  # (num_image_pairs, 3, 3)
    prev_homography = prev_homography.reshape(-1, 1, 9)  # (num_image_pairs, 1, 9)
    error = (
        (A * prev_homography[matches_data.image_pair_idx]).sum(dim=-1).abs()
    )  # (num_point_pairs, 2)

    # weight
    epsilon = 1e-4
    weights = 1.0 / (error**0.5 + epsilon)  # (num_point_pairs, 2)
    weights[~inlier_mask] = 0.0  # (num_point_pairs, 2)
    A = A * weights[..., None]  # (num_point_pairs, 2, 9)
    del prev_homography, error, epsilon

    ##### Solve the homography matrix #####
    # get A^T A
    AT_A = torch.zeros(
        num_image_pairs, 9, 9, device=device, dtype=dtype
    )  # (num_image_pairs, 9, 9)
    # AT_A = A.transpose(-1, -2) @ A  # (num_point_pairs, 9, 9)
    AT_A.scatter_reduce_(
        dim=0,
        index=matches_data.image_pair_idx[:, None, None].expand(-1, 9, 9),
        src=A.transpose(-1, -2) @ A,
        reduce="sum",
    )

    # solve
    homography = torch.zeros(
        (num_image_pairs, 9), device=device, dtype=dtype
    )  # (num_image_pairs, 9)
    homography[matches_data.image_pair_mask] = torch.linalg.eigh(
        AT_A[matches_data.image_pair_mask]
    ).eigenvectors[..., 0]
    homography = homography.reshape(-1, 3, 3)  # (num_image_pairs, 3, 3)
    homography = normalize_matrix(homography)  # (num_image_pairs, 3, 3)
    homography[~matches_data.image_pair_mask] = torch.nan  # (num_image_pairs, 3, 3)

    ##### Return #####
    return homography


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
def re_estimate_homography(
    matches: Matches, num_iters: int = 10, precision: torch.dtype = torch.float64
):
    """Re-esimate the homography matrix in the matches container. This function is used to refine the homography matrix in the matches container after the undistortion process.

    Args:
        matches: Matches
        num_iters: int, the number of iterations to estimate the homography matrix
        precision: torch.dtype, the precision for computation
    """
    assert num_iters >= 1
    device = matches.device
    dtype = precision

    # make sure each image pair has at least 4 point pairs
    assert torch.all(matches.num_point_pairs >= 4)

    # allocate the memory for the normalization matrix (including slots for epipolar pairs that will never be used)
    T1 = torch.nan + torch.zeros(
        matches.num_image_pairs, 3, 3, device=device, dtype=dtype
    )  # (num_image_pairs, 3, 3)
    T2 = torch.nan + torch.zeros(
        matches.num_image_pairs, 3, 3, device=device, dtype=dtype
    )  # (num_image_pairs, 3, 3)

    # compute the normalization matrix
    logger.info("Computing normalization matrix...")
    for matches_data in matches.query(matches.is_homography):

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

    # allocate the memory for the homography matrix (after normalization). It includes slots for epipolar pairs that will never be used.
    homography = matches.matrix.clone().to(dtype)  # (num_image_pairs, 3, 3)
    homography = T2 @ homography @ T1.inverse()  # (num_image_pairs, 3, 3)
    homography = normalize_matrix(homography)  # (num_image_pairs, 3, 3)
    homography[~matches.is_homography] = torch.nan  # (num_image_pairs, 3, 3)

    # loop over iterations
    logger.info(f"Iterative re-weighted least squares for {num_iters} iterations...")
    for iter_idx in range(num_iters):
        # loop over batches
        for matches_data in matches.query(matches.is_homography):

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

            # estimate the homography matrix
            current_homography = _estimate_homography(
                matches_data=matches_data,
                prev_homography=homography,
            )  # (num_image_pairs, 3, 3)

            # store the homography
            homography[matches_data.image_pair_mask] = current_homography[
                matches_data.image_pair_mask
            ]

    # transform the homography matrix back
    homography = T2.inverse() @ homography @ T1  # (num_image_pairs, 3, 3)

    # normalize the homography matrix
    homography = normalize_matrix(homography)  # (num_image_pairs, 3, 3)

    # convert back to the original dtype
    homography = homography.to(matches.dtype)  # (num_image_pairs, 3, 3)

    # write to the matches container
    matches.matrix[matches.is_homography] = homography[matches.is_homography]
    assert not torch.any(torch.isnan(matches.matrix))
