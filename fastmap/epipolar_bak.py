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
        self.t_w2c_fixed = t_w2c.clone().to(precision)[:2].detach()  # (num_images, 3)
        self.t_w2c = nn.Parameter(
            t_w2c[2:].clone().to(precision), requires_grad=True
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
        t_w2c = torch.cat([self.t_w2c_fixed, t_w2c], dim=0)
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
        weights: torch.Tensor float (num_point_pairs,), the weights for each point pair
    """
    ##### Get some information #####
    device = point_pairs.device
    num_point_pairs = point_pairs.num_point_pairs
    num_valid_point_pairs = point_pair_mask.long().sum().item()
    assert image_pair_idx.shape == (num_point_pairs,)
    assert precision in [torch.float32, torch.float64]

    ##### Estimate the standard deviation of the residuals #####
    # initialize the residuals for each point pair
    residuals = torch.zeros(
        num_point_pairs, device=device, dtype=precision
    )  # (num_point_pairs,)

    # populate the residuals
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
        batch_fundamental = prev_fundamental[batch_image_pair_idx].to(
            precision
        )  # (num_valid_point_pairs_in_batch, 3, 3)
        batch_xy_homo1 = F.normalize(
            to_homogeneous(batch_data.xy1[batch_point_pair_mask].to(precision)),
            p=2,
            dim=-1,
        )  # (num_valid_point_pairs_in_batch, 3)
        batch_xy_homo2 = F.normalize(
            to_homogeneous(batch_data.xy2[batch_point_pair_mask].to(precision)),
            p=2,
            dim=-1,
        )  # (num_valid_point_pairs_in_batch, 3)
        assert batch_image_pair_idx.shape == (num_valid_point_pairs_in_batch,)
        assert batch_fundamental.shape == (num_valid_point_pairs_in_batch, 3, 3)
        assert batch_xy_homo1.shape == (num_valid_point_pairs_in_batch, 3)
        assert batch_xy_homo2.shape == (num_valid_point_pairs_in_batch, 3)
        del num_valid_point_pairs_in_batch

        # compute the linear coefficients
        batch_w = torch.einsum(
            "bi,bj->bij", batch_xy_homo2, batch_xy_homo1
        )  # (num_valid_point_pairs_in_batch, 3, 3)
        batch_w = batch_w.reshape(-1, 9)  # (num_valid_point_pairs_in_batch, 9)

        # compute residuals
        batch_residual = (batch_w * batch_fundamental.view(-1, 9)).sum(
            dim=-1
        )  # (num_valid_point_pairs_in_batch,)

        # write the residuals
        residuals[batch_data.point_pair_idx[batch_point_pair_mask]] = batch_residual
        del batch_data, batch_point_pair_mask

    # compute the standard deviation of the residuals
    std = 1.4826 * torch.median(residuals[point_pair_mask].abs())
    std = std.item()  # scalar
    # std = 0.00035  # debug: use a fixed value for testing
    print("std", std)
    del residuals

    ##### Accumulate the quadratic form #####
    # initialize the quadratic form (always use float64 for accumulation)
    W = torch.zeros(
        num_image_pairs, 9, 9, device=device, dtype=torch.float64
    )  # (num_image_pairs, 9, 9)

    # initialize the weights for each point pair
    weights = torch.zeros(
        num_point_pairs, device=device, dtype=precision
    )  # (num_point_pairs,)

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
        batch_fundamental = prev_fundamental[batch_image_pair_idx].to(
            precision
        )  # (num_valid_point_pairs_in_batch, 3, 3)
        batch_xy_homo1 = F.normalize(
            to_homogeneous(batch_data.xy1[batch_point_pair_mask].to(precision)),
            p=2,
            dim=-1,
        )  # (num_valid_point_pairs_in_batch, 3)
        batch_xy_homo2 = F.normalize(
            to_homogeneous(batch_data.xy2[batch_point_pair_mask].to(precision)),
            p=2,
            dim=-1,
        )  # (num_valid_point_pairs_in_batch, 3)
        assert batch_image_pair_idx.shape == (num_valid_point_pairs_in_batch,)
        assert batch_fundamental.shape == (num_valid_point_pairs_in_batch, 3, 3)
        assert batch_xy_homo1.shape == (num_valid_point_pairs_in_batch, 3)
        assert batch_xy_homo2.shape == (num_valid_point_pairs_in_batch, 3)
        del num_valid_point_pairs_in_batch

        # compute the linear coefficients
        batch_w = torch.einsum(
            "bi,bj->bij", batch_xy_homo2, batch_xy_homo1
        )  # (num_valid_point_pairs_in_batch, 3, 3)
        batch_w = batch_w.reshape(-1, 9)  # (num_valid_point_pairs_in_batch, 9)
        del batch_xy_homo1, batch_xy_homo2

        # compute residuals
        batch_residual = (batch_w * batch_fundamental.view(-1, 9)).sum(
            dim=-1
        )  # (num_valid_point_pairs_in_batch,)

        # scale the residuals
        batch_residual /= std

        # compute weights for huber loss
        batch_weights = 1.0 / torch.clamp(
            batch_residual.abs(), min=1.0
        )  # (num_valid_point_pairs_in_batch,)
        weights[batch_data.point_pair_idx[batch_point_pair_mask]] = batch_weights.to(
            precision
        )
        del batch_data, batch_point_pair_mask

        # compute w w^T
        batch_W = torch.einsum(
            "bi,bj->bij", batch_w, batch_w
        )  # (num_valid_point_pairs_in_batch, 9, 9)

        # scale by the standard deviation
        # batch_W /= (std**2.0) * float(
        #     num_valid_point_pairs
        # )  # (num_valid_point_pairs_in_batch, 9, 9)

        batch_W /= std**2.0
        # batch_W /= std

        # scale by the number of point pairs
        batch_W /= float(
            num_valid_point_pairs
        )  # (num_valid_point_pairs_in_batch, 9, 9)

        # get weighted quadratic form
        batch_W = (
            batch_W * batch_weights[..., None, None]
        )  # (num_valid_point_pairs_in_batch, 9, 9)
        del batch_fundamental, batch_weights

        # convert to float64
        batch_W = batch_W.to(torch.float64)  # (num_valid_point_pairs_in_batch, 9)

        # accumulate
        W.scatter_reduce_(
            dim=0,
            index=batch_image_pair_idx[:, None, None].expand(-1, 9, 9),
            src=batch_W,
            reduce="sum",
            include_self=True,
        )  # (num_image_pairs, 9, 9)

    # scale by the number of point pairs
    # W /= point_pair_mask.to(W).sum()  # (num_image_pairs, 9, 9)
    # weights /= point_pair_mask.to(weights).sum()  # (num_point_pairs,)

    # convert to the desired precision
    W = W.to(precision)  # (num_image_pairs, 9, 9)
    assert weights.dtype == precision
    # print(W)
    # print(weights)
    # quit()

    ##### Return #####
    return W, weights


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
    num_irls_steps: int = 3,
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
        num_irls_steps: int, the number of IRLS steps for each round
        precision: torch.dtype, the precision for the optimization.
        log_interval: int, the log interval in number of iterations.

    Returns:
        R_w2c: torch.Tensor float (num_images, 3, 3), w2c global rotation
        t_w2c: torch.Tensor float (num_images, 3), w2c global translation
        focal_scale: torch.Tensor float (num_cameras,), the optimized scale factor on focal lengths.
    """
    precision = torch.float64
    ##### Get some information #####
    device = R_w2c.device
    num_point_pairs = point_pair_mask.shape[0]

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

    ##### Initialize the weights to be all zero #####
    weights = torch.zeros(
        num_point_pairs, device=device, dtype=precision
    )  # (num_point_pairs,)

    ##### Iterative Reweighted Least Squares (IRLS) loop #####
    for irls_step in range(10):

        ##### Compute the quadratic form #####
        # compute the current fundamental matrix
        current_fundamental = _compute_fundamental_matrix(
            image_idx1=image_idx1,
            image_idx2=image_idx2,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            focal_scale=focal_scale,
            camera_idx=camera_idx,
        )  # (num_image_pairs, 3, 3)
        debug_old_fundamental = (
            current_fundamental.clone()
        )  # debug: keep the old fundamental for debugging

        # compute the weighted quadratic form
        old_weights = weights.clone()  # (num_point_pairs,)
        W, weights = quadratic_form(
            num_image_pairs=num_image_pairs,
            point_pairs=point_pairs,
            image_pair_idx=image_pair_idx,
            prev_fundamental=current_fundamental,
            point_pair_mask=point_pair_mask,
            precision=precision,
        )  # (num_image_pairs, 9, 9)

        # prevent misuse
        del current_fundamental

        ##### Optimization loop #####
        with torch.enable_grad():
            ##### Initialize parameters for optimization #####
            params = EpipolarAdjustmentParameters(
                R_w2c=R_w2c.contiguous(),
                t_w2c=t_w2c.contiguous(),
                focal_scale=focal_scale.contiguous(),
                precision=precision,
            )

            ############## jiahao debug begins ##############
            (
                R_w2c,
                t_w2c,
                focal_scale,
            ) = params()  # (num_images, 3, 3), (num_images, 3), (num_cameras,)
            # print(
            #     f"t max {t_w2c.abs().max().item():.8f}, t min {t_w2c.abs().min().item():.8f}, t median {t_w2c.abs().median().item():.8f}"
            # )

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

            # t_c2w = torch.einsum(
            #     "bij,bj->bi", R_w2c.transpose(-1, -2), -t_w2c
            # )  # (num_images, 3)
            # loss = (
            #     loss
            #     + 0.1 * (t_c2w[0] ** 2.0).sum()
            #     + 0.1 * (((t_c2w[0] - t_c2w[50]).norm() - 1.0) ** 2.0)
            #     # debug: add a regularization term to the loss
            # )

            # def signed_sqrt(x):
            #     """Compute the signed square root of a tensor."""
            #     return torch.sign(x) * torch.sqrt(torch.abs(x))

            def compute_hessian(X):
                grad = torch.autograd.grad(loss, X, create_graph=True)[0]
                grad_flat = grad.flatten()
                # print("grad_flat", grad_flat.shape)
                # print("grad flat magnitude", grad_flat.abs().sort()[0][:100])
                # print("grad flat magnitude", grad_flat.abs().sort()[0][-100:])

                hessian = []
                for idx in range(grad_flat.shape[0]):
                    # print(idx, grad_flat.shape[0])
                    grad2 = torch.autograd.grad(grad_flat[idx], X, retain_graph=True)[0]
                    hessian.append(grad2.flatten())

                hessian_matrix = torch.stack(hessian, dim=0)  # shape: (10N, 10N)
                return hessian_matrix.detach()

            hessian = compute_hessian(params.t_w2c)
            print("hessian condition nubmer", torch.linalg.cond(hessian).item())
            print("hessian dtype", hessian.dtype)
            # hessian_reg = hessian + 1e-12 * torch.eye(
            #     hessian.shape[0], device=hessian.device, dtype=hessian.dtype
            # )
            # eig_vals, eig_vecs = torch.linalg.eigh(hessian_reg)
            eig_vals, eig_vecs = torch.linalg.eigh(
                hessian
                + 1e-8
                * torch.eye(
                    hessian.shape[0], device=hessian.device, dtype=hessian.dtype
                )
            )
            eig_vals = eig_vals.real
            # eig_vals = eig_vals.clamp(
            #     min=1e-8
            # )  # debug: set negative eigenvalues to a small value
            # eig_vals[eig_vals < 1e-8] = 0.0  # debug: set negative eigenvalues to zero
            # print(hessian.shape)
            # print(torch.diag(hessian).sort()[0][:100])
            # print(torch.diag(hessian).sort()[0][-100:])
            # print(
            #     "hessian eig", torch.linalg.eigvals(hessian).real.abs().sort()[0][:100]
            # )
            # print(
            #     "hessian eig", torch.linalg.eigvals(hessian).real.abs().sort()[0][-100:]
            # )
            # print(eig_vals[:100])
            # print(eig_vals[-100:])
            # quit()
            eig_vals_sqrt = eig_vals.sqrt()
            eig_vals_inv_sqrt = 1.0 / eig_vals_sqrt
            # eig_vals_inv_sqrt[eig_vals_sqrt == 0.0] = (
            #     0.0  # debug: set zero eigenvalues to zero
            # )
            # print(eig_vals_inv_sqrt[:100])
            # print(eig_vals_inv_sqrt[-100:])
            # print(eig_vals_inv_sqrt * eig_vals_sqrt)
            # print(eig_vecs.T @ eig_vecs)
            # print(eig_vecs @ eig_vecs.T)
            # quit()

            # transform the values of the parameters
            if True:
                params.t_w2c.data.copy_(
                    torch.einsum("ij,j->i", eig_vecs.T, params.t_w2c.data.view(-1)).view(
                        -1, 3
                    )
                )
                params.t_w2c.data.copy_(
                    (eig_vals_sqrt * params.t_w2c.data.view(-1)).view(-1, 3)
                )

            from copy import deepcopy

            fresh_params = deepcopy(
                params
            )  # debug: keep a copy of the fresh parameters
            # del R_w2c, t_w2c, focal_scale # debug: resume
            ############## jiahao debug ends ##############

            ##### Create the L-BFGS optimizer #####
            optimizer = torch.optim.LBFGS(
                # [params.rot6d_w2c, params.t_w2c, params.focal_scale],
                # [params.t_w2c, params.focal_scale],
                [params.t_w2c],
                lr=1.0,
                max_iter=100,  # debug: use argument
                history_size=20,  # debug: use argument
                tolerance_grad=1e-9,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )

            # optimizer = torch.optim.Adam(
            #     params.parameters(), lr=1e-3
            # )  # debug: use Adam optimizer

            ##### Define the L-BFGS closure #####
            # use out-of-scope variables to keep track of L-BFGS info
            final_loss = 0.0
            num_lbfgs_evals = 0

            def lbfgs_closure():
                # clear previous gradients
                nonlocal optimizer
                optimizer.zero_grad()

                # forward the parameters
                nonlocal params
                (
                    R_w2c,
                    t_w2c,
                    focal_scale,
                ) = params()  # (num_images, 3, 3), (num_images, 3), (num_cameras,)
                # print(
                #     f"t max {t_w2c.abs().max().item():.8f}, t min {t_w2c.abs().min().item():.8f}, t median {t_w2c.abs().median().item():.8f}"
                # )

                # transform the t_w2c
                if True:
                    t_w2c_fixed = t_w2c[:2].detach()  # (2, 3)
                    t_w2c = t_w2c[2:].view(-1)
                    t_w2c = eig_vals_inv_sqrt * t_w2c
                    t_w2c = torch.einsum("ij,j->i", eig_vecs, t_w2c)
                    t_w2c = t_w2c.view(-1, 3)
                    t_w2c = torch.cat([t_w2c_fixed, t_w2c], dim=0)  # (num_images, 3)

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
                loss = (
                    0.5
                    * torch.einsum("bi,bij,bj->b", fundamental, W, fundamental).sum()
                )

                # t_c2w = torch.einsum(
                #     "bij,bj->bi", R_w2c.transpose(-1, -2), -t_w2c
                # )  # (num_images, 3)
                # loss = (
                #     loss
                #     + 0.05 * (t_c2w[0] ** 2.0).sum()
                #     + 0.05 * (((t_c2w[0] - t_c2w[50]).norm() - 1.0) ** 2.0)
                #     # debug: add a regularization term to the loss
                # )
                # print(
                #     f"t_w2c: min {t_w2c.abs().min().item():.8f}, max {t_w2c.abs().max().item():.8f}, median {t_w2c.abs().median().item():.8f}"
                # )
                # print(
                #     f"t_c2w: min {t_c2w.abs().min().item():.8f}, max {t_c2w.abs().max().item():.8f}, median {t_c2w.abs().median().item():.8f}"
                # )

                def compute_hessian(X):
                    grad = torch.autograd.grad(loss, X, create_graph=True)[0]
                    grad_flat = grad.flatten()
                    print("grad_flat", grad_flat.shape)
                    print("grad flat magnitude", grad_flat.abs().sort()[0][:100])
                    print("grad flat magnitude", grad_flat.abs().sort()[0][-100:])

                    hessian = []
                    for idx in range(grad_flat.shape[0]):
                        # print(idx, grad_flat.shape[0])
                        grad2 = torch.autograd.grad(
                            grad_flat[idx], X, retain_graph=True
                        )[0]
                        hessian.append(grad2.flatten())

                    hessian_matrix = torch.stack(hessian, dim=0)  # shape: (10N, 10N)
                    return hessian_matrix.detach()

                # hessian = compute_hessian(params.t_w2c)
                # print(hessian.shape)
                # print(hessian)
                # print(torch.diag(hessian).sort()[0][:100])
                # print(torch.diag(hessian).sort()[0][-100:])
                # print(torch.linalg.eigvals(hessian).real.abs().sort()[0][:100])
                # print(torch.linalg.eigvals(hessian).real.abs().sort()[0][-100:])
                # quit()

                # backprop
                loss.backward()
                params.rot6d_w2c.grad[params.rot6d_w2c.grad.isnan()] = 0.0
                params.t_w2c.grad[params.t_w2c.grad.isnan()] = 0.0
                # params.rot6d_w2c.grad *= 0.0
                # params.t_w2c.grad *= 0.0
                # params.rot6d_w2c.grad[0] *= 0.0
                # params.t_w2c.grad[0] *= 0.0
                # params.t_w2c.grad[50] *= 0.0
                # params.focal_scale.grad *= (
                #     0.0  # debug: zero out the focal scale gradient
                # )

                # print(
                #     "R grad",
                #     params.rot6d_w2c.grad.abs().min().item(),
                #     params.rot6d_w2c.grad.abs().max().item(),
                #     params.rot6d_w2c.grad.abs().median().item(),
                # )
                # print(
                #     "t grad",
                #     params.t_w2c.grad.abs().min().item(),
                #     params.t_w2c.grad.abs().max().item(),
                #     params.t_w2c.grad.abs().median().item(),
                # )
                # print(
                #     "focal grad",
                #     params.focal_scale.grad.abs().min().item(),
                #     params.focal_scale.grad.abs().max().item(),
                #     params.focal_scale.grad.abs().median().item(),
                # )
                # quit()

                # update the final loss
                nonlocal final_loss
                final_loss = loss.item()

                # update the number of evaluations
                nonlocal num_lbfgs_evals
                num_lbfgs_evals += 1
                print(f"BFGS step {num_lbfgs_evals}, loss: {final_loss:.8f}")

                # return the loss
                return loss

            ##### Run the L-BFGS loop #####
            optimizer.step(lbfgs_closure)
            import time

            time.sleep(5)

            num_lbfgs_evals = 0  # debug: reset the number of evaluations
            params = fresh_params  # debug: reset the parameters to the fresh ones
            optimizer = torch.optim.Adam(
                [params.t_w2c], lr=1e-3, betas=(0.9, 0.95)
            )  # debug: use Adam optimizer
            for i in range(100):
                optimizer.step(lbfgs_closure)
            quit()
            time.sleep(5)  # debug: wait for a while to see the results

            num_lbfgs_evals = 0  # debug: reset the number of evaluations
            optimizer = torch.optim.LBFGS(
                params.parameters(),
                lr=1.0,
                max_iter=10000,  # debug: use argument
                history_size=20,  # debug: use argument
                tolerance_grad=1e-9,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )
            optimizer.step(lbfgs_closure)
            quit()
            print(f"Final loss: {final_loss:.8f}")
            print(f"Number of L-BFGS evaluations: {num_lbfgs_evals}")

        ##### Update the results #####
        ########
        old_R_w2c = R_w2c  # debug
        old_t_w2c = t_w2c  # debug
        old_focal_scale = focal_scale  # debug
        ########
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
        print("#################")
        fundamental = _compute_fundamental_matrix(  # debug: recompute fundamental matrix to check if it has changed
            image_idx1=image_idx1,
            image_idx2=image_idx2,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            focal_scale=focal_scale,
            camera_idx=camera_idx,
        )  # (num_image_pairs, 3, 3)
        print(f"R_w2c max change: {torch.max(torch.abs(R_w2c - old_R_w2c)):.8f}")
        print(f"t_w2c max change: {torch.max(torch.abs(t_w2c - old_t_w2c)):.8f}")
        print(
            f"focal_scale max change: {torch.max(torch.abs(focal_scale - old_focal_scale)):.8f}"
        )
        print(
            f"fundamental max change: {torch.max(torch.abs(fundamental - debug_old_fundamental)):.8f}"
        )
        print("#################")

    ##### Convert to the original dtype #####
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
    num_prune_steps = 0
    for i in range(num_prune_steps + 1):
        with timer(f"Round {i}"):
            logger.info(f"[Round {i+1} / {num_prune_steps+1}]")
            R_w2c, t_w2c, focal_scale = loop(
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                focal_scale=focal_scale,
                point_pairs=point_pairs,
                point_pair_mask=point_pair_mask,
                camera_idx=cameras.camera_idx,
                num_irls_steps=num_irls_steps,
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

    ##### Return #####
    return R_w2c, t_w2c, focal_scale, point_pair_mask
