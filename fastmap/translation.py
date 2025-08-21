from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmap.container import ImagePairs
from fastmap.utils import ConvergenceManager, geometric_median


class TranslationParameters(nn.Module):
    """Module for managing translation alignment parameters for multiple initializations independently."""

    def __init__(self, t_c2w) -> None:
        """
        Args:
            t_c2w (torch.Tensor): (num_init, num_images, 3), the c2w camera translations
        """
        super().__init__()
        assert len(t_c2w.shape) == 3 and t_c2w.shape[-1] == 3

        # build parameters
        self.t_c2w = nn.Parameter(t_c2w.clone(), requires_grad=True)

    def forward(self):
        """Return the parameters after some processing"""
        # get the parameters
        t_c2w = self.t_c2w  # (num_init, num_images, 3)

        # return the results
        return t_c2w


class TorchComputeGradientModule(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        o1: torch.Tensor,  # float (B, 3)
        o2: torch.Tensor,  # float (B, 3)
        o12_gt: torch.Tensor,  # float (B, 3)
    ):
        o12 = o2 - o1  # (num_init, num_image_pairs, 3)
        o12_norm = o12.norm(dim=-1, keepdim=True)  # (num_init, num_image_pairs, 1)
        o12_normalized = o12 / (o12_norm + 1e-12)  # (num_init, num_image_pairs, 3)
        loss = F.l1_loss(o12_gt, o12_normalized, reduction="mean")
        d_o12_normalized = torch.where(o12_normalized > o12_gt, 1.0, -1.0) / len(
            o12_gt.flatten()
        )  # (num_init, num_image_pairs, 3)
        d_o12 = (
            d_o12_normalized
            - (o12_normalized * d_o12_normalized).sum(dim=-1, keepdim=True)
            * o12_normalized
        ) / o12_norm
        d_o1 = -d_o12  # (num_init, num_image_pairs, 3)
        d_o2 = d_o12  # (num_init, num_image_pairs, 3)
        return loss, d_o1, d_o2


class CUDAComputeGradientModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._initialized = False
        from fastmap.cuda import translation_gradient

        self.gradient_fn = translation_gradient

    @torch.no_grad()
    def forward(
        self,
        o1: torch.Tensor,  # float (B, 3)
        o2: torch.Tensor,  # float (B, 3)
        o12_gt: torch.Tensor,  # float (B, 3)
    ):
        if not self._initialized:
            # make sure everything is contiguous
            assert o1.is_contiguous()
            assert o2.is_contiguous()
            assert o12_gt.is_contiguous()

            # get device and dtype
            device = o1.device
            dtype = o1.dtype

            # initialize the output tensors
            self.loss = torch.zeros((1,), device=device, dtype=dtype)  # scalar
            self.d_o1 = torch.zeros_like(o1)  # (B, 3)
            self.d_o2 = torch.zeros_like(o2)  # (B, 3)

            # set flag
            self._initialized = True

        self.gradient_fn(
            o1=o1,
            o2=o2,
            o12_gt=o12_gt,
            loss=self.loss,
            d_o1=self.d_o1,
            d_o2=self.d_o2,
        )

        return (
            self.loss,
            self.d_o1,
            self.d_o2,
        )


@torch.no_grad()
def loop(
    t_w2c: torch.Tensor,
    R_w2c: torch.Tensor,
    image_pairs: ImagePairs,
    lr: float = 1e-3,
    log_interval: int = 500,
):
    """Global translation optimization loop using relative poses and global rotations. It optimizes from multiple initializations simultaneously.

    Args:
        t_w2c: torch.Tensor, float, shape=(num_init, num_images, 3), initial world to camera translation vector
        R_w2c: torch.Tensor, float, shape=(num_images, 3, 3), world to camera rotation matrix
        image_pairs: ImagePairs container
        lr: float, learning rate
        log_interval: int, log interval in iterations

    Returns:
        t_w2c: torch.Tensor, float, shape=(num_init, num_images, 3), world to camera translation vector
    """
    # get data
    translation = image_pairs.translation  # (num_image_pairs, 3, 3)
    image_idx1 = image_pairs.image_idx1  # (num_image_pairs,)
    image_idx2 = image_pairs.image_idx2  # (num_image_pairs,)

    # get information
    num_init = t_w2c.shape[0]

    # compute the gt target
    R_c2w2 = R_w2c[image_idx2].transpose(-1, -2)  # (num_image_pairs, 3, 3)
    o12_gt = torch.einsum("bij,bj->bi", R_c2w2, -translation)  # (num_image_pairs, 3)
    o12_gt = o12_gt.expand(num_init, -1, -1).clone()  # (num_init, num_image_pairs, 3)
    del R_c2w2

    try:
        compute_gradients = CUDAComputeGradientModule()
    except ImportError:
        logger.warning(
            "CUDA kernel extension for global translation alignment is not available, falling back to the slower PyTorch implementation."
        )
        compute_gradients = TorchComputeGradientModule()

    # construct the parameters
    t_c2w = torch.einsum(
        "nij,mnj->mni", R_w2c.transpose(-1, -2), -t_w2c
    )  # (num_init, num_images, 3)
    del t_w2c  # prevent misuse
    params = TranslationParameters(t_c2w=t_c2w)
    del t_c2w

    # construct the optimizer
    optimizer = torch.optim.Adam(params.parameters(), lr=lr)

    # convergence manager
    convergence_manager = ConvergenceManager(
        warmup_steps=10, decay=0.99, convergence_window=500
    )
    convergence_manager.start()

    with torch.enable_grad():
        # loop for optimization
        for iter_idx in range(1000000):
            # forward
            t_c2w = params()  # (num_init, num_images, 3)
            o1 = torch.index_select(
                input=t_c2w, dim=1, index=image_idx1
            )  # (num_init, num_image_pairs, 3)
            o2 = torch.index_select(
                input=t_c2w, dim=1, index=image_idx2
            )  # (num_init, num_image_pairs, 3)

            # compute gradients
            loss, d_o1, d_o2 = compute_gradients(o1=o1, o2=o2, o12_gt=o12_gt)

            # backward to parameters
            optimizer.zero_grad()
            torch.autograd.backward(
                tensors=[o1, o2],
                grad_tensors=[d_o1, d_o2],
                retain_graph=False,
            )

            # gradient step
            optimizer.step()

            # check convergence
            moving_loss, if_converged = convergence_manager.step(
                step=iter_idx, loss=loss
            )
            if if_converged:
                logger.info(
                    f"Converged at iter {iter_idx} with moving loss {moving_loss:.6f}"
                )
                break

            if iter_idx % log_interval == 0:
                logger.info(
                    f"[Iter {iter_idx}] loss={loss.item():.6f}, moving_loss={moving_loss:.6f}"
                )

    ##### Return the optimized parameters #####
    t_c2w = params()
    t_w2c = torch.einsum("nij,mnj->mni", R_w2c, -t_c2w)  # (num_init, num_images, 3)
    return t_w2c


@torch.no_grad()
def global_translation(
    R_w2c: torch.Tensor,
    image_pairs: ImagePairs,
    image_mask: torch.Tensor,
    num_init: int = 3,
    log_interval: int = 500,
):
    """Global alignment of translation.

    Args:
        R_w2c: torch.Tensor, float, shape=(num_images, 3, 3), world to camera rotation matrix
        image_pairs: ImagePairs container
        image_mask: torch.Tensor, bool, shape=(num_images,), True if the image is valid
        num_init: int, number of random initialization
        log_interval: int, log interval in iterations

    Returns:
        t_w2c: torch.Tensor, float, shape=(num_images, 3), world to camera translation vector
    """
    # get information
    device = R_w2c.device
    dtype = R_w2c.dtype
    num_images = R_w2c.shape[0]

    ##### Find solutions with different random initialization #####
    # randomly initialize the translation
    t_w2c_solutions = (
        torch.rand(num_init, num_images, 3, device=device, dtype=dtype) * 2.0 - 1.0
    )  # (num_init, num_images, 3)

    # loop
    logger.info(f"Optimizing translation with {num_init} random initializations...")
    t_w2c_solutions = loop(
        t_w2c=t_w2c_solutions,
        R_w2c=R_w2c,
        image_pairs=image_pairs,
        log_interval=log_interval,
    )  # (num_init, num_images, 3)

    # set invalid images to nan to expose subtle bugs
    t_w2c_solutions[:, ~image_mask] = torch.nan

    ##### Align the solutions #####
    # convert to c2w (that is, camera centers in world frame)
    t_c2w_solutions = torch.einsum(
        "nij,bnj->bni", R_w2c.transpose(-1, -2), -t_w2c_solutions
    )  # (num_init, num_images, 3)

    # offset the solutions to have zero geometric median
    t_c2w_centers = torch.zeros(
        num_init, 3, device=device, dtype=dtype
    )  # (num_init, 3)
    for i in range(num_init):
        t_c2w_centers[i] = geometric_median(t_c2w_solutions[i, image_mask])
    t_c2w_solutions -= t_c2w_centers[:, None, :]  # (num_init, num_images, 3)
    del t_c2w_centers

    # re-scale so that the geometric median of the norms is 1
    t_c2w_scales = torch.zeros(num_init, device=device, dtype=dtype)  # (num_init,)
    for i in range(num_init):
        t_c2w_scales[i] = geometric_median(
            t_c2w_solutions[i, image_mask].norm(dim=-1, keepdim=True)
        ).item()
    t_c2w_solutions /= t_c2w_scales[:, None, None]  # (num_init, num_images, 3)
    del t_c2w_scales

    # rotate the solutions to align with the first solution
    if num_init > 1:
        A = t_c2w_solutions[1:, image_mask]  # [num_solutions-1, num_valid_images, 3]
        B = t_c2w_solutions[0, image_mask].expand_as(
            A
        )  # [num_solutions-1, num_valid_images, 3]
        M = B.transpose(-1, -2) @ A  # [num_solutions-1, 3, 3]
        U, _, Vh = torch.linalg.svd(
            M
        )  # [num_solutions-1, 3, 3], [num_solutions-1, 3], [num_solutions-1, 3, 3]
        delta_R = U @ Vh  # [num_solutions-1, 3, 3]
        t_c2w_solutions[1:] = torch.einsum("bij,bnj->bni", delta_R, t_c2w_solutions[1:])
        del A, B, M, U, Vh, delta_R

    ##### Compute per image error #####

    # compute image pair error
    R_c2w2 = R_w2c[image_pairs.image_idx2].transpose(-1, -2)  # (num_image_pairs, 3, 3)
    o12_gt = torch.einsum(
        "bij,bj->bi", R_c2w2, -image_pairs.translation
    )  # (num_image_pairs, 3)
    del R_c2w2
    o12_pred = F.normalize(
        t_c2w_solutions[:, image_pairs.image_idx2]
        - t_c2w_solutions[:, image_pairs.image_idx1],
        dim=-1,
    )  # (num_init, num_image_pairs, 3)
    image_pair_error = (
        (o12_gt[None] - o12_pred).abs().mean(dim=-1)
    )  # (num_init, num_image_pairs)
    del o12_gt, o12_pred

    # compute per image error
    error = torch.zeros(
        num_init, num_images, device=device, dtype=dtype
    )  # (num_init, num_images)
    count = torch.zeros(
        num_init, num_images, device=device, dtype=dtype
    )  # (num_init, num_images)
    index1 = image_pairs.image_idx1[None].expand(
        num_init, -1
    )  # (num_init, num_image_pairs)
    index2 = image_pairs.image_idx2[None].expand(
        num_init, -1
    )  # (num_init, num_image_pairs)
    error.scatter_reduce_(
        dim=1,
        index=index1,
        src=image_pair_error,
        reduce="sum",
    )
    count.scatter_reduce_(
        dim=1,
        index=index1,
        src=torch.ones_like(image_pair_error),
        reduce="sum",
    )
    error.scatter_reduce_(
        dim=1,
        index=index2,
        src=image_pair_error,
        reduce="sum",
    )
    count.scatter_reduce_(
        dim=1,
        index=index2,
        src=torch.ones_like(image_pair_error),
        reduce="sum",
    )
    error /= count + 1e-6  # (num_init, num_images)

    # set error of invalid images to inf
    error[:, ~image_mask] = torch.inf

    ##### Get the solution with lowest error #####
    t_c2w = t_c2w_solutions[
        error.argmin(dim=0), torch.arange(num_images, device=device)
    ]  # (num_images, 3)

    # set invalid images to nan to expose subtle bugs
    t_c2w[~image_mask] = torch.nan

    ##### Final round of optimization #####
    t_w2c = torch.einsum("bij,bj->bi", R_w2c, -t_c2w)  # (num_images, 3)
    del t_c2w
    t_w2c = t_w2c[None]  # (1, num_images, 3)
    logger.info(
        f"Optimizing with merged results from {num_init} random initializations..."
    )
    t_w2c = loop(
        t_w2c=t_w2c,
        R_w2c=R_w2c,
        image_pairs=image_pairs,
        log_interval=log_interval,
    )  # (1, num_images, 3)
    t_w2c = t_w2c.squeeze(0)  # (num_images, 3)

    ##### Return #####
    # set invalid images to nan to expose subtle bugs
    t_w2c[~image_mask] = torch.nan
    return t_w2c
