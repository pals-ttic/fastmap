from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmap.timer import timer
from fastmap.container import ImagePairs, Images
from fastmap.utils import (
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
    ConvergenceManager,
    find_connected_components,
)


def compute_rotation_angle_error(R1, R2, clamp_value=1.0, use_degree=True):
    """Compute the rotation angle error between two rotation matrices.
    Args:
        R1: torch.Tensor, float, shape=(..., 3, 3), rotation matrix
        R2: torch.Tensor, float, shape=(..., 3, 3), rotation matrix
        clamp_value: float, clamp the value to avoid numerical issues
        use_degree: bool, whether to return the angle in degrees
    Returns:
        angle: torch.Tensor, float, shape=(...), rotation angle error
    """
    # Compute the relative rotation matrix
    R = R1.transpose(-1, -2).contiguous() @ R2

    # Compute the trace of the relative rotation matrix
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Compute the rotation angle (in radians)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-clamp_value, clamp_value))

    # Convert radians to degrees
    if use_degree:
        angle = torch.rad2deg(angle)

    return angle


@torch.no_grad()
def initialization(
    images: Images,
    image_pairs: ImagePairs,
    image_pair_mask,
    batch_size: int = 4096 * 8,
):
    """Initialize global alignment of the camera rotations.

    Args:
        images: Images container
        image_pairs: ImagePairs container
        image_pair_mask: Optional torch.Tensor, bool, shape=(num_all_image_pairs,), mask for the valid image pairs
        batch_size: int, batch size for processing the dense constraint matrix. The batch size is assumed for 1024 images, and will be scaled according to the number of images

    Returns:
        R_w2c: torch.Tensor, float, shape=(num_images, 3, 3), world to camera rotation matrix
    """
    ##### Mask out the invalid image pairs #####
    if image_pair_mask is not None:
        rotation = image_pairs.rotation[image_pair_mask]  # (num_image_pairs, 3, 3)
        image_idx1 = image_pairs.image_idx1[image_pair_mask]  # (num_image_pairs,)
        image_idx2 = image_pairs.image_idx2[image_pair_mask]  # (num_image_pairs,)
    else:
        rotation = image_pairs.rotation  # (num_image_pairs, 3, 3)
        image_idx1 = image_pairs.image_idx1  # (num_image_pairs,)
        image_idx2 = image_pairs.image_idx2  # (num_image_pairs,)
    del image_pairs, image_pair_mask

    ##### Make sure the image pairs contain only the valid images #####
    assert images.mask[image_idx1].all()
    assert images.mask[image_idx2].all()

    ##### Make sure all valid images appear at least once in the image pairs #####
    _count = torch.zeros(
        images.num_images, dtype=torch.long, device=images.device
    )  # (num_images,)
    _count.scatter_add_(
        dim=0,
        index=image_idx1,
        src=torch.ones_like(image_idx1),
    )
    _count.scatter_add_(
        dim=0,
        index=image_idx2,
        src=torch.ones_like(image_idx2),
    )
    assert torch.all(_count[images.mask] > 0)
    del _count

    ##### Get information #####
    # get number of image pairs
    num_image_pairs = image_idx1.shape[0]
    # get number of images
    num_images = images.num_images
    # get device and dtype
    device = rotation.device
    dtype = rotation.dtype

    ##### Construct the constraints in COO sparse matrix #####
    # initialize memory for diagonal 3x3 blocks in A^T A
    AT_A_diag_values = torch.zeros(
        num_images, 3, 3, device=device, dtype=dtype
    )  # (num_images, 3, 3)

    # set the diagonal values for invalid images to force their results to be zero
    if images.num_invalid_images > 0:
        AT_A_diag_values[~images.mask] = (
            torch.eye(3, device=device, dtype=dtype)
            .expand(images.num_invalid_images, 3, 3)
            .clone()
        )

    # set the diagonal values for relative rotation constraints
    AT_A_diag_values.scatter_add_(
        dim=0,
        index=image_idx1.view(-1, 1, 1).expand(-1, 3, 3),
        src=rotation.transpose(-1, -2) @ rotation,
    )
    AT_A_diag_values.scatter_add_(
        dim=0,
        index=image_idx2.view(-1, 1, 1).expand(-1, 3, 3),
        src=torch.eye(3, device=device, dtype=dtype)
        .expand(num_image_pairs, 3, 3)
        .clone(),
    )

    # stack and flatten all the values
    assert torch.all(image_idx1 < image_idx2)
    AT_A_values = torch.cat(
        [AT_A_diag_values, -rotation.transpose(-1, -2), -rotation], dim=0
    )  # (num_values / 9, 3, 3)
    AT_A_values = AT_A_values.flatten()  # (num_values,)
    del AT_A_diag_values

    # scale the values by the number of images and image pairs
    AT_A_values *= float(num_images) / num_image_pairs  # (num_values,)

    # stack all the indices
    AT_A_row_idx = torch.cat(
        [torch.arange(num_images, device=device), image_idx1, image_idx2], dim=0
    )  # (num_values / 9,)
    AT_A_row_idx = AT_A_row_idx.view(-1, 1, 1) * 3 + torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]], device=device
    ).view(
        1, 3, 3
    )  # (num_values / 9, 3, 3)
    AT_A_row_idx = AT_A_row_idx.flatten()  # (num_values,)
    AT_A_col_idx = torch.cat(
        [torch.arange(num_images, device=device), image_idx2, image_idx1], dim=0
    )  # (num_values / 9,)
    AT_A_col_idx = AT_A_col_idx.view(-1, 1, 1) * 3 + torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]], device=device
    ).view(
        1, 3, 3
    )  # (num_values / 9, 3, 3)
    AT_A_col_idx = AT_A_col_idx.flatten()  # (num_values,)
    AT_A_indices = torch.stack([AT_A_row_idx, AT_A_col_idx], dim=0)  # (2, num_values)
    del AT_A_row_idx, AT_A_col_idx

    # construct the sparse matrix and solve
    assert AT_A_values.shape[0] == AT_A_indices.shape[1]
    AT_A = torch.sparse_coo_tensor(
        indices=AT_A_indices, values=AT_A_values, size=(num_images * 3, num_images * 3)
    )  # sparse_coo(num_images * 3, num_images * 3)

    # solve (first use SVD; if OOM, use lobpcg)
    try:
        logger.info(
            f"Solving x column using SVD on dense matrix of shape {' x '.join(map(str, AT_A.shape))}..."
        )
        # xcol = torch.linalg.svd(AT_A.to_dense()).Vh[-1]  # (num_images * 3,)
        xcol = torch.linalg.eigh(AT_A.to_dense()).eigenvectors[
            ..., 0
        ]  # (num_images * 3,)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise e
        logger.info(
            f"OOM detected when applying SVD on dense matrix of shape {' x '.join(map(str, AT_A.shape))}..."
        )
        logger.info(
            f"Solving x column using torch.lobpcg on sparse matrix of shape {' x '.join(map(str, AT_A.shape))}..."
        )
        xcol = torch.lobpcg(AT_A, k=1, largest=False, method="ortho")[
            1
        ].squeeze()  # (num_images * 3,)
    xcol = xcol.view(num_images, 3)  # (num_images, 3)
    del AT_A

    # normalize
    xcol = F.normalize(xcol, dim=-1)  # (num_images, 3)

    ##### Solve the y column of the rotation matrix #####
    # get the values
    AT_A_ortho_values = (
        xcol[images.mask][:, :, None] * xcol[images.mask][:, None, :]
    )  # (num_valid_images, 3, 3)
    AT_A_ortho_values = AT_A_ortho_values.flatten()  # (num_valid_images * 9,)

    # get the indices
    AT_A_ortho_row_idx = torch.arange(num_images, device=device)[
        images.mask
    ]  # (num_valid_images,)
    AT_A_row_idx = AT_A_ortho_row_idx.view(-1, 1, 1) * 3 + torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]], device=device
    ).view(
        1, 3, 3
    )  # (num_valid_images, 3, 3)
    AT_A_row_idx = AT_A_row_idx.flatten()  # (num_valid_images * 9,)
    AT_A_ortho_col_idx = torch.arange(num_images, device=device)[
        images.mask
    ]  # (num_valid_images,)
    AT_A_col_idx = AT_A_ortho_col_idx.view(-1, 1, 1) * 3 + torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]], device=device
    ).view(
        1, 3, 3
    )  # (num_valid_images, 3, 3)
    AT_A_col_idx = AT_A_col_idx.flatten()  # (num_valid_images * 9,)
    AT_A_ortho_indices = torch.stack(
        [AT_A_row_idx, AT_A_col_idx], dim=0
    )  # (2, num_valid_images * 9)
    del AT_A_row_idx, AT_A_col_idx

    # construct the sparse matrix
    assert AT_A_ortho_values.shape[0] == AT_A_ortho_indices.shape[1]
    AT_A = torch.sparse_coo_tensor(
        indices=AT_A_indices, values=AT_A_values, size=(num_images * 3, num_images * 3)
    ) + torch.sparse_coo_tensor(
        indices=AT_A_ortho_indices,
        values=AT_A_ortho_values,
        size=(num_images * 3, num_images * 3),
    )  # sparse_coo(num_images * 3, num_images * 3)
    AT_A = AT_A.coalesce()
    del AT_A_values, AT_A_indices, AT_A_ortho_values, AT_A_ortho_indices

    # solve (first use SVD; if OOM, use lobpcg)
    try:
        logger.info(
            f"Solving y column using SVD on dense matrix of shape {' x '.join(map(str, AT_A.shape))}..."
        )
        # ycol = torch.linalg.svd(AT_A.to_dense()).Vh[-1]  # (num_images * 3,)
        ycol = torch.linalg.eigh(AT_A.to_dense()).eigenvectors[
            ..., 0
        ]  # (num_images * 3,)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise e
        logger.info(
            f"OOM detected when applying SVD on dense matrix of shape {' x '.join(map(str, AT_A.shape))}..."
        )
        logger.info(
            f"Solving y column using torch.lobpcg on sparse matrix of shape {' x '.join(map(str, AT_A.shape))}..."
        )
        ycol = torch.lobpcg(AT_A, k=1, largest=False, method="ortho")[
            1
        ].squeeze()  # (num_images * 3,)
    ycol = ycol.view(num_images, 3)  # (num_images, 3)
    del AT_A

    # make sure it is orthogonal to the x column
    ycol = ycol - (xcol * ycol).sum(-1, keepdim=True) * xcol  # (num_images, 3)

    # normalize
    ycol = F.normalize(ycol, dim=-1)  # (num_images, 3)

    ##### Get the z column of the rotation matrix #####
    zcol = torch.cross(xcol, ycol, dim=-1)  # (num_images, 3)

    ##### Assemble the rotation matrix #####
    R_w2c = torch.stack([xcol, ycol, zcol], dim=-1)  # (num_images, 3, 3)
    logger.info("Done.")

    ##### Set invalid images to nan #####
    R_w2c[~images.mask] = torch.nan

    ##### return the rotation matrix #####
    return R_w2c


class TorchComputeGradientModule(nn.Module):
    def __init__(self, clamp_thr: float = 0.9999999):
        super().__init__()
        self.clamp_thr = clamp_thr

    @torch.no_grad()
    def forward(
        self,
        R_rel: torch.Tensor,  # float (B, 3, 3)
        R_w2c1: torch.Tensor,  # float (B, 3, 3)
        R_w2c2: torch.Tensor,  # float (B, 3, 3)
    ):
        clamp_thr = self.clamp_thr
        R1 = R_rel @ R_w2c1  # (B, 3, 3)
        R2 = R_w2c2  # (B, 3, 3)
        # Frobenius inner product: tr(R1^T R2) = sum_ij R1_ij * R2_ij
        trace = (R1 * R2).sum(dim=(-1, -2))  # shape (...,)
        cos_value = (trace - 1.0) * 0.5  # shape (...,)
        clamp_mask = (cos_value > -clamp_thr) & (cos_value < clamp_thr)
        cos_value_clamped = cos_value.clamp(
            min=-clamp_thr, max=clamp_thr
        )  # detach-free clamp
        angle = torch.acos(cos_value_clamped)  # radians
        loss = angle.mean()  # shape ()

        d_angle = torch.ones((len(R1),), device=R1.device, dtype=R1.dtype) / len(R1)

        d_cos_value_clamped = -d_angle / torch.sqrt(
            1.0 - cos_value_clamped * cos_value_clamped
        )

        d_cos_value = d_cos_value_clamped * clamp_mask.to(
            d_cos_value_clamped
        )  # shape (...,)

        d_trace = d_cos_value * 0.5  # shape (...,)

        d_R1 = d_trace[..., None, None] * R2  # shape (..., 3, 3)
        d_R2 = d_trace[..., None, None] * R1  # shape (..., 3, 3)

        d_R_w2c1 = R_rel.transpose(-1, -2) @ d_R1  # (B, 3, 3)
        d_R_w2c2 = d_R2  # (B, 3, 3)

        return loss, d_R_w2c1, d_R_w2c2


class CUDAComputeGradientModule(nn.Module):
    def __init__(self, clamp_thr: float = 0.9999999):
        super().__init__()
        self.clamp_thr = clamp_thr
        self._initialized = False
        from fastmap.cuda import rotation_gradient

        self.gradient_fn = rotation_gradient

    @torch.no_grad()
    def forward(
        self,
        R_rel: torch.Tensor,  # float (B, 3, 3)
        R_w2c1: torch.Tensor,  # float (B, 3, 3)
        R_w2c2: torch.Tensor,  # float (B, 3, 3)
    ):
        if not self._initialized:
            # make sure everything is contiguous
            assert R_rel.is_contiguous()
            assert R_w2c1.is_contiguous()
            assert R_w2c2.is_contiguous()

            # get device and dtype
            device = R_rel.device
            dtype = R_rel.dtype

            # initialize the output tensors
            self.loss = torch.zeros((1,), device=device, dtype=dtype)  # scalar
            self.d_R_w2c1 = torch.zeros_like(R_w2c1)  # (B,3,3)
            self.d_R_w2c2 = torch.zeros_like(R_w2c2)  # (B,3,3)

            # set flag
            self._initialized = True

        self.gradient_fn(
            R_rel=R_rel,
            R_w2c1=R_w2c1,
            R_w2c2=R_w2c2,
            loss=self.loss,
            d_R_w2c1=self.d_R_w2c1,
            d_R_w2c2=self.d_R_w2c2,
            clamp_thr=self.clamp_thr,
        )

        return (
            self.loss,
            self.d_R_w2c1,
            self.d_R_w2c2,
        )


@torch.no_grad()
def loop(
    R_w2c: torch.Tensor,
    image_pairs: ImagePairs,
    image_pair_mask: torch.Tensor,
    lr: float = 0.0001,
    log_interval: int = 500,
):
    """Global rotation alignment loop.

    Args:
        R_w2c: torch.Tensor, float, shape=(num_images, 3, 3), initial world to camera rotation matrix
        image_pairs: ImagePairs container
        image_pair_mask: torch.Tensor, bool, shape=(num_all_image_pairs,), mask for the valid image pairs
        lr: float, learning rate
        log_interval: int, log interval in number of iterations

    Returns:
        R_w2c: torch.Tensor, float, shape=(num_images, 3, 3), world to camera rotation matrix
    """
    ##### Mask out the invalid image pairs #####
    rotation = image_pairs.rotation[image_pair_mask]  # (num_image_pairs, 3, 3)
    image_idx1 = image_pairs.image_idx1[image_pair_mask]  # (num_image_pairs,)
    image_idx2 = image_pairs.image_idx2[image_pair_mask]  # (num_image_pairs,)
    del image_pairs, image_pair_mask

    try:
        compute_gradent = CUDAComputeGradientModule()
    except ImportError:
        logger.warning(
            "CUDA kernel extension for global rotation alignment is not available, falling back to the slower PyTorch implementation."
        )
        compute_gradent = TorchComputeGradientModule()

    ##### Finetune with gradient descent #####
    with torch.enable_grad():
        # initialize the parameters and optimizer
        rot6d_w2c = rotation_matrix_to_6d(R_w2c)  # (num_images, 6)
        params = nn.Parameter(
            rot6d_w2c.clone(),
            requires_grad=True,
        )  # (num_images, 4)
        optimizer = torch.optim.Adam([params], lr=lr)

        # convergence manager
        convergence_manager = ConvergenceManager(
            warmup_steps=5, decay=0.95, convergence_window=10
        )
        convergence_manager.start()

        # loop for optimization
        for iter_idx in range(1000000):
            # forward
            R_w2c1 = rotation_6d_to_matrix(
                torch.index_select(input=params, dim=0, index=image_idx1)
            )  # (num_trainable_image_pairs, 3, 3)
            R_w2c2 = rotation_6d_to_matrix(
                torch.index_select(input=params, dim=0, index=image_idx2)
            )  # (num_trainable_image_pairs, 3, 3)

            # compute gradient
            loss, d_R_w2c1, d_R_w2c2 = compute_gradent(
                R_w2c1=R_w2c1,
                R_w2c2=R_w2c2,
                R_rel=rotation,
            )

            # backward to parameters
            optimizer.zero_grad()
            torch.autograd.backward(
                tensors=[R_w2c1, R_w2c2],
                grad_tensors=[d_R_w2c1, d_R_w2c2],
            )

            # gradient step
            optimizer.step()

            # check convergence
            moving_loss, if_converged = convergence_manager.step(
                step=iter_idx, loss=loss
            )
            if if_converged:
                logger.info(
                    f"Loop finished at iter {iter_idx} with moving loss {moving_loss:.6f}"
                )
                break

            if iter_idx % log_interval == 0:
                logger.info(
                    f"[Iter {iter_idx}] loss={loss.item():.6f}, moving_loss={moving_loss:.6f}"
                )

    # convert back to rotation matrix
    R_w2c = rotation_6d_to_matrix(params.data)  # (num_images, 3, 3)

    ##### return the rotation matrix #####
    return R_w2c


@torch.no_grad()
def global_rotation(
    images: Images,
    image_pairs: ImagePairs,
    max_inlier_thr: int = 128,
    min_inlier_thr: int = 16,
    min_inlier_increment_frac: float = 0.01,
    max_angle_thr: float = 50.0,
    min_angle_thr: float = 10.0,
    angle_step: float = 10.0,
    lr: float = 0.0001,
    log_interval: int = 500,
):
    """Global alignment of the camera rotations using relative poses. Also update the image mask to remove sparsely connected images.

    Args:
        images: Images container
        image_pairs: ImagePairs container
        max_inlier_thr: int, only consider image pairs with number of inliers exceeding this threshold. It will be halved until the graph is connected or min_inlier_thr is reached.
        min_inlier_thr: int, minimum number of inliers to consider an image pair. If the graph is still disconnected, the images not in the largest connected component will be ignored, and the corresponding value in image_mask will be set to False.
        min_inlier_increment_frac: float, minimum increment fraction for the inlier threshold. The inlier threshold halving will be stopped if the number of new images in the largest connected component is less than num_images * min_inlier_increment_frac.
        max_angle_thr: float, maximum rotation angle threshold to consider an image pair as a valid constraint
        min_angle_thr: float, minimum rotation angle threshold to consider an image pair as a valid constraint
        angle_step: float, step size for the rotation angle threshold decrement
        lr: float, learning rate for the optimization
        log_interval: int, log interval in number of iterations

    Returns:
        R_w2c: torch.Tensor, float, shape=(num_images, 3, 3), world to camera rotation matrix
        In place update the image mask in the Images container
    """
    ##### Find a threshold for number of inliers #####
    min_inlier_increment = max(1, int(images.num_images * min_inlier_increment_frac))
    logger.info(
        f"Using min_inlier_increment = {min_inlier_increment}. The inlier threshold halving will be stopped if the number of newly added images in the largest connected component is less than {min_inlier_increment}."
    )
    inlier_thr = max_inlier_thr
    del max_inlier_thr, min_inlier_increment_frac
    num_images_in_largest_component = 0
    prev_num_added_images = 0
    while inlier_thr >= min_inlier_thr:
        # get the mask for the image pairs under the current inlier threshold
        tentative_mask = image_pairs.num_inliers >= inlier_thr  # (num_image_pairs,)

        # find the connected components
        component_idx = find_connected_components(
            image_idx1=image_pairs.image_idx1,
            image_idx2=image_pairs.image_idx2,
            num_images=images.num_images,
            image_pair_mask=tentative_mask,
        )
        num_components = component_idx.max().item() + 1

        # get the number of newly added images in the largest component
        _, counts = component_idx.unique(return_counts=True)  # (num_components,)
        num_added_images = (
            counts[component_idx] == counts.max()
        ).long().sum().item() - num_images_in_largest_component
        del counts
        logger.info(
            f"Newly added images in the largest component: {num_added_images} out of {images.num_images}."
        )

        # stopping criterion
        if num_components == 1:  # stop if the graph is connected
            logger.info(f"Final inlier thr = {inlier_thr} (all images are connected).")
            break
        elif (
            # make sure the number of newly added images is not increasing
            num_added_images <= prev_num_added_images
            # stop if the number of newly added images is too small
            and num_added_images < min_inlier_increment
        ):
            logger.info(
                f'"Final inlier thr = {inlier_thr} (the number of newly added images {num_added_images} is below the minimum increment threshold {min_inlier_increment}).'
            )
            break
        # stop if halving the inlier threshold will make it too small
        elif inlier_thr // 2 < min_inlier_thr:
            inlier_thr = min_inlier_thr
            logger.info(
                f"Final inlier thr = {min_inlier_thr}. The minimum inlier threshold {min_inlier_thr} is reached."
            )
            break
        else:
            inlier_thr = inlier_thr // 2
            logger.info(f"Graph disconnected: inlier thr halved to {inlier_thr}")

        # update
        num_images_in_largest_component += num_added_images
        prev_num_added_images = num_added_images
        del num_added_images

    logger.info(f"Using inlier thr = {inlier_thr}")
    del num_images_in_largest_component

    ##### Update image mask and get the image pair mask #####
    # image pair mask
    image_pair_mask = image_pairs.num_inliers >= inlier_thr

    # update image mask
    component_idx = find_connected_components(
        image_idx1=image_pairs.image_idx1,
        image_idx2=image_pairs.image_idx2,
        num_images=images.num_images,
        image_pair_mask=image_pair_mask,
    )  # (num_images,)
    _, counts = component_idx.unique(return_counts=True)  # (num_components,)
    image_mask = counts[component_idx] == counts.max()  # (num_images,)
    images.mask &= image_mask
    logger.info(
        f"Number of images in largest component: {image_mask.long().sum().item()} out of {images.num_images}"
    )
    del component_idx, counts, image_mask

    # eliminate the image pairs involving invalid images
    image_pair_mask &= (
        images.mask[image_pairs.image_idx1] & images.mask[image_pairs.image_idx2]
    )
    assert images.mask[image_pairs.image_idx1[image_pair_mask]].all()
    assert images.mask[image_pairs.image_idx2[image_pair_mask]].all()
    logger.info(
        f"Using {image_pair_mask.long().sum().item()} out of {image_pairs.num_image_pairs} image pairs."
    )

    ##### Initialize global rotation #####
    with timer("Initialization"):
        # R_w2c = initialization(
        R_w2c = initialization(
            images=images,
            image_pairs=image_pairs,
            image_pair_mask=image_pair_mask,
        )

    ##### Optimization #####
    with timer("Optimization"):
        angle_thr = max_angle_thr

        while angle_thr >= min_angle_thr:
            # loop
            R_w2c = loop(
                R_w2c=R_w2c,
                image_pairs=image_pairs,
                image_pair_mask=image_pair_mask,
                lr=lr,
                log_interval=log_interval,
            )  # (num_images, 3, 3)

            # eliminate the image pairs beyond the angle threshold
            error = compute_rotation_angle_error(
                R1=R_w2c[image_pairs.image_idx2]
                @ R_w2c[image_pairs.image_idx1].transpose(-1, -2),
                R2=image_pairs.rotation,
                clamp_value=0.9999999,
                use_degree=True,
            )  # (num_image_pairs,)

            # get new image pair mask
            image_pair_mask = (error <= angle_thr) & (
                image_pairs.num_inliers >= min_inlier_thr
            )  # (num_image_pairs,)

            # eliminate the image pairs involving invalid images
            image_pair_mask &= (
                images.mask[image_pairs.image_idx1]
                & images.mask[image_pairs.image_idx2]
            )
            assert images.mask[image_pairs.image_idx1[image_pair_mask]].all()
            assert images.mask[image_pairs.image_idx2[image_pair_mask]].all()

            # check if the graph is connected
            component_idx = find_connected_components(
                image_idx1=image_pairs.image_idx1,
                image_idx2=image_pairs.image_idx2,
                num_images=images.num_images,
                image_pair_mask=image_pair_mask,
            )  # (num_images,)
            _, component_counts = component_idx.unique(
                return_counts=True
            )  # (num_components,)
            if component_counts.max() < images.mask.long().sum():
                logger.info(f"Graph disconnected at angle thr = {angle_thr}")
                break

            # decrease the angle threshold
            angle_thr -= angle_step

    ##### set rotation to nan for invalid images #####
    R_w2c[~images.mask] = torch.nan

    return R_w2c
