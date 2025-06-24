import typing
from typing import Union, Optional

import torch
import torch.nn.functional as F


def to_homogeneous(x):
    """Convert a batch of vectors to homogeneous coordinates
    Args:
        x: torch.Tensor (..., D) where D is the dimension of the vector
    Returns:
        x_h: torch.Tensor (..., D+1)
    """
    ones = torch.ones_like(x[..., :1])  # (..., 1)
    x_h = torch.cat([x, ones], dim=-1)  # (..., D+1)
    return x_h


def vector_to_skew_symmetric_matrix(vectors):
    """
    Create skew-symmetric matrices from (..., 3) vectors.

    Args:
        vectors (torch.Tensor): shape (..., 3)

    Returns:
        torch.Tensor: shape (..., 3, 3) skew-symmetric matrices
    """
    assert vectors.shape[-1] == 3, "Last dimension must be 3"

    zero = torch.zeros_like(vectors[..., :1])

    vx = vectors[..., 0:1]
    vy = vectors[..., 1:2]
    vz = vectors[..., 2:3]

    skew = torch.cat(
        [
            torch.cat([zero, -vz, vy], dim=-1).unsqueeze(-2),
            torch.cat([vz, zero, -vx], dim=-1).unsqueeze(-2),
            torch.cat([-vy, vx, zero], dim=-1).unsqueeze(-2),
        ],
        dim=-2,
    )

    return skew


def skew_symmetric_matrix_to_vector(skew_matrices):
    """
    Convert skew-symmetric matrices (..., 3, 3) back to vectors (..., 3).

    Args:
        skew_matrices (torch.Tensor): shape (..., 3, 3)

    Returns:
        torch.Tensor: shape (..., 3)
    """
    assert skew_matrices.shape[-2:] == (3, 3), "Input must have shape (..., 3, 3)"

    vx = skew_matrices[..., 2, 1]
    vy = skew_matrices[..., 0, 2]
    vz = skew_matrices[..., 1, 0]

    vectors = torch.stack([vx, vy, vz], dim=-1)
    return vectors


def rotation_matrix_to_6d(R):
    """Convert a batch of rotation matrices to 6D representation
    Args:
        R: torch.Tensor (..., 3, 3)
    Returns:
        rot6d: torch.Tensor (..., 6)
    """
    assert R.shape[-2:] == (3, 3)
    R = R.clone()
    return R[..., :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def rotation_6d_to_matrix(rot6d):
    """Convert a batch of 6D rotation representations to rotation matrices
    Args:
        rot6d: torch.Tensor (..., 6)
    Returns:
        R: torch.Tensor (..., 3, 3)
    """
    assert rot6d.shape[-1] == 6
    col1 = F.normalize(rot6d[..., :3], dim=-1)  # (..., 3)
    col2 = F.normalize(
        (rot6d[..., 3:] - (col1 * rot6d[..., 3:]).sum(dim=-1, keepdim=True) * col1),
        dim=-1,
    )  # (..., 3)
    col3 = torch.cross(col1, col2, dim=-1)  # (..., 3)
    R = torch.stack([col1, col2, col3], dim=-1)  # (..., 3, 3)
    return R


def axis_angle_to_rotation_matrix(axis_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of axis-angle vectors to rotation matrices using Rodrigues' rotation formula.

    Each axis-angle vector defines a 3D rotation:
        - The direction of the vector is the axis of rotation.
        - The norm of the vector is the rotation angle (in radians).

    Args:
        axis_angles (Tensor): Tensor of shape (B, 3) containing axis-angle vectors.

    Returns:
        Tensor: Tensor of shape (B, 3, 3) containing corresponding rotation matrices.

    Notes:
        - The conversion is differentiable.
        - For small angles, stability is maintained using a small epsilon.
    """
    assert (
        axis_angles.ndim == 2 and axis_angles.shape[1] == 3
    ), "Input must be of shape (B, 3)"

    B = axis_angles.shape[0]
    eps = 1e-8

    # Compute the angle (magnitude) of each axis-angle vector: (B, 1)
    angles = torch.norm(axis_angles, dim=1, keepdim=True)  # shape (B, 1)

    # Normalize to get unit vectors: (B, 3)
    axes = axis_angles / (angles + eps)

    # Compute sine and cosine of the angles
    sin = torch.sin(angles).view(-1, 1, 1)  # (B, 1, 1)
    cos = torch.cos(angles).view(-1, 1, 1)  # (B, 1, 1)
    one_minus_cos = 1.0 - cos

    # Extract x, y, z components
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]  # Each (B,)

    # Construct skew-symmetric matrices for each axis: (B, 3, 3)
    K = torch.zeros((B, 3, 3), dtype=axis_angles.dtype, device=axis_angles.device)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x

    # Outer product u u^T for each normalized axis: (B, 3, 3)
    axis_outer = axes.unsqueeze(2) @ axes.unsqueeze(1)

    # Identity matrix for each batch: (B, 3, 3)
    I = torch.eye(3, device=axis_angles.device).expand(B, 3, 3)

    # Rodrigues' rotation formula
    R = cos * I + sin * K + one_minus_cos * axis_outer

    return R


@torch.no_grad()
def quantile_of_big_tensor(tensor: torch.Tensor, q: float):
    sorted = tensor.flatten().sort().values
    idx = int((len(sorted) - 1) * q)
    idx = min(len(sorted) - 1, max(0, idx))
    quantile = sorted[idx]
    return quantile


class ConvergenceManager:
    def __init__(
        self,
        warmup_steps: int,
        decay: float,
        convergence_window: int,
        margin: float = 0.0,
    ):
        """
        Args:
            warmup_steps (int): number of steps of accumulate the initial moving loss
            decay (float): the moving loss decay
            convergence_window (int): the window size for convergence check
            margin (float): the margin for convergence check
        """
        self.warmup_steps = warmup_steps
        self.decay = decay
        self.convergence_window = convergence_window
        self.margin = margin

        self.start()

    def start(self):
        self._moving_loss = None
        self._min_moving_loss = None
        self._min_moving_loss_step = None

    def step(self, step: int, loss: Union[torch.Tensor, float]):
        """Returns tuple (moving_loss, if_converged)"""

        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        # update moving loss
        if self._moving_loss is None:
            self._moving_loss = loss
            self._min_moving_loss = loss
            self._min_moving_loss_step = step
        else:
            self._moving_loss = self.decay * self._moving_loss + (1 - self.decay) * loss
        assert isinstance(self._moving_loss, float)
        assert isinstance(self._min_moving_loss, float)
        assert isinstance(self._min_moving_loss_step, int)

        # set min_moving_loss_step and check convergence
        if_converged = False
        if (
            step < self.warmup_steps
            or self._moving_loss <= self._min_moving_loss - self.margin
        ):
            # update min moving loss and step if still in warmup or current loss is smaller
            self._min_moving_loss = self._moving_loss
            self._min_moving_loss_step = step
        elif step - self._min_moving_loss_step >= self.convergence_window:
            if_converged = True
        else:
            pass

        # return
        return self._moving_loss, if_converged


@torch.no_grad()
def find_connected_components(
    image_idx1: torch.Tensor,
    image_idx2: torch.Tensor,
    num_images: int,
    image_pair_mask: Optional[torch.Tensor] = None,
):
    """Find the connected components in the pose graph
    Args:
        image_idx1: torch.Tensor, long, shape=(num_image_pairs), the first image index for each image pair
        image_idx2: torch.Tensor, long, shape=(num_image_pairs), the second image index for each image pair
        num_images: int, number of images
        image_pair_mask: torch.Tensor, bool, shape=(num_image_pairs), mask for valid image pairs
    Returns:
        component_idx: torch.Tensor, long, shape=(num_images,), the connected component index for each image (starting with 0)
    """
    device = image_idx1.device

    # initialize the smallest image idx for each connected component
    smallest_idx = torch.arange(
        num_images, device=device, dtype=torch.long
    )  # (num_images,)

    # get the valid image pairs
    if image_pair_mask is not None:
        image_idx1 = image_idx1[
            image_pair_mask
        ]  # (num_image_pairs,) note that here num_image_pairs is the number of valid image pairs
        image_idx2 = image_idx2[
            image_pair_mask
        ]  # (num_image_pairs,), note that here num_image_pairs is the number of valid image pairs

    # iteratively update the smallest image idx
    converge_flag = False
    while not converge_flag:
        converge_flag = True
        current_smallest_idx1 = smallest_idx[image_idx1]  # (num_image_pairs,)
        current_smallest_idx2 = smallest_idx[image_idx2]  # (num_image_pairs,)
        if torch.any(current_smallest_idx1 != current_smallest_idx2):
            converge_flag = False
            current_smallest_idx = torch.min(
                current_smallest_idx1, current_smallest_idx2
            )  # (num_image_pairs,)
            smallest_idx.scatter_reduce_(
                dim=0, index=image_idx1, src=current_smallest_idx, reduce="min"
            )
            smallest_idx.scatter_reduce_(
                dim=0, index=image_idx2, src=current_smallest_idx, reduce="min"
            )

    # get the connected component idx
    component_idx = torch.zeros_like(smallest_idx)  # (num_images,)
    unique_smallest_idx = torch.unique(smallest_idx)
    for new_idx in range(len(unique_smallest_idx)):
        component_idx[smallest_idx == unique_smallest_idx[new_idx]] = new_idx

    return component_idx


@torch.no_grad()
def packed_quantile(
    values: torch.Tensor,
    group_idx: torch.Tensor,
    q: float,
    num_iters: int = 20,
    num_groups: Optional[int] = None,
):
    """Find the quantile for each group of values where different groups have different number of values.

    Args:
        values: torch.Tensor float (num_values,), the values to find the quantile
        group_idx: torch.Tensor long (num_values,), the group index for each value
        q: float, the quantile to find
        num_iters: int, the number of iterations for the quantile estimation
        num_groups: Optional[int], the number of groups. If not provided, it is inferred from the group index
    Returns:
        quantile: torch.Tensor float (num_groups,), the quantile for each group
    """
    # get the device and dtype
    device, dtype = values.device, values.dtype

    # infer the number of groups if not provided
    if num_groups is None:
        num_groups = typing.cast(int, group_idx.max().item() + 1)
    assert num_groups is not None

    # initialize the group min and max
    group_min = torch.inf * torch.ones(
        num_groups, device=device, dtype=dtype
    )  # (num_groups,)
    group_min.scatter_reduce_(
        dim=0, index=group_idx, src=values, reduce="min"
    )  # (num_groups,)
    group_max = -torch.inf * torch.ones(
        num_groups, device=device, dtype=dtype
    )  # (num_groups,)
    group_max.scatter_reduce_(
        dim=0, index=group_idx, src=values, reduce="max"
    )  # (num_groups,)

    # iterate to find the quantile
    for _ in range(num_iters):
        # get the mid point
        mid_point = 0.5 * (group_min + group_max)  # (num_groups,)

        # compute the fraction of values below the mid point
        mask_below = values < mid_point[group_idx]  # (num_values,)
        fraction_below = torch.zeros(
            num_groups, device=device, dtype=dtype
        )  # (num_groups,)
        fraction_below.scatter_reduce_(
            dim=0,
            index=group_idx,
            src=mask_below.to(dtype),
            reduce="mean",
            include_self=False,
        )
        del mask_below

        # update the min and max
        group_min = torch.where(
            fraction_below < q, mid_point, group_min
        )  # (num_groups,)
        group_max = torch.where(
            fraction_below >= q, mid_point, group_max
        )  # (num_groups,)

    # get the final quantile
    quantile = 0.5 * (group_min + group_max)  # (num_groups,)

    # return
    return quantile


def normalize_matrix(matrix):
    """Normalize a batch of matrices to have unit norm.
    Args:
        matrix (torch.Tensor): A tensor of shape (..., m, n) of matrices.
    Returns:
        matrix (torch.Tensor): A tensor of shape (..., m, n) of normalized matrices.
    """
    assert matrix.ndim >= 2
    n, m = matrix.shape[-2:]
    batch_dims = matrix.shape[:-2]
    matrix = matrix.view(-1, n * m)
    matrix = F.normalize(matrix, dim=-1)
    matrix = matrix.view(*batch_dims, n, m)
    return matrix


@torch.no_grad()
def geometric_median(points, eps=1e-5, max_iter=5000):
    """
    Compute the geometric median (the point minimizes the sum of distances to all the points) of a set of points using Weiszfeld's algorithm.

    Args:
        points (torch.Tensor): Tensor of shape (N, D), N points in D dimensions.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        torch.Tensor: 1D tensor of shape (D,) representing the geometric median.
    """
    y = points.mean(dim=0)

    for _ in range(max_iter):
        diff = points - y
        dist = torch.norm(diff, dim=1)
        mask = dist > eps

        if mask.sum() == 0:
            break

        inv_dist = 1.0 / dist[mask]
        weighted_sum = (points[mask] * inv_dist.unsqueeze(1)).sum(dim=0)
        y_new = weighted_sum / inv_dist.sum()

        # Handle points exactly at current estimate
        num_zeros = (~mask).sum()
        if num_zeros > 0:
            y_new = (y_new * (1 - num_zeros / points.shape[0])) + (
                y * (num_zeros / points.shape[0])
            )

        if torch.norm(y - y_new) < eps:
            break

        y = y_new

    assert y.shape == (points.shape[1],)

    return y
