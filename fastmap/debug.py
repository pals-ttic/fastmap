import time
from loguru import logger
import prettytable
import numpy as np
import torch

from fastmap.container import ColmapModel, Images, Cameras
from fastmap.utils import quantile_of_big_tensor


class DebugTimer:
    _disabled = False

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if self._disabled:
            return self
        torch.cuda.synchronize()  # ensure all previous operations are done
        self.start = time.perf_counter()
        return self  # allows use of `as` if needed

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._disabled:
            return
        torch.cuda.synchronize()  # ensure all previous operations are done
        end = time.perf_counter()
        elapsed = end - self.start
        logger.debug(f"[{self.name}] Elapsed time: {elapsed:.6f} seconds")

    @classmethod
    def disable(cls):
        """Disable all DebugTimer instances."""
        cls._disabled = True

    @classmethod
    def enable(cls):
        """Enable all DebugTimer instances."""
        cls._disabled = False


def pairwise_rotation_angle_error(
    R_w2c1: torch.Tensor,
    R_w2c2: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Compute a full *n × n* matrix of geodesic-angle errors (in degrees) between the
    **relative** camera rotations implied by two global world-to-camera rotation
    sets.

    Let

        R_rel1_ij = R_w2c1[j] @ R_w2c1[i].T        (1)
        R_rel2_ij = R_w2c2[j] @ R_w2c2[i].T        (2)

    be the relative rotations that carry camera *i* to camera *j* for the two
    solutions.  The metric used is the standard geodesic (arc-length) distance on
    SO(3):

        θ_ij = acos( (trace(R_rel1_ij.T @ R_rel2_ij) − 1) / 2 )  [radians]      (3)

    This function evaluates (3) for every ordered pair *(i,j)*, converts the
    result to **degrees**, and returns the *n × n* matrix **Θ** with
    Θ[i, j] = θ_ij.

    The computation is chunked so that at most `chunk_size × chunk_size` pairs
    are resident in memory at once, which keeps peak RAM usage at
    `O(chunk_size²)` regardless of *n*.  If *n* < `chunk_size` the code
    automatically falls back to a single pass.

    Parameters
    ----------
    R_w2c1 : torch.Tensor
        Tensor of shape *(n, 3, 3)* holding the first set of global
        world-to-camera rotation matrices.
    R_w2c2 : torch.Tensor
        Tensor of shape *(n, 3, 3)* holding the second set of global
        world-to-camera rotation matrices.
    chunk_size : int, optional
        Maximum number of indices processed along either axis per block.
        Defaults to **256**.

    Returns
    -------
    torch.Tensor
        Tensor of shape *(n, n)*.  Entry *(i, j)* equals the geodesic-angle error
        (in **degrees**) between the relative rotations of camera *i* and
        camera *j* implied by the two input solutions.

    Notes
    -----
    * Both input tensors must reside on the same device and have identical
      dtype; the output inherits those properties.
    * The result is **symmetric** and has an exact (or numerical-noise) zero
      diagonal.
    * Numerical safety: `acos` is applied to a value clamped to the interval
      *(–1, 1)* to guard against round-off.
    """
    if R_w2c1.shape != R_w2c2.shape or R_w2c1.ndim != 3 or R_w2c1.shape[1:] != (3, 3):
        raise ValueError(
            "Inputs must have identical shape (n, 3, 3); "
            f"got {R_w2c1.shape=} and {R_w2c2.shape=}."
        )

    n: int = R_w2c1.shape[0]
    device, dtype = R_w2c1.device, R_w2c1.dtype

    # Pre-allocate the output (shares device/dtype with the inputs).
    errors = torch.empty((n, n), dtype=dtype, device=device)

    # Loop over upper-triangular blocks; copy to the lower half later.
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        R1_i = R_w2c1[i0:i1]  # (Bi, 3, 3)
        R2_i = R_w2c2[i0:i1]
        R1_i_T = R1_i.transpose(1, 2)  # (Bi, 3, 3)
        R2_i_T = R2_i.transpose(1, 2)

        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            R1_j = R_w2c1[j0:j1]  # (Bj, 3, 3)
            R2_j = R_w2c2[j0:j1]

            # Broadcasted batch matmuls:
            #   R_rel1[b_i, b_j] = R1_j[b_j] @ R1_i_T[b_i]
            R_rel1 = torch.matmul(R1_j.unsqueeze(0), R1_i_T.unsqueeze(1))
            R_rel2 = torch.matmul(R2_j.unsqueeze(0), R2_i_T.unsqueeze(1))

            # trace(A.T @ B)  =  (A * B).sum(-1).sum(-1)
            cos_theta = ((R_rel1 * R_rel2).sum(dim=(-1, -2)) - 1.0) * 0.5
            cos_theta.clamp_(min=-1.0, max=1.0)  # numerical safety

            theta_deg = torch.rad2deg(torch.acos(cos_theta))  # (Bi, Bj)

            # Write the block and its symmetric counterpart
            errors[i0:i1, j0:j1] = theta_deg
            if i0 != j0:  # avoid double-writing the diagonal block
                errors[j0:j1, i0:i1] = theta_deg.transpose(0, 1)

    # Diagonal should be exactly zero; enforce to avoid tiny round-off.
    errors.diagonal().zero_()
    return errors


def pairwise_translation_angle_error(
    R_w2c1: torch.Tensor,
    R_w2c2: torch.Tensor,
    t_w2c1: torch.Tensor,
    t_w2c2: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    r"""
    Compute an :math:`N\times N` **angular‐error matrix** (in degrees) that
    compares *relative translation directions* derived from two pose sets.

    ----------
    Pose notation
    ----------
    Each pose is given as a **world‐to‐camera** transform
    :math:`(\mathbf R_{w\to c},\;\mathbf t_{w\to c})`:

    .. math::
        \mathbf x_c \;=\; \mathbf R_{w\to c}\,\mathbf x_w \;+\; \mathbf t_{w\to c}

    For a pair of views *(i,j)* **within the **same** pose set**, the
    translation of view *j* expressed in the coordinate frame of view *i* is

    .. math::
        \mathbf t^{(i)}_{i\to j}
        \;=\;
        \mathbf t_{w\to c_j}
        \;-\;
        \mathbf R_{w\to c_j}\,\mathbf R_{w\to c_i}^{\mathsf T}\,
        \mathbf t_{w\to c_i}
        \;=\;
        \mathbf t_{w\to c_i}
        \;-\;
        \mathbf R_{w\to c_i}\,\mathbf R_{w\to c_j}^{\mathsf T}\,
        \mathbf t_{w\to c_j}        (*)

    The **direction** obtained from (*) is *invariant* to any global rigid
    transform applied to the entire pose set, so the two sets need **not**
    share a common world frame.

    ----------
    Parameters
    ----------
    R_w2c1, R_w2c2 : torch.Tensor
        World‐to‐camera rotation matrices of the *first* and *second* pose sets;
        shape **(N, 3, 3)**.
    t_w2c1, t_w2c2 : torch.Tensor
        World‐to‐camera translation vectors of the two pose sets; shape
        **(N, 3)**.
    chunk_size : int, optional
        Number of *row* indices processed at once (default **256**).  Keeps the
        peak memory of the temporary tensors proportional to
        ``chunk_size × N``.

    ----------
    Returns
    ----------
    torch.Tensor
        Square tensor of shape **(N, N)** where entry *(i,j)* is the absolute
        angular error (degrees) between the unit translation directions
        :math:`\widehat{\mathbf t}^{(i)}_{i\to j}` of pose-set 1 and pose-set 2.
        The diagonal is set to **0**.

    ----------
    Notes for large *N*
    ----------
    Computing all :math:`N^2` relative translations naïvely would allocate an
    *(N,N,3)* tensor per pose set.  This implementation keeps the memory
    bounded to roughly ``chunk_size × N × 3`` by iterating **row‐wise**.
    """

    # --------------------------------------------------------------------- #
    # 0. Basic validation                                                   #
    # --------------------------------------------------------------------- #
    if R_w2c1.shape != R_w2c2.shape or R_w2c1.shape[-2:] != (3, 3):
        raise ValueError("Rotation tensors must both have shape (N,3,3).")
    if t_w2c1.shape != t_w2c2.shape or t_w2c1.shape[-1] != 3:
        raise ValueError("Translation tensors must both have shape (N,3).")
    if R_w2c1.shape[0] != t_w2c1.shape[0]:
        raise ValueError("Rotation and translation tensors must share the same N.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    N = R_w2c1.shape[0]
    device = R_w2c1.device
    dtype = R_w2c1.dtype

    # --------------------------------------------------------------------- #
    # 1. Pre-compute  p := Rᵀ t   for each view (per pose set)              #
    #    This avoids an extra batched matmul inside the double loop.        #
    # --------------------------------------------------------------------- #
    # (N,3,3)^T × (N,3,1) → (N,3)
    p1 = torch.bmm(R_w2c1.transpose(1, 2), t_w2c1.unsqueeze(2)).squeeze(2)  # Rᵀ t
    p2 = torch.bmm(R_w2c2.transpose(1, 2), t_w2c2.unsqueeze(2)).squeeze(2)

    # --------------------------------------------------------------------- #
    # 2. Allocate output error matrix                                       #
    # --------------------------------------------------------------------- #
    err_deg = torch.empty((N, N), dtype=dtype, device=device)

    # --------------------------------------------------------------------- #
    # 3. Row-wise chunked computation                                       #
    # --------------------------------------------------------------------- #
    eps = torch.finfo(dtype).eps  # numerical safety for normalisation
    for row_start in range(0, N, chunk_size):
        row_end = min(row_start + chunk_size, N)
        B = row_end - row_start  # rows in this chunk
        rows = slice(row_start, row_end)

        # 3-a) Gather per-row slices (pose set-1 and set-2)
        R1_i = R_w2c1[rows]  # (B,3,3)
        R2_i = R_w2c2[rows]  # (B,3,3)
        t1_i = t_w2c1[rows]  # (B,3)
        t2_i = t_w2c2[rows]  # (B,3)

        # 3-b) term2 :=  R_i · (R_jᵀ t_j)      — see equation (*)
        #       Shape (B,N,3) via efficient broadcasting with einsum.
        term2_1 = torch.einsum("bij,nj->bni", R1_i, p1)  # pose set-1
        term2_2 = torch.einsum("bij,nj->bni", R2_i, p2)  # pose set-2

        # 3-c) Relative translations t_rel_ij = t_i − term2
        t_rel1 = t1_i.unsqueeze(1) - term2_1  # (B,N,3)
        t_rel2 = t2_i.unsqueeze(1) - term2_2

        # 3-d) Normalise to unit length
        u1 = t_rel1 / t_rel1.norm(dim=-1, keepdim=True).clamp_min(eps)
        u2 = t_rel2 / t_rel2.norm(dim=-1, keepdim=True).clamp_min(eps)

        # 3-e) Angular error (in degrees)
        cos_theta = (u1 * u2).sum(-1).clamp(-1.0, 1.0)  # (B,N)
        err_block = torch.rad2deg(torch.acos(cos_theta))

        # 3-f) Insert block into the global matrix
        err_deg[rows, :] = err_block

    # --------------------------------------------------------------------- #
    # 4. Zero the diagonal (self-pairs)                                     #
    # --------------------------------------------------------------------- #
    diag = torch.arange(N, device=device)
    err_deg[diag, diag] = 0.0

    return err_deg


def _get_intersection_idx(
    images: Images,
    gt_model: ColmapModel,
):
    """Get a vector of image idx for the images being processed and the images in the ground truth model, such that indexing gives us the images in the intersection (with the correct order).
    Args:
        images: Images container
        gt_model: ColmapModel container, ground truth model
    Returns:
        pred_idx: torch.Tensor int (num_images_in_intersection,), indices for the images being processed
        gt_idx: torch.Tensor int (num_images_in_intersection,), indices for the images in the ground truth model
    """
    # get device
    device = images.device

    # get the set of valid image names
    valid_image_names = set(gt_model.names) & set(
        [name for i, name in enumerate(images.names) if images.mask[i]]
    )
    if len(valid_image_names) == 0:
        raise ValueError(
            "No valid images found in the intersection of ground truth and images. Are you sure the provided GT model was run on the same database?"
        )

    # sort the image idx according to image names
    pred_sort_idx = np.argsort(images.names).tolist()  # List[int] len=num_images
    gt_sort_idx = np.argsort(gt_model.names).tolist()  # List[int] len=num_gt_images

    # drop the images that are not in the intersection
    pred_idx = [
        idx for idx in pred_sort_idx if images.names[idx] in valid_image_names
    ]  # List[int] len=num_images_in_intersection
    gt_idx = [
        idx for idx in gt_sort_idx if gt_model.names[idx] in valid_image_names
    ]  # List[int] len=num_images_in_intersection
    assert len(pred_idx) == len(gt_idx)

    # convert to torch tensors
    pred_idx = torch.tensor(
        pred_idx, dtype=torch.long, device=device
    )  # (num_images_in_intersection,)
    gt_idx = torch.tensor(
        gt_idx, dtype=torch.long, device=device
    )  # (num_images_in_intersection,)

    # return
    return pred_idx, gt_idx


def log_pairwise_rotation_angle_error(
    R_w2c_pred: torch.Tensor,
    images: Images,
    gt_model: ColmapModel,
    quantiles: list = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99],
):
    """Compute and log the pairwise rotation angle error between two sets of global world-to-camera rotation matrices.
    Args:
        R_w2c_pred: torch.Tensor float (num_images, 3, 3), predicted global world-to-camera rotation matrices.
        images: Images container
        gt_model: ColmapModel container, ground truth model
        quantiles: list of quantiles to compute in addition to min and max values.
    """
    # init the log string
    log_str = "\nLogging pairwise rotation angle error for debugging:\n"

    # make sure the shapes are correct
    num_images = R_w2c_pred.shape[0]
    assert R_w2c_pred.shape == (num_images, 3, 3)
    log_str += f"Total number of images: {num_images}\n"

    # get the ground truth rotations
    R_w2c_gt = gt_model.rotation  # (num_gt_images, 3, 3)

    # get idx of images in the intersection of current images and ground truth images
    pred_idx, gt_idx = _get_intersection_idx(
        images, gt_model
    )  # (num_images_in_intersection,), (num_images_in_intersection,)

    # log the number of valid and ground truth images
    log_str += f"Number of valid images: {images.mask.long().sum().item()}\n"
    log_str += f"Number of ground truth images: {len(gt_model.names)}\n"

    log_str += f"Using {pred_idx.shape[0]} images for error computation (intersection of valid images and gt images)\n"

    # compute the pairwise rotation angle error and drop the diagonal values
    errors = pairwise_rotation_angle_error(
        R_w2c1=R_w2c_pred[pred_idx], R_w2c2=R_w2c_gt[gt_idx]
    )  # (num_valid_images, num_valid_images)
    errors = errors[
        ~torch.eye(errors.shape[0], dtype=torch.bool, device=errors.device)
    ]  # (num_off_diagonal_pairs,)
    del pred_idx, gt_idx

    # compute the quantiles of the errors
    values = {
        f"q{int(q * 100)}": quantile_of_big_tensor(errors, q).item() for q in quantiles
    }
    values["min"] = errors.min().item()
    values["max"] = errors.max().item()

    # make the table
    table = prettytable.PrettyTable()
    table.field_names = ["Quantile", "Rotation Angle Error (degrees)"]
    table.add_row(["min", f"{values['min']:.2f}"])
    for q in sorted(values.keys()):
        if q == "min" or q == "max":
            continue
        v = values[q]
        table.add_row([q, f"{v:.2f}"])
    table.add_row(["max", f"{values['max']:.2f}"])
    log_str += f"{table}\n"

    # log the string
    logger.debug(log_str)


def log_pairwise_translation_angle_error(
    R_w2c_pred: torch.Tensor,
    t_w2c_pred: torch.Tensor,
    images: Images,
    gt_model: ColmapModel,
    quantiles: list = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99],
):
    """Compute and log the pairwise relative translation angle error between two sets of global world-to-camera poses.
    Args:
        R_w2c_pred: torch.Tensor float (num_images, 3, 3), predicted global world-to-camera rotation matrices.
        t_w2c_pred: torch.Tensor float (num_images, 3), predicted global world-to-camera translation vectors.
        images: Images container
        gt_model: ColmapModel container, ground truth model
        quantiles: list of quantiles to compute in addition to min and max values.
    """
    # init the log string
    log_str = "\nLogging pairwise translation angle error for debugging:\n"

    # make sure the shapes are correct
    num_images = R_w2c_pred.shape[0]
    assert R_w2c_pred.shape == (num_images, 3, 3)
    log_str += f"Total number of images: {num_images}\n"

    # get the ground truth poses
    R_w2c_gt = gt_model.rotation  # (num_gt_images, 3, 3)
    t_w2c_gt = gt_model.translation  # (num_gt_images, 3)

    # get idx of images in the intersection of current images and ground truth images
    pred_idx, gt_idx = _get_intersection_idx(
        images, gt_model
    )  # (num_images_in_intersection,), (num_images_in_intersection,)

    # log the number of valid and ground truth images
    log_str += f"Number of valid images: {images.mask.long().sum().item()}\n"
    log_str += f"Number of ground truth images: {len(gt_model.names)}\n"

    log_str += f"Using {pred_idx.shape[0]} images for error computation (intersection of valid images and gt images)\n"

    # compute the pairwise translation angle error and drop the diagonal values
    errors = pairwise_translation_angle_error(
        R_w2c1=R_w2c_pred[pred_idx],
        R_w2c2=R_w2c_gt[gt_idx],
        t_w2c1=t_w2c_pred[pred_idx],
        t_w2c2=t_w2c_gt[gt_idx],
    )  # (num_valid_images, num_valid_images)
    errors = errors[
        ~torch.eye(errors.shape[0], dtype=torch.bool, device=errors.device)
    ]  # (num_off_diagonal_pairs,)
    del pred_idx, gt_idx

    # compute the quantiles of the errors
    values = {
        f"q{int(q * 100)}": quantile_of_big_tensor(errors, q).item() for q in quantiles
    }
    values["min"] = errors.min().item()
    values["max"] = errors.max().item()

    # make the table
    table = prettytable.PrettyTable()
    table.field_names = ["Quantile", "Translation Angle Error (degrees)"]
    table.add_row(["min", f"{values['min']:.2f}"])
    for q in sorted(values.keys()):
        if q == "min" or q == "max":
            continue
        v = values[q]
        table.add_row([q, f"{v:.2f}"])
    table.add_row(["max", f"{values['max']:.2f}"])
    log_str += f"{table}\n"

    # log the string
    logger.debug(log_str)


def log_pairwise_angle_error(
    R_w2c_pred: torch.Tensor,
    t_w2c_pred: torch.Tensor,
    images: Images,
    gt_model: ColmapModel,
    quantiles: list = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99],
):
    """Compute and log both the pairwise relative rotation and translation angle error between two sets of global world-to-camera poses.
    Args:
        R_w2c_pred: torch.Tensor float (num_images, 3, 3), predicted global world-to-camera rotation matrices.
        t_w2c_pred: torch.Tensor float (num_images, 3), predicted global world-to-camera translation vectors.
        images: Images container
        gt_model: ColmapModel container, ground truth model
        quantiles: list of quantiles to compute in addition to min and max values.
    """
    # init the log string
    log_str = "\nLogging pairwise rotation and translation angle error for debugging:\n"

    # make sure the shapes are correct
    num_images = R_w2c_pred.shape[0]
    assert R_w2c_pred.shape == (num_images, 3, 3)
    log_str += f"Total number of images: {num_images}\n"

    # get the ground truth poses
    R_w2c_gt = gt_model.rotation  # (num_gt_images, 3, 3)
    t_w2c_gt = gt_model.translation  # (num_gt_images, 3)

    # get idx of images in the intersection of current images and ground truth images
    pred_idx, gt_idx = _get_intersection_idx(
        images, gt_model
    )  # (num_images_in_intersection,), (num_images_in_intersection,)

    # log the number of valid and ground truth images
    log_str += f"Number of valid images: {images.mask.long().sum().item()}\n"
    log_str += f"Number of ground truth images: {len(gt_model.names)}\n"

    log_str += f"Using {pred_idx.shape[0]} images for error computation (intersection of valid images and gt images)\n"

    # compute the pairwise rotation angle error and drop the diagonal values
    R_errors = pairwise_rotation_angle_error(
        R_w2c1=R_w2c_pred[pred_idx], R_w2c2=R_w2c_gt[gt_idx]
    )  # (num_valid_images, num_valid_images)
    R_errors = R_errors[
        ~torch.eye(R_errors.shape[0], dtype=torch.bool, device=R_errors.device)
    ]  # (num_off_diagonal_pairs,)

    # compute the quantiles of the rotation errors
    R_values = {
        f"q{int(q * 100)}": quantile_of_big_tensor(R_errors, q).item()
        for q in quantiles
    }
    R_values["min"] = R_errors.min().item()
    R_values["max"] = R_errors.max().item()

    # compute the pairwise translation angle error and drop the diagonal values
    t_errors = pairwise_translation_angle_error(
        R_w2c1=R_w2c_pred[pred_idx],
        R_w2c2=R_w2c_gt[gt_idx],
        t_w2c1=t_w2c_pred[pred_idx],
        t_w2c2=t_w2c_gt[gt_idx],
    )  # (num_valid_images, num_valid_images)
    t_errors = t_errors[
        ~torch.eye(t_errors.shape[0], dtype=torch.bool, device=t_errors.device)
    ]  # (num_off_diagonal_pairs,)
    del pred_idx, gt_idx

    # compute the quantiles of the translation errors
    t_values = {
        f"q{int(q * 100)}": quantile_of_big_tensor(t_errors, q).item()
        for q in quantiles
    }
    t_values["min"] = t_errors.min().item()
    t_values["max"] = t_errors.max().item()

    # make the table
    table = prettytable.PrettyTable()
    table.field_names = [
        "Quantile",
        "Rotation Error (degrees)",
        "Translation Error (degrees)",
    ]
    table.add_row(["min", f"{R_values['min']:.2f}", f"{t_values['min']:.2f}"])
    assert tuple(sorted(R_values.keys())) == tuple(sorted(t_values.keys()))
    for q in sorted(R_values.keys()):
        if q == "min" or q == "max":
            continue
        R_v = R_values[q]
        t_v = t_values[q]
        table.add_row([q, f"{R_v:.2f}", f"{t_v:.2f}"])
    table.add_row(["max", f"{R_values['max']:.2f}", f"{t_values['max']:.2f}"])
    log_str += f"{table}\n"

    # log the string
    logger.debug(log_str)


def log_intrinsics(
    images: Images,
    cameras: Cameras,
    gt_model: ColmapModel,
):
    """Log and compare the estimated intrinsics (focal and distortion) with those of the ground truth model.
    Args:
        images: Images container
        cameras: Cameras container
        gt_model: ColmapModel container, ground truth model
    """
    # init the log string
    log_str = "\nLogging intrinsics for debugging:\n"

    # get the set of gt image names
    gt_image_names = set(gt_model.names)

    # create the dict for storing the intrinsics
    values = {"focal_pred": [], "focal_gt": [], "k1_pred": [], "k1_gt": []}

    # loop over the cameras
    for camera_idx in range(cameras.num_cameras):
        # write the predicted intrinsics
        values["focal_pred"].append(cameras.focal[camera_idx].item())
        values["k1_pred"].append(cameras.k1[camera_idx].item())

        # get the set of image names for this camera that are also in the ground truth model
        names = set(
            [
                name
                for i, name in enumerate(images.names)
                if cameras.camera_idx[i] == camera_idx and name in gt_image_names
            ]
        )

        # get the mask for the images that are in the ground truth model
        mask = torch.tensor(
            [name in names for name in gt_model.names],
            dtype=torch.bool,
            device=images.device,
        )  # (num_gt_images,)

        # write the ground truth intrinsics by averaging over the images
        if len(names) > 0:
            values["focal_gt"].append(gt_model.focal[mask].mean().item())
            values["k1_gt"].append(gt_model.k1[mask].mean().item())
        else:
            # if no images are found, write NaN
            values["focal_gt"].append(float("nan"))
            values["k1_gt"].append(float("nan"))

    # make the table
    table = prettytable.PrettyTable()
    table.field_names = [
        "Camera Idx",
        "Focal (Pred)",
        "Focal (GT Avg)",
        "k1 (Pred)",
        "k1 (GT Avg)",
    ]
    for camera_idx, (focal_pred, focal_gt, k1_pred, k1_gt) in enumerate(
        zip(
            values["focal_pred"], values["focal_gt"], values["k1_pred"], values["k1_gt"]
        )
    ):
        table.add_row(
            [
                camera_idx,
                f"{focal_pred:.2f}",
                f"{focal_gt:.2f}",
                f"{k1_pred:.5f}",
                f"{k1_gt:.5f}",
            ]
        )
    log_str += f"{table}\n"

    # log the string
    logger.debug(log_str)
