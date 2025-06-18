from loguru import logger
import prettytable
import torch

from fastmap.container import ColmapModel, Images
from fastmap.utils import quantile_of_big_tensor


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
            cos_theta.clamp_(min=-0.99999, max=0.99999)  # numerical safety

            theta_deg = torch.rad2deg(torch.acos(cos_theta))  # (Bi, Bj)

            # Write the block and its symmetric counterpart
            errors[i0:i1, j0:j1] = theta_deg
            if i0 != j0:  # avoid double-writing the diagonal block
                errors[j0:j1, i0:i1] = theta_deg.transpose(0, 1)

    # Diagonal should be exactly zero; enforce to avoid tiny round-off.
    errors.diagonal().zero_()
    return errors


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

    # get info
    device = R_w2c_pred.device

    # make sure the shapes are correct
    num_images = R_w2c_pred.shape[0]
    assert R_w2c_pred.shape == (num_images, 3, 3)
    log_str += f"Total number of images: {num_images}\n"

    # get the ground truth rotations
    R_w2c_gt = gt_model.rotation  # (num_gt_images, 3, 3)

    # get the set of valid image names
    valid_image_names = set(gt_model.names) & set(
        [name for i, name in enumerate(images.names) if images.mask[i]]
    )
    if len(valid_image_names) == 0:
        raise ValueError(
            "No valid images found in the intersection of ground truth and images. Are you sure the provided GT model was run on the same database?"
        )

    # log the number of valid and ground truth images
    log_str += f"Number of valid images: {images.mask.long().sum().item()}\n"
    log_str += f"Number of ground truth images: {len(gt_model.names)}\n"

    # make valid mask for the predicted rotations
    mask_pred = torch.tensor(
        [name in valid_image_names for name in images.names],
        dtype=torch.bool,
        device=device,
    )  # (num_images,)

    # make valid mask for the ground truth rotations
    mask_gt = torch.tensor(
        [name in valid_image_names for name in gt_model.names],
        dtype=torch.bool,
        device=device,
    )  # (num_images,)

    # log the number of images used for the error computation
    assert len(valid_image_names) == mask_pred.long().sum() == mask_gt.long().sum()
    log_str += f"Using {mask_pred.long().sum().item()} images for error computation (intersection of valid images and gt images)\n"

    # compute the pairwise rotation angle error and drop the diagonal values
    errors = pairwise_rotation_angle_error(
        R_w2c1=R_w2c_pred[mask_pred], R_w2c2=R_w2c_gt[mask_gt]
    )  # (num_valid_images, num_valid_images)
    errors = errors[
        ~torch.eye(errors.shape[0], dtype=torch.bool, device=errors.device)
    ]  # (num_off_diagonal_pairs,)
    del mask_pred, mask_gt

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
