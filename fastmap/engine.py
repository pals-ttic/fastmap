from loguru import logger
import os
import torch
from dataclasses import asdict
import yaml

from fastmap.config import Config
from fastmap.io import read_model, write_model
from fastmap.timer import timer
from fastmap.container import ImagePairs, Tracks, PointPairs, Points3D
from fastmap.database import read_database
from fastmap.focal import estimate_focals, calibrate_matches
from fastmap.fundamental import re_estimate_fundamental
from fastmap.homography import re_estimate_homography
from fastmap.distortion import estimate_distortion, undistort_matches, alpha_to_k1
from fastmap.decompose import decompose
from fastmap.rel_t import re_estimate_relative_t
from fastmap.rotation import global_rotation
from fastmap.translation import global_translation
from fastmap.track import build_tracks
from fastmap.color import TrackColor2DReader
from fastmap.point_pair import point_pairs_from_tracks
from fastmap.epipolar import epipolar_adjustment
from fastmap.sparse import sparse_reconstruction
from fastmap.debug import (
    log_pairwise_rotation_angle_error,
    log_pairwise_translation_angle_error,
    log_pairwise_angle_error,
    log_intrinsics,
)


@torch.no_grad()
def engine(
    cfg: Config,
    device: str,
    database_path: str,
    output_dir: str | None = None,
    pinhole: bool = False,
    headless: bool = False,
    calibrated: bool = False,
    image_dir: str | None = None,
    gt_model_path: str | None = None,
):
    # start timer
    timer.start()

    # check if image_dir exists
    if image_dir is not None:
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image dir {image_dir} does not exist")
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"Image dir {image_dir} is not a directory")

    # set up output dir
    if output_dir is not None:
        # prevent overwriting
        if os.path.exists(output_dir):
            raise FileExistsError(f"Output dir {output_dir} already exists")
        # make output dir
        os.makedirs(output_dir, exist_ok=True)
        # set up logger
        log_path = os.path.join(output_dir, "log.txt")
        logger.add(log_path)

    # log and save config
    config_yaml: str = yaml.dump(
        asdict(cfg), sort_keys=False, default_flow_style=False
    ).strip()
    logger.info(f"Config:\n{config_yaml}")
    if output_dir is not None:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(config_yaml)
    del config_yaml

    # load gt model if provided
    if gt_model_path is not None:
        if not os.path.exists(gt_model_path):
            raise FileNotFoundError(f"GT model path {gt_model_path} does not exist")
        gt_model = read_model(
            model_path=gt_model_path,
            device=device,
        )
        logger.debug(f"Loaded GT model for debugging from {gt_model_path}")
    else:
        gt_model = None

    # read colmap database
    with timer("Read COLMAP Database"):
        logger.info(f"Reading database from {database_path}...")
        matches, cameras, images = read_database(
            database_path=database_path,
            device=device,
        )  # Matches, Cameras, Images
        logger.info(
            f"The database contains {images.num_images} images, {matches.num_image_pairs} image pairs, and {matches.num_point_pairs.sum().item()} point pairs"
        )

    # estimate intrinsics
    if calibrated:
        cameras.calibrated = True
        alpha = None
        T = None
        logger.info("Using intrinsics read from the database")

        # re-estimate matrices anyway
        with timer("Re-Estimate Matrices"):
            with timer("Fundamental"):
                re_estimate_fundamental(
                    matches, num_iters=cfg.fundamental.num_iters
                )  # inplace
            with timer("Homography"):
                re_estimate_homography(
                    matches, num_iters=cfg.homography.num_iters
                )  # inplace
    else:
        # distortion
        if not pinhole:
            with timer("Distortion"):
                # estimate distortion
                with timer("Estimation"):
                    alpha, T = estimate_distortion(
                        matches=matches,
                        cameras=cameras,
                        num_levels=cfg.distortion.num_levels,
                        num_samples=cfg.distortion.num_samples,
                        alpha_range=(
                            cfg.distortion.alpha_min,
                            cfg.distortion.alpha_max,
                        ),
                    )  # float, (3, 3)

                # undistort
                with timer("Undistort Matches"):
                    undistort_matches(
                        matches=matches, cameras=cameras, alpha=alpha, T=T
                    )  # inplace

                # re-estimate matrices
                with timer("Re-Estimate Matrices"):
                    with timer("Fundamental"):
                        re_estimate_fundamental(
                            matches, num_iters=cfg.fundamental.num_iters
                        )  # inplace
                    with timer("Homography"):
                        re_estimate_homography(
                            matches, num_iters=cfg.homography.num_iters
                        )  # inplace
        else:
            alpha = None
            T = None

        # estimate focal length after fundamental fine-tuning
        with timer("Focal Length"):
            # estimate focal length
            with timer("Estimation"):
                estimate_focals(
                    cameras=cameras,
                    matches=matches,
                    min_fov=cfg.focal.min_fov,
                    max_fov=cfg.focal.max_fov,
                    num_samples=cfg.focal.num_samples,
                    std=cfg.focal.std,
                )

    # calibrate matches
    with timer("Calibrate Matches"):
        calibrate_matches(matches=matches, cameras=cameras)

    # convert alpha to k1 used in OpenCV and COLMAP
    if (not pinhole) and (not calibrated):
        assert alpha is not None
        assert T is not None
        k1 = alpha_to_k1(alpha=alpha, T=T, cameras=cameras)
    else:
        assert alpha is None
        assert T is None
        k1 = torch.zeros_like(cameras.focal)
    cameras.k1 = k1

    # log debugging info for initial cameras
    if gt_model is not None:
        log_intrinsics(
            images=images,
            cameras=cameras,
            gt_model=gt_model,
        )

    # estimate relative pose (4 solutions)
    with timer("Relative Pose Decomposition"):
        image_pairs: ImagePairs = decompose(
            matches=matches, error_thr=cfg.pose_decomposition.error_thr
        )

    # estimate global w2c rotation
    with timer("Global Rotation Alignment"):
        R_w2c = global_rotation(
            images=images,
            image_pairs=image_pairs,
            max_inlier_thr=cfg.rotation.max_inlier_thr,
            min_inlier_thr=cfg.rotation.min_inlier_thr,
            min_inlier_increment_frac=cfg.rotation.min_inlier_increment_frac,
            max_angle_thr=cfg.rotation.max_angle_thr,
            min_angle_thr=cfg.rotation.min_angle_thr,
            angle_step=cfg.rotation.angle_step,
            lr=cfg.rotation.lr,
            log_interval=cfg.rotation.log_interval,
        )
    del image_pairs  # prevent misuse

    # log debugging info for global rotation
    if gt_model is not None:
        log_pairwise_rotation_angle_error(
            R_w2c_pred=R_w2c,
            images=images,
            gt_model=gt_model,
        )

    # build tracks container and extract point pairs
    with timer("Build Tracks"):
        tracks: Tracks = build_tracks(
            matches=matches,
            min_track_size=cfg.track.min_track_size,
        )

        # release memory of matches because it is no longer needed
        del matches

        # get all possible point pairs from tracks
        with timer("Get Point Pairs"):
            point_pairs: PointPairs = point_pairs_from_tracks(tracks=tracks)

    # asynchronously read color for 2D points in tracks
    if image_dir is not None:
        with timer("Populate Color Reading Queue"):
            color_reader = TrackColor2DReader(
                tracks=tracks,
                images=images,
                image_dir=image_dir,
                use_cpu=True,
            )
            color_reader.start()
        logger.info("Started asynchronous color reader for 2D points")
    else:
        color_reader = None

    # re-estimate relative translation from tracks and get new image pairs
    with timer("Re-Estimate Relative Translation"):
        (
            image_pairs,
            point_pair_mask,
        ) = re_estimate_relative_t(
            point_pairs=point_pairs,
            R_w2c=R_w2c,
            images=images,
            num_candidates=cfg.relative_translation.num_candidates,
            epipolar_error_thr=cfg.relative_translation.epipolar_error_thr,
            ray_angle_thr=cfg.relative_translation.ray_angle_thr,
            min_num_inliers=cfg.relative_translation.min_num_inliers,
            min_degree=cfg.relative_translation.min_degree,
        )  # ImagePairs, (num_point_pairs,)

    # translation alignment
    with timer("Global Translation Alignment"):
        t_w2c = global_translation(
            R_w2c=R_w2c,
            image_pairs=image_pairs,
            image_mask=images.mask,
            num_init=cfg.global_translation.num_init,
            log_interval=cfg.global_translation.log_interval,
        )  # (num_images, 3)

    # log debugging info for global translation
    if gt_model is not None:
        log_pairwise_translation_angle_error(
            R_w2c_pred=R_w2c, t_w2c_pred=t_w2c, images=images, gt_model=gt_model
        )

    # global epipolar optimization
    with timer("Global Epipolar Optimization"):
        R_w2c, t_w2c, focal_scale, point_pair_mask = epipolar_adjustment(
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            point_pairs=point_pairs,
            point_pair_mask=point_pair_mask,
            images=images,
            cameras=cameras,
            num_irls_steps=cfg.epipolar_adjustment.num_irls_steps,
            num_prune_steps=cfg.epipolar_adjustment.num_prune_steps,
            max_thr=cfg.epipolar_adjustment.max_thr,
            min_thr=cfg.epipolar_adjustment.min_thr,
            lr=cfg.epipolar_adjustment.lr,
            lr_decay=cfg.epipolar_adjustment.lr_decay,
            log_interval=cfg.epipolar_adjustment.log_interval,
        )  # (num_images, 3), (num_images, 3), (num_cameras,), (num_point_pairs,)

    # log debugging info for final poses
    if gt_model is not None:
        log_pairwise_angle_error(
            R_w2c_pred=R_w2c,
            t_w2c_pred=t_w2c,
            images=images,
            gt_model=gt_model,
        )

    # update cameras with focal scale
    cameras.focal *= focal_scale

    # log debugging info for final cameras
    if gt_model is not None:
        log_intrinsics(
            images=images,
            cameras=cameras,
            gt_model=gt_model,
        )

    # wait for the color reader to finish
    if color_reader is not None:
        logger.info("Waiting for color reader to finish...")
        with timer("Wait for Color Reading"):
            color2d = color_reader.join()
        logger.info("Color reader finished")
    else:
        color2d = None

    # sparse reconstruction
    with timer("Sparse Reconstruction"):
        points3d: Points3D = sparse_reconstruction(
            tracks=tracks,
            point_pairs=point_pairs,
            point_pair_mask=point_pair_mask,
            cameras=cameras,
            images=images,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            color2d=color2d,
            reproj_err_thr=cfg.sparse_reconstruction.reproj_err_thr,
            min_ray_angle=cfg.sparse_reconstruction.min_ray_angle,
            batch_size=cfg.sparse_reconstruction.batch_size,
        )

    # output
    if output_dir is not None:
        with timer("Write Results"):
            # always output only one model
            write_model(
                save_dir=os.path.join(output_dir, "sparse/0"),
                images=images,
                cameras=cameras,
                points3d=points3d,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
            )

    # log time
    timer.end()
    timer.log()

    # visualize poses and point cloud
    if not headless:
        from fastmap.vis import visualize

        visualize(
            R_w2c=R_w2c[images.mask],
            t_w2c=t_w2c[images.mask],
            xyz=points3d.xyz,
            rgb=points3d.rgb,
        )
