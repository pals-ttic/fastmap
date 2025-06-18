import typing
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass
import torch


class COLMAP_TWO_VIEW_GEOMETRY_TYPE(Enum):
    # Undefined configuration
    UNDEFINED = 0
    # Degenerate configuration (e.g., no overlap or not enough inliers).
    DEGENERATE = 1
    # Essential matrix.
    CALIBRATED = 2
    # Fundamental matrix.
    UNCALIBRATED = 3
    # Homography, planar scene with baseline.
    PLANAR = 4
    # Homography, pure rotation without baseline.
    PANORAMIC = 5
    # Homography, planar or panoramic.
    PLANAR_OR_PANORAMIC = 6
    # Watermark, pure 2D translation in image borders.
    WATERMARK = 7
    # Multi-model configuration, i.e. the inlier matches result from multiple
    # individual, non-degenerate configurations.
    MULTIPLE = 8


@dataclass
class ColmapModel:
    """Container for estimated model read from COLMAP format."""

    # List[str], filenames of images
    names: List[str]
    # torch.Tensor, float, shape=(num_images, 3, 3), global rotation of w2c
    rotation: torch.Tensor
    # torch.Tensor, float, shape=(num_images, 3), global translation of w2c
    translation: torch.Tensor
    # torch.Tensor, float, shape=(num_images,), focal lengths in pixels
    focal: torch.Tensor
    # torch.Tensor, float, shape=(num_images,), k1 radial distortion coefficient (nan if pinhole)
    k1: torch.Tensor
    # None | torch.Tensor, float, shape=(num_points3d, 3), 3D coordinates of scene points
    points3d: None | torch.Tensor
    # None | torch.Tensor, uint8, shape=(num_points3d, 3), RGB color of scene points in [0, 255]
    rgb: None | torch.Tensor

    @property
    def device(self):
        return self.rotation.device

    @property
    def dtype(self):
        return self.rotation.dtype

    @property
    def num_images(self):
        return len(self.names)


@dataclass
class Images:
    """Container for storing some information of images."""

    # List[str] filenames of images
    names: List[str]
    # torch.Tensor, long, shape=(num_images,), width of each image
    widths: torch.Tensor
    # torch.Tensor, long, shape=(num_images,), height of each image
    heights: torch.Tensor
    # torch.Tensor, bool, shape=(num_images,), mask of images. True if the image is registered.
    mask: torch.Tensor

    @property
    def num_images(self):
        return len(self.names)

    @property
    def num_valid_images(self):
        n = self.mask.long().sum().item()
        n = typing.cast(int, n)
        assert n <= self.num_images
        return n

    @property
    def num_invalid_images(self):
        return self.num_images - self.num_valid_images

    @property
    @torch.no_grad()
    def device(self):
        return self.widths.device


@dataclass
class MatchesData:
    """Container for storing data of queries of matching results."""

    # torch.Tensor, bool, shape=(total_num_image_pairs,), a mask of image pairs. True if the data of the image pair is in this container.
    image_pair_mask: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), xy coordinates in pixels of keypoints in image 1
    xy1_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), xy coordinates in pixels of keypoints in image 2
    xy2_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), (calibrated) xy coordinates of keypoints in image 1
    xy1: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), (calibrated) xy coordinates of keypoints in image 2
    xy2: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), idx of image 1
    image_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), idx of image 2
    image_idx2: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), local keypoint idx of keypoint 1 in image 1
    keypoint_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), local keypoint idx of keypoint 2 in image 2
    keypoint_idx2: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), idx of the image pair
    image_pair_idx: torch.Tensor

    @property
    @torch.no_grad()
    def device(self):
        return self.xy1.device

    @property
    @torch.no_grad()
    def dtype(self):
        return self.xy1.dtype

    @torch.no_grad()
    def to_dtype(self, dtype: torch.dtype):
        self.xy1 = self.xy1.to(dtype)
        self.xy2 = self.xy2.to(dtype)

    @property
    @torch.no_grad()
    def num_image_pairs(self):
        # number of image pairs in this query
        return typing.cast(int, self.image_pair_mask.long().sum().item())

    @property
    @torch.no_grad()
    def num_point_pairs(self):
        # number of point pairs in this query
        return self.image_idx1.shape[0]


@dataclass
class Matches:
    """Container for matching results."""

    # bool, if xy and matrix are calibrated
    calibrated: bool
    # torch.Tensor, float, shape=(num_images, num_keypoints, 2), xy coordinates of keypoints in pixels
    xy_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_images, num_keypoints, 2), xy coordinates of keypoints. It is originally in pixels and will be converted to calibrated coordinates later.
    xy: torch.Tensor
    # torch.Tensor, long, shape=(num_image_pairs,), idx of image 1 in the image pair
    image_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_image_pairs,), idx of image 2 in the image pair
    image_idx2: torch.Tensor
    # torch.Tensor, float, shape=(num_image_pairs, 3, 3), fundamental or homography matrix. Note that the essential matrix is converted to the fundamental matrix when the two view geometry type is CALIBRATED.
    matrix: torch.Tensor
    # torch.Tensor, uint8, shape=(num_image_pairs,), two view geometry type (calibrated, uncalibrated, homography, etc.) represented as an integer (same convention as COLMAP)
    colmap_two_view_geometry_type: torch.Tensor
    # torch.Tensor, long, shape=(num_image_pairs,), starting idx of point pair for each image pair
    start_point_pair_idx: torch.Tensor
    # torch.Tensor, long, shape=(num_image_pairs,), number of point pair for each image pair
    num_point_pairs: torch.Tensor
    # torch.Tensor, long, shape=(num_point_pairs,), local idx of keypoints in image 1
    keypoint_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_point_pairs,), local idx of keypoints in image 2
    keypoint_idx2: torch.Tensor
    # torch.Tensor, long, shape=(num_point_pairs,), idx of the image pair for each point pair
    image_pair_idx: torch.Tensor

    @property
    @torch.no_grad()
    def device(self):
        return self.xy.device

    @property
    @torch.no_grad()
    def dtype(self):
        return self.xy.dtype

    @property
    @torch.no_grad()
    def num_images(self):
        return self.xy.shape[0]

    @property
    @torch.no_grad()
    def num_keypoints_per_image(self):
        return self.xy.shape[1]

    @property
    @torch.no_grad()
    def num_image_pairs(self):
        return self.image_idx1.shape[0]

    @property
    @torch.no_grad()
    def is_homography(self):
        # torch.Tensor, bool, shape=(num_image_pairs,)
        return (
            self.colmap_two_view_geometry_type
            == COLMAP_TWO_VIEW_GEOMETRY_TYPE.PLANAR_OR_PANORAMIC.value
        )

    @property
    def is_epipolar(self):
        # torch.Tensor, bool, shape=(num_image_pairs,)
        return ~self.is_homography

    @torch.no_grad()
    def query(
        self,
        image_pair_mask: Optional[torch.Tensor] = None,
        batch_size: int = 4096 * 1,
    ):
        """A generator for querying matching data for a set of image pairs. It guarantees that a batch of data contains either none or all of the matches of each image pair.
        Args:
            image_pair_mask (torch.Tensor): bool, shape=(num_image_pairs,), a mask of image pairs to query. If None, query all image pairs.
            batch_size (int): batch size (of image pairs) to yield
        Yields:
            matches_data: MatchesData container
        """
        # if image_pair_mask is None, query all image pairs
        if image_pair_mask is None:
            image_pair_mask = torch.ones(
                self.num_image_pairs, dtype=torch.bool, device=self.device
            )  # (num_image_pairs,)

        # get information
        assert image_pair_mask.dtype == torch.bool
        query_image_pair_idx = image_pair_mask.nonzero(as_tuple=False).squeeze(
            1
        )  # (num_query_image_pairs,)
        num_query_image_pairs = len(query_image_pair_idx)

        # loop over batches
        num_batches = (num_query_image_pairs + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_query_image_pairs)
            batch_query_image_pair_idx = query_image_pair_idx[start_idx:end_idx]  # (B,)
            batch_image_pair_mask = torch.zeros_like(
                image_pair_mask
            )  # (num_image_pairs,)
            batch_image_pair_mask[batch_query_image_pair_idx] = True
            del start_idx, end_idx, batch_query_image_pair_idx

            # get data
            batch_point_pair_mask = batch_image_pair_mask[
                self.image_pair_idx
            ]  # (num_point_pairs,)
            batch_image_pair_idx = self.image_pair_idx[
                batch_point_pair_mask
            ]  # (num_query_point_pairs,)
            batch_image_idx1 = self.image_idx1[
                batch_image_pair_idx
            ]  # (num_query_point_pairs,)
            batch_image_idx2 = self.image_idx2[
                batch_image_pair_idx
            ]  # (num_query_point_pairs,)
            batch_keypoint_idx1 = self.keypoint_idx1[
                batch_point_pair_mask
            ]  # (num_query_point_pairs,)
            batch_keypoint_idx2 = self.keypoint_idx2[
                batch_point_pair_mask
            ]  # (num_query_point_pairs,)
            batch_xy1_pixels = self.xy_pixels[
                batch_image_idx1, batch_keypoint_idx1
            ].clone()  # (num_query_point_pairs, 2)
            batch_xy2_pixels = self.xy_pixels[
                batch_image_idx2, batch_keypoint_idx2
            ].clone()  # (num_query_point_pairs, 2)
            batch_xy1 = self.xy[
                batch_image_idx1, batch_keypoint_idx1
            ].clone()  # (num_query_point_pairs, 2)
            batch_xy2 = self.xy[
                batch_image_idx2, batch_keypoint_idx2
            ].clone()  # (num_query_point_pairs, 2)

            # build MatchesData
            matches_data = MatchesData(
                image_pair_mask=batch_image_pair_mask,
                xy1_pixels=batch_xy1_pixels,
                xy2_pixels=batch_xy2_pixels,
                xy1=batch_xy1,
                xy2=batch_xy2,
                image_idx1=batch_image_idx1,
                image_idx2=batch_image_idx2,
                keypoint_idx1=batch_keypoint_idx1,
                keypoint_idx2=batch_keypoint_idx2,
                image_pair_idx=batch_image_pair_idx,
            )  # MatchesData

            # yield
            yield matches_data


@dataclass
class Cameras:
    """Container for camera parameters."""

    # torch.Tensor, long, shape=(num_images,), camera idx for each image
    camera_idx: torch.Tensor
    # torch.Tensor, float, shape=(num_cameras,), cx in pixels
    cx: torch.Tensor
    # torch.Tensor, float, shape=(num_cameras,), cy in pixels
    cy: torch.Tensor
    # torch.Tensor, float, shape=(num_cameras,), focal prior (e.g. read from EXIF) in pixels. Zero if no prior focal length.
    focal_prior: torch.Tensor
    # torch.Tensor, float, shape=(num_cameras,), focal length in pixels (set to focal_prior if not yet estimated)
    focal: torch.Tensor
    # torch.Tensor, float, shape=(num_cameras,), k1 radial distortion coefficient (zero if no distortion or not yet estimated)
    k1: torch.Tensor
    # bool, whether the focal length is estimated
    calibrated: bool

    @property
    def device(self):
        return self.focal.device

    @property
    def dtype(self):
        return self.focal.dtype

    @property
    def num_cameras(self):
        # int, number of cameras
        return self.focal.shape[0]

    @property
    def has_focal_prior(self):
        # torch.Tensor, bool, shape=(num_cameras,), whether each camera has a focal prior
        return self.focal_prior > 0.0

    @property
    def K(self):
        # torch.Tensor, float, shape=(num_cameras, 3, 3), intrinsics matrix
        K = torch.zeros(
            self.num_cameras, 3, 3, device=self.device, dtype=self.dtype
        )  # (num_cameras, 3, 3)
        K[:, 0, 2] = self.cx
        K[:, 1, 2] = self.cy
        K = K + torch.diag_embed(
            torch.stack([self.focal, self.focal, torch.ones_like(self.focal)], dim=-1)
        )  # (num_cameras, 3, 3)
        return K


@dataclass
class Tracks:
    """Container for tracks. Note that points in the same track are grouped together."""

    # torch.Tensor, long, shape=(num_tracks,), number of points in each track
    track_size: torch.Tensor
    # torch.Tensor, long, shape=(num_tracks,), first point idx in each track (assuming the track idx is non-decreasing as the point idx increases)
    track_start: torch.Tensor
    # torch.Tensor, long, shape=(num_points,), track idx for each 2D point (it should be non-decreasing as the point idx increases by construction)
    track_idx: torch.Tensor
    # torch.Tensor, float, shape=(num_points,), xy coordinates of keypoints in pixels
    xy_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_points,), calibrated xy coordinates of keypoints
    xy: torch.Tensor
    # torch.Tensor, long, shape=(num_points,), image idx for each 2D point
    image_idx: torch.Tensor
    # torch.Tensor, long, shape=(num_points,), local keypoint idx for each 2D point
    keypoint_idx: torch.Tensor

    @property
    @torch.no_grad()
    def device(self):
        return self.track_size.device

    @property
    @torch.no_grad()
    def num_tracks(self):
        return self.track_size.shape[0]

    @property
    @torch.no_grad()
    def num_points2d(self):
        return self.xy.shape[0]


@dataclass
class PointPairsData:
    """Container for storing data of queries of point pairs"""

    # torch.Tensor, long, shape=(num_query_point_pairs, 2), point pair idx
    point_pair_idx: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), xy coordinates of keypoints in image 1 in pixels
    xy1_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), xy coordinates of keypoints in image 2 in pixels
    xy2_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), calibrated xy coordinates of keypoints in image 1
    xy1: torch.Tensor
    # torch.Tensor, float, shape=(num_query_point_pairs, 2), calibrated xy coordinates of keypoints in image 2
    xy2: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), idx of image 1
    image_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), idx of image 2
    image_idx2: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), local keypoint idx of keypoint 1
    keypoint_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), local keypoint idx of keypoint 2
    keypoint_idx2: torch.Tensor
    # torch.Tensor, long, shape=(num_query_point_pairs, 2), track idx for each point pair
    track_idx: torch.Tensor

    @property
    @torch.no_grad()
    def device(self):
        return self.xy1.device

    @property
    @torch.no_grad()
    def dtype(self):
        return self.xy1.dtype

    @property
    @torch.no_grad()
    def num_point_pairs(self):
        return self.point_pair_idx.shape[0]


@dataclass
class PointPairs:
    """Container for point pairs."""

    # torch.Tensor, float, shape=(num_points, 2), xy coordinates in pixels
    xy_pixels: torch.Tensor
    # torch.Tensor, float, shape=(num_points, 2), calibrated xy coordinates
    xy: torch.Tensor
    # torch.Tensor, long, shape=(num_points,), image idx for the point
    image_idx: torch.Tensor
    # torch.Tensor, long, shape=(num_points,), local keypoint idx for the point
    keypoint_idx: torch.Tensor
    # torch.Tensor, long, shape=(num_points,), track idx for the point
    track_idx: torch.Tensor
    # torch.Tensor, long, shape=(num_point_pairs,), idx of first point in the pair
    point_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_point_pairs,), idx of second point in the pair
    point_idx2: torch.Tensor

    @property
    @torch.no_grad()
    def device(self):
        return self.xy.device

    @property
    @torch.no_grad()
    def dtype(self):
        return self.xy.dtype

    @property
    @torch.no_grad()
    def num_point_pairs(self):
        return self.point_idx1.shape[0]

    @torch.no_grad()
    def query(
        self,
        point_pair_mask: Optional[torch.Tensor] = None,
        randomize: bool = False,
        infinite: bool = False,
        drop_last: bool = False,
        batch_size: int = 4096 * 16,
    ):
        """A generator for querying point pair data in batches.
        Args:
            point_pair_mask (torch.Tensor): bool, shape=(num_point_pairs,), a mask of point pairs to query. If None, query all point pairs.
            randomize (bool): whether to randomize the order of point pairs
            infinite (bool): whether to loop infinitely
            drop_last (bool): whether to drop the last batch if it is not full
            batch_size (int): batch size of point pairs to yield
        Yields:
            point_pairs_data: PointPairsData container
        """
        # if point_pair_mask is None, query all point pairs
        if point_pair_mask is None:
            point_pair_mask = torch.ones(
                self.num_point_pairs, dtype=torch.bool, device=self.device
            )  # (num_point_pairs,)

        # get the point pair idx
        query_point_pair_idx = point_pair_mask.nonzero(as_tuple=False).squeeze(
            1
        )  # (num_point_pairs,)
        num_query_point_pairs = len(query_point_pair_idx)
        del point_pair_mask

        # set the number of epochs
        if infinite:
            num_epochs = 10000000000000
        else:
            num_epochs = 1

        for epoch_idx in range(num_epochs):
            # randomize the order of point pairs
            if randomize:
                query_point_pair_idx = query_point_pair_idx[
                    torch.randperm(num_query_point_pairs)
                ]  # (num_point_pairs,)

            # get the number of point pair batches
            num_batches = (num_query_point_pairs + batch_size - 1) // batch_size

            # loop
            for i in range(num_batches):
                # get the batch of point pair idx
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_query_point_pairs)
                point_pair_idx = query_point_pair_idx[start_idx:end_idx]  # (B,)
                del start_idx, end_idx

                # drop last batch if it is not full
                if drop_last and point_pair_idx.shape[0] < batch_size:
                    break

                # get data for the batch
                point_idx1 = self.point_idx1[point_pair_idx]  # (B,)
                point_idx2 = self.point_idx2[point_pair_idx]  # (B,)
                image_idx1 = self.image_idx[point_idx1]  # (B,)
                image_idx2 = self.image_idx[point_idx2]  # (B,)
                keypoint_idx1 = self.keypoint_idx[point_idx1]  # (B,)
                keypoint_idx2 = self.keypoint_idx[point_idx2]  # (B,)
                xy1_pixels = self.xy_pixels[point_idx1]  # (B, 2)
                xy2_pixels = self.xy_pixels[point_idx2]  # (B, 2)
                xy1 = self.xy[point_idx1]  # (B, 2)
                xy2 = self.xy[point_idx2]  # (B, 2)
                track_idx = self.track_idx[point_idx1]  # (B,)
                assert torch.all(track_idx == self.track_idx[point_idx2])
                del point_idx1, point_idx2

                # yield
                yield PointPairsData(
                    point_pair_idx=point_pair_idx,
                    xy1_pixels=xy1_pixels,
                    xy2_pixels=xy2_pixels,
                    xy1=xy1,
                    xy2=xy2,
                    image_idx1=image_idx1,
                    image_idx2=image_idx2,
                    keypoint_idx1=keypoint_idx1,
                    keypoint_idx2=keypoint_idx2,
                    track_idx=track_idx,
                )


@dataclass
class ImagePairs:
    """Container for image pairs."""

    # torch.Tensor, long, shape=(num_image_pairs,), idx of image 1
    image_idx1: torch.Tensor
    # torch.Tensor, long, shape=(num_image_pairs,), idx of image 2
    image_idx2: torch.Tensor
    # torch.Tensor, float, shape=(num_image_pairs, 3, 3), rotation matrix in the transformation 1->2
    rotation: torch.Tensor
    # torch.Tensor, float, shape=(num_image_pairs, 3), unit translation vector in the transformation 1->2
    translation: torch.Tensor
    # torch.Tensor, long, shape=(num_image_pairs,), number of inlier point pairs
    num_inliers: torch.Tensor

    @property
    def device(self):
        return self.rotation.device

    @property
    def dtype(self):
        return self.rotation.dtype

    @property
    def num_image_pairs(self):
        # int, number of image pairs
        return self.image_idx1.shape[0]


@dataclass
class Points3D:
    """Container for 3D points. It is mainly used for storing sparse reconstruction results and will not be used for further computation, so some data structures are inefficient for GPU computation but are more human-readable."""

    # torch.Tensor, float, shape=(num_points, 3), 3D coordinates of points
    xyz: torch.Tensor
    # torch.Tensor, uint8, shape=(num_points, 3), RGB color of points in [0, 255]
    rgb: torch.Tensor
    # torch.Tensor, float, shape=(num_points,), mean reprojection error in pixels
    error: torch.Tensor
    # List[List[int]], a list of length `num_points`, each element is a list of image idx in the corresponding track
    track_image_idx: List[List[int]]
    # List[List[int]], a list of length `num_points`, each element is a list of local keypoint idx in the corresponding image
    track_keypoint_idx: List[List[int]]

    @property
    @torch.no_grad()
    def device(self):
        return self.xyz.device

    @property
    @torch.no_grad()
    def num_points(self):
        return self.xyz.shape[0]
