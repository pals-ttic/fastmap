"""This file is modified from https://github.com/colmap/colmap/blob/main/scripts/python/database.py"""

# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script is based on an original implementation by True Price.

from loguru import logger
from typing import List, Union, Tuple
import os
from enum import Enum
import sys
import sqlite3
import numpy as np
import torch

from fastmap.container import Matches, Cameras, Images


INVALID_IDX = int(2**31 - 1)

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


class TWO_VIEW_GEOMETRY_TYPE(Enum):
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


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path, load_to_memory=False):
        disk_conn = sqlite3.connect(database_path, factory=COLMAPDatabase)
        if load_to_memory:
            mem_conn = sqlite3.connect(":memory:", factory=COLMAPDatabase)
            disk_conn.backup(mem_conn)
            disk_conn.close()
            return mem_conn
        else:
            return disk_conn

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        prior_q=np.full(4, np.nan),
        prior_t=np.full(3, np.nan),
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )


def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = (
        0,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0)),
    )
    model2, width2, height2, params2 = (
        2,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0, 0.1)),
    )

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [
        image_ids_to_pair_id(*pair)
        for pair in (
            (image_id1, image_id2),
            (image_id2, image_id3),
            (image_id3, image_id4),
        )
    ]

    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)


@torch.no_grad()
def _build_cameras_container(
    image_names: List,
    colmap_cameras_focal: torch.Tensor,
    colmap_cameras_cx: torch.Tensor,
    colmap_cameras_cy: torch.Tensor,
    colmap_cameras_k1: torch.Tensor,
    is_prior: torch.Tensor,
    colmap_camera_ids: torch.Tensor,
) -> Cameras:
    """Builds a Cameras container from camera parameters and camera ids from a COLMAP database. Note that the detected number of cameras might be different from the number of cameras in the database because images from different subdirectories are considered to be from different cameras.
    Args:
        image_names: A list of image names. Note that the name might be a path containing multiple slashes (subdirectories). Images in different subdirectories should be considered from different cameras.
        colmap_cameras_focal: A tensor of shape (num_colmap_cameras,) containing the focal length of the cameras read from the COLMAP database.
        colmap_cameras_cx: A tensor of shape (num_colmap_cameras,) containing the cx of the cameras read from the COLMAP database.
        colmap_cameras_cy: A tensor of shape (num_colmap_cameras,) containing the cy of the cameras read from the COLMAP database.
        colmap_cameras_k1: A tensor of shape (num_colmap_cameras,) containing the k1 of the cameras read from the COLMAP database.
        is_prior: A tensor of shape (num_colmap_cameras,) containing a boolean indicating whether the camera is a prior.
        colmap_camera_ids: A tensor of shape (num_images,) containing the camera ids (starting from 1).
    Returns:
        A Cameras container. Note that the number of distinct cameras and the camera idx are possibly different.
    """
    # get information
    device = colmap_cameras_focal.device
    dtype = colmap_cameras_focal.dtype
    num_images = colmap_camera_ids.shape[0]
    assert colmap_camera_ids.min() == 1  # make sure colmap camera ids are 1-based

    # get a unique id for each directory
    unique_dirs = set([os.path.dirname(name) for name in image_names])
    unique_dir_to_id = {dir: idx for idx, dir in enumerate(unique_dirs)}
    dir_ids = torch.tensor(
        [unique_dir_to_id[os.path.dirname(name)] for name in image_names],
        device=device,
        dtype=torch.long,
    )  # (num_images,)
    del unique_dirs, unique_dir_to_id

    # get unique dir_id, focal, cx, cy, is_prior combinations
    unique_items, camera_idx = torch.unique(
        torch.stack(
            [
                colmap_cameras_focal[colmap_camera_ids - 1],
                colmap_cameras_cx[colmap_camera_ids - 1],
                colmap_cameras_cy[colmap_camera_ids - 1],
                colmap_cameras_k1[colmap_camera_ids - 1],
                dir_ids.to(device=device, dtype=dtype),
                is_prior[colmap_camera_ids - 1].to(device=device, dtype=dtype),
            ],
            dim=-1,
        ),
        return_inverse=True,
        dim=0,
    )  # (num_cameras, 5), (num_images,) note that here camera_idx is 0-based
    focal_prior = unique_items[:, 0]  # (num_cameras,)
    focal = focal_prior.clone()  # (num_cameras,)
    cx = unique_items[:, 1]  # (num_cameras,)
    cy = unique_items[:, 2]  # (num_cameras,)
    k1 = unique_items[:, 3]  # (num_cameras,)
    num_cameras = unique_items.shape[0]
    del dir_ids, unique_items, is_prior

    # build Cameras container
    assert camera_idx.shape == (num_images,)
    assert focal.shape == (num_cameras,)
    assert focal_prior.shape == (num_cameras,)
    assert cx.shape == (num_cameras,)
    assert cy.shape == (num_cameras,)
    assert k1.shape == (num_cameras,)
    cameras = Cameras(
        camera_idx=camera_idx,
        focal=focal,
        focal_prior=focal_prior,
        calibrated=False,
        cx=cx,
        cy=cy,
        k1=k1,
    )

    # return
    return cameras


@torch.no_grad()
def read_database(
    database_path: str, device: Union[str, torch.device], min_num_inliers: int = 8
) -> Tuple[Matches, Cameras, Images]:
    """Reads database
    Args:
        database_path: The path to the database.
        device: The device to use.
        min_num_inliers: The minimum number of inliers for an image pair to be considered.
    Returns:
        matches: A Matches container.
        cameras: A Cameras container.
        images: An Images container.
    """
    assert os.path.exists(database_path)

    db = COLMAPDatabase.connect(database_path, load_to_memory=True)

    # read image paths and camera ids
    image_names = []
    camera_ids = []
    for image_idx, (image_id, image_name, camera_id) in enumerate(
        db.execute("SELECT image_id, name, camera_id FROM images")
    ):
        # image_idx is 0 based but image_id and camera_id are 1-based
        assert image_id == image_idx + 1
        image_names.append(image_name)
        camera_ids.append(camera_id)
    camera_ids = torch.tensor(
        camera_ids, device=device, dtype=torch.long
    )  # (num_images,)

    # get number of images
    num_images = len(image_names)

    # read camera info
    colmap_cameras = []
    for (
        camera_id,
        camera_model_type,
        image_width,
        image_height,
        camera_params,
        camera_is_prior,
    ) in db.execute("SELECT * FROM cameras"):
        camera_params = blob_to_array(camera_params, np.float64)
        camera_params = torch.from_numpy(camera_params).float().to(device)
        assert camera_is_prior in [0, 1]
        camera_is_prior = bool(camera_is_prior)
        colmap_cameras.append(
            {
                "id": camera_id,
                "model_type": camera_model_type,
                "image_width": image_width,
                "image_height": image_height,
                "camera_params": camera_params,
                "is_prior": camera_is_prior,
            }
        )

    # sort cameras by id
    colmap_cameras = sorted(colmap_cameras, key=lambda x: x["id"])
    assert torch.all(
        torch.tensor([cam["id"] for cam in colmap_cameras])
        == torch.arange(1, len(colmap_cameras) + 1)
    )  # make sure camera ids are 1-based

    # build Images container
    images = Images(
        names=image_names,
        widths=torch.zeros(num_images, device=device, dtype=torch.long),
        heights=torch.zeros(num_images, device=device, dtype=torch.long),
        mask=torch.zeros(num_images, device=device, dtype=torch.bool),
    )
    for image_idx, camera_id in enumerate(camera_ids):
        images.widths[image_idx] = colmap_cameras[camera_id - 1]["image_width"]
        images.heights[image_idx] = colmap_cameras[camera_id - 1]["image_height"]
    del image_names

    # define supported camera model types (the values are the same as in colmap)
    class CAMERA_MODEL_TYPE(Enum):
        SIMPLE_PINHOLE = 0
        PINHOLE = 1
        SIMPLE_RADIAL = 2

    # separate the camera parameters
    num_colmap_cameras = len(colmap_cameras)
    is_prior = torch.tensor(
        [cam["is_prior"] for cam in colmap_cameras], dtype=torch.bool, device=device
    )  # (num_colmap_cameras,)
    colmap_cameras_focal = torch.zeros(
        num_colmap_cameras, device=device, dtype=torch.float32
    )  # (num_colmap_cameras,)
    colmap_cameras_cx = torch.zeros(
        num_colmap_cameras, device=device, dtype=torch.float32
    )  # (num_colmap_cameras,)
    colmap_cameras_cy = torch.zeros(
        num_colmap_cameras, device=device, dtype=torch.float32
    )  # (num_colmap_cameras,)
    colmap_cameras_k1 = torch.zeros(
        num_colmap_cameras, device=device, dtype=torch.float32
    )  # (num_colmap_cameras,)
    for i in range(num_colmap_cameras):
        cam = colmap_cameras[i]
        if cam["model_type"] == CAMERA_MODEL_TYPE.SIMPLE_PINHOLE.value:
            colmap_cameras_focal[i] = cam["camera_params"][0]
            colmap_cameras_cx[i] = cam["camera_params"][1]
            colmap_cameras_cy[i] = cam["camera_params"][2]
            colmap_cameras_k1[i] = 0.0
        elif cam["model_type"] == CAMERA_MODEL_TYPE.PINHOLE.value:
            colmap_cameras_focal[i] = (
                cam["camera_params"][0] + cam["camera_params"][1]
            ) / 2.0
            colmap_cameras_cx[i] = cam["camera_params"][2]
            colmap_cameras_cy[i] = cam["camera_params"][3]
            colmap_cameras_k1[i] = 0.0
        elif cam["model_type"] == CAMERA_MODEL_TYPE.SIMPLE_RADIAL.value:
            colmap_cameras_focal[i] = cam["camera_params"][0]
            colmap_cameras_cx[i] = cam["camera_params"][1]
            colmap_cameras_cy[i] = cam["camera_params"][2]
            colmap_cameras_k1[i] = cam["camera_params"][3]
        else:
            raise ValueError(
                f"Currently only support SIMPLE_PINHOLE (1) and SIMPLE_RADIAL (2) camera models, but got {cam['model_type']}."
            )
    del colmap_cameras

    # build Cameras container
    cameras: Cameras = _build_cameras_container(
        image_names=images.names,
        colmap_cameras_focal=colmap_cameras_focal,
        colmap_cameras_cx=colmap_cameras_cx,
        colmap_cameras_cy=colmap_cameras_cy,
        colmap_cameras_k1=colmap_cameras_k1,
        is_prior=is_prior,
        colmap_camera_ids=camera_ids,
    )
    del is_prior, camera_ids
    logger.info(f"Found {cameras.num_cameras} unique cameras in the database.")

    # intialize result dict for matches
    res = {}

    # read keypoints
    num_keypoints = dict(
        (image_id, n)
        for image_id, n in db.execute("SELECT image_id, rows FROM keypoints")
    )
    assert len(num_keypoints) == images.num_images
    max_num_keypoints = max(num_keypoints.values())
    keypoints = torch.nan + torch.zeros(
        images.num_images, max_num_keypoints, 2, device=device, dtype=torch.float32
    )  # (num_images, max_num_keypoints, 2)
    for image_id, data in db.execute("SELECT image_id, data FROM keypoints"):
        if not data:
            continue
        data = blob_to_array(
            data, np.float32, (num_keypoints[image_id], -1)
        )  # numpy array of shape (num_keypoints, 6)
        data = torch.from_numpy(data).to(device).float()  # (num_keypoints, 6)
        data = data[:, :2]  # (num_keypoints, 2) only preserve the xy coordinates
        # image_id is 1-based
        keypoints[image_id - 1, : num_keypoints[image_id], :] = data
        images.mask[image_id - 1] = True
    res["keypoints"] = keypoints
    del keypoints, num_keypoints, max_num_keypoints

    # read matches
    res.update(
        {
            "image_idx1": [],
            "image_idx2": [],
            "matrix": [],
            "two_view_geometry_type": [],
            "keypoint_idx1": [],
            "keypoint_idx2": [],
            "start_point_pair_idx": [],
            "num_point_pairs": [],
            "image_pair_idx": [],
        }
    )
    current_start_point_pair_idx = 0
    current_image_pair_idx = 0
    # See the above and src/colmap/scene/two_view_geometry.h for more information.
    homography_count = 0
    fundamental_count = 0
    essential_count = 0
    for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in db.execute(
        "SELECT * FROM two_view_geometries"
    ):
        if rows < min_num_inliers:
            continue
        config = TWO_VIEW_GEOMETRY_TYPE(config)
        if config == TWO_VIEW_GEOMETRY_TYPE.CALIBRATED:
            inlier_matches = blob_to_array(data, np.uint32, (rows, cols))
            matrix = blob_to_array(E, np.float64, (3, 3))
            essential_count += 1
        elif config == TWO_VIEW_GEOMETRY_TYPE.UNCALIBRATED:
            inlier_matches = blob_to_array(data, np.uint32, (rows, cols))
            matrix = blob_to_array(F, np.float64, (3, 3))
            fundamental_count += 1
        elif config == TWO_VIEW_GEOMETRY_TYPE.PLANAR_OR_PANORAMIC:
            inlier_matches = blob_to_array(data, np.uint32, (rows, cols))
            matrix = blob_to_array(H, np.float64, (3, 3))
            homography_count += 1
        elif config == TWO_VIEW_GEOMETRY_TYPE.UNDEFINED:
            continue
        elif config == TWO_VIEW_GEOMETRY_TYPE.WATERMARK:
            continue
        elif config == TWO_VIEW_GEOMETRY_TYPE.DEGENERATE:
            continue
        else:
            raise Exception(
                f"Unsupported two view geometry type in COLMAP database: {config}"
            )

        image_ids = pair_id_to_image_ids(pair_id)
        res["image_idx1"].append(image_ids[0] - 1)  # make it 0-based
        res["image_idx2"].append(image_ids[1] - 1)  # make it 0-based
        res["keypoint_idx1"].append(torch.from_numpy(inlier_matches).long()[..., 0])
        res["keypoint_idx2"].append(torch.from_numpy(inlier_matches).long()[..., 1])
        num_point_pairs = inlier_matches.shape[0]
        res["num_point_pairs"].append(num_point_pairs)
        res["start_point_pair_idx"].append(current_start_point_pair_idx)
        current_start_point_pair_idx += num_point_pairs
        res["image_pair_idx"].append(
            torch.tensor([current_image_pair_idx] * num_point_pairs, dtype=torch.long)
        )
        current_image_pair_idx += 1
        res["two_view_geometry_type"].append(config.value)
        res["matrix"].append(torch.from_numpy(matrix).float())

    logger.info(f"homography_count {homography_count}")
    logger.info(f"fundamental_count {fundamental_count}")
    logger.info(f"essential_count {essential_count}")

    db.close()

    # convert to torch and concat
    res["image_idx1"] = torch.tensor(res["image_idx1"], dtype=torch.long)
    res["image_idx2"] = torch.tensor(res["image_idx2"], dtype=torch.long)
    res["keypoint_idx1"] = torch.cat(res["keypoint_idx1"], dim=0)
    res["keypoint_idx2"] = torch.cat(res["keypoint_idx2"], dim=0)
    res["start_point_pair_idx"] = torch.tensor(
        res["start_point_pair_idx"], dtype=torch.long
    )
    res["num_point_pairs"] = torch.tensor(res["num_point_pairs"], dtype=torch.long)
    res["image_pair_idx"] = torch.cat(res["image_pair_idx"], dim=0)
    res["matrix"] = torch.stack(res["matrix"], dim=0)
    res["two_view_geometry_type"] = torch.tensor(
        res["two_view_geometry_type"], dtype=torch.uint8
    )

    # build matches container. As of now it still might contain some essential matrices
    matches = Matches(
        calibrated=False,
        xy_pixels=res["keypoints"].to(device),
        xy=res["keypoints"].to(device),
        image_idx1=res["image_idx1"].to(device),
        image_idx2=res["image_idx2"].to(device),
        matrix=res["matrix"].to(device),
        colmap_two_view_geometry_type=res["two_view_geometry_type"].to(device),
        start_point_pair_idx=res["start_point_pair_idx"].to(device),
        num_point_pairs=res["num_point_pairs"].to(device),
        keypoint_idx1=res["keypoint_idx1"].to(device),
        keypoint_idx2=res["keypoint_idx2"].to(device),
        image_pair_idx=res["image_pair_idx"].to(device),
    )
    del res

    # convert essential matrix to fundamental matrix (it seems this fundamental matrix does not yield the lowest error, but it doesn't matter becauuse we will re-estimate it during undistortion)
    essential_mask = (
        matches.colmap_two_view_geometry_type == TWO_VIEW_GEOMETRY_TYPE.CALIBRATED.value
    )  # (num_image_pairs,)
    if torch.any(essential_mask):
        essential = matches.matrix[essential_mask]  # (num_essential, 3, 3)
        camera_idx1 = cameras.camera_idx[
            matches.image_idx1[essential_mask]
        ]  # (num_essential,)
        camera_idx2 = cameras.camera_idx[
            matches.image_idx2[essential_mask]
        ]  # (num_essential,)
        K1 = cameras.K[camera_idx1]  # (num_essential, 3, 3)
        K2 = cameras.K[camera_idx2]  # (num_essential, 3, 3)
        del camera_idx1, camera_idx2
        fundamental = (
            K2.inverse().transpose(-1, -2) @ essential @ K1.inverse()
        )  # (num_essential, 3, 3)
        matches.matrix[essential_mask] = fundamental
    del essential_mask

    # return matches, image_paths, image_width, image_height
    return matches, cameras, images
