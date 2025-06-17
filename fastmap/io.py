"""This file is modified from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py"""

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


from loguru import logger
import typing
from typing import Dict, Union
import os
import collections
import numpy as np
import struct
import torch

from fastmap.container import ColmapModel
from fastmap.container import Images as ImagesContainer
from fastmap.container import Cameras as CamerasContainer
from fastmap.container import Points3D as Points3DContainer


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_binary(cameras: Dict[int, Camera], path: str):
    """Write cameras to a binary file. See `read_cameras_binary` for the format.
    Args:
        cameras (Dict[int, Camera]): Dict of cameras (keys are camera_ids) to write.
        path (str): Path to the binary file.
    """
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as fid:
        num_cameras = len(cameras)
        write_next_bytes(fid, data=num_cameras, format_char_sequence="Q")
        for camera_id in sorted(cameras.keys()):
            assert camera_id > 0  # make sure camera_id is 1-based
            camera: Camera = cameras[camera_id]
            assert camera_id == camera.id  # just to be sure
            del camera_id
            assert camera.model == "SIMPLE_RADIAL"  # only support this model
            camera_properties = [
                camera.id,
                CAMERA_MODEL_NAMES[camera.model].model_id,
                camera.width,
                camera.height,
            ]
            write_next_bytes(fid, camera_properties, format_char_sequence="iiQQ")
            del camera_properties
            num_params = CAMERA_MODEL_NAMES[camera.model].num_params
            assert camera.params.shape == (num_params,)
            write_next_bytes(
                fid,
                data=camera.params.tolist(),
                format_char_sequence="d" * num_params,
            )


def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_binary(images: Dict[int, Image], path: str):
    """Write images (with poses) to a binary file. See `read_images_binary` for the format.
    Args:
        images (Dict[int, Image]): Dict of images (keys are image_ids) to write.
        path (str): Path to the binary file.
    """
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as fid:
        num_reg_images = len(images)
        write_next_bytes(fid, data=num_reg_images, format_char_sequence="Q")
        for image_id in sorted(images.keys()):
            assert image_id > 0  # make sure image_id is 1-based
            image: Image = images[image_id]
            assert image_id == image.id  # just to be sure
            del image_id
            binary_image_properties = (
                [image.id]
                + image.qvec.tolist()
                + image.tvec.tolist()
                + [image.camera_id]
            )
            assert len(binary_image_properties) == 9  # just to be sure
            write_next_bytes(
                fid, data=binary_image_properties, format_char_sequence="idddddddi"
            )
            del binary_image_properties
            binary_image_name = image.name.encode("utf-8")
            write_next_bytes(
                fid,
                data=binary_image_name + b"\x00",
                format_char_sequence=f"{len(binary_image_name)+1}s",
            )
            del binary_image_name
            # TODO: support writing 2D and 3D points
            assert image.xys is None  # do not write 2D points
            assert image.point3D_ids is None  # do not write 3D points
            num_points2D = 0
            write_next_bytes(fid, data=num_points2D, format_char_sequence="Q")


def read_points3D_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_binary(points3D: Dict[int, Point3D], path: str):
    """Write 3D points to a binary file. See `read_points3D_binary` for the format.
    Args:
        images (Dict[int, Image]): Dict of Point3D (keys are point3D_id) to write.
        path (str): Path to the binary file.
    """
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as fid:
        num_points = len(points3D)
        write_next_bytes(fid, data=num_points, format_char_sequence="Q")
        for point3D_id in sorted(points3D.keys()):
            point3D: Point3D = points3D[point3D_id]
            assert point3D_id == point3D.id  # just to be sure

            binary_point_line_properties = (
                [point3D_id]
                + point3D.xyz.tolist()
                + point3D.rgb.tolist()
                + [point3D.error.item()]
            )
            write_next_bytes(
                fid, data=binary_point_line_properties, format_char_sequence="QdddBBBd"
            )
            track_length = len(point3D.image_ids)
            write_next_bytes(fid, data=track_length, format_char_sequence="Q")
            track_elems = np.stack(
                [point3D.image_ids, point3D.point2D_idxs], axis=-1
            )  # (track_length, 2)
            track_elems = track_elems.flatten().tolist()
            write_next_bytes(
                fid,
                data=track_elems,
                format_char_sequence="ii" * track_length,
            )


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        logger.info("Detected model format: '" + ext + "'")
        return True

    return False


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_model(
    model_path: str, device: Union[torch.device, str] = "cpu", ext: str = ""
):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(model_path, ".bin"):
            ext = ".bin"
        elif detect_model_format(model_path, ".txt"):
            ext = ".txt"
        else:
            raise Exception(f"Model not found in {model_path}")

    # read the data
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(model_path, "cameras" + ext))
        images = read_images_text(os.path.join(model_path, "images" + ext))
        points3D = read_points3D_text(os.path.join(model_path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(model_path, "cameras" + ext))
        images = read_images_binary(os.path.join(model_path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(model_path, "points3D") + ext)

    # allocate the memory
    num_images = len(images)
    num_points3d = len(points3D)
    names = []
    mask = torch.zeros(num_images, dtype=torch.bool, device=device)  # (num_images,)
    rotation = torch.nan + torch.zeros(
        num_images, 3, 3, device=device, dtype=torch.float32
    )  # (num_images, 3, 3)
    translation = torch.nan + torch.zeros(
        num_images, 3, device=device, dtype=torch.float32
    )  # (num_images, 3)
    focal = torch.nan + torch.zeros(
        num_images, device=device, dtype=torch.float32
    )  # (num_images,)
    k1 = torch.nan + torch.zeros(
        num_images, device=device, dtype=torch.float32
    )  # (num_images,)
    xyz = torch.nan + torch.zeros(
        num_points3d, 3, device=device, dtype=torch.float32
    )  # (num_points3d, 3)
    rgb = torch.zeros(
        num_points3d, 3, device=device, dtype=torch.uint8
    )  # (num_points3d, 3)

    # fill the pose data
    for i, image_id in enumerate(sorted(images.keys())):
        names.append(images[image_id].name)
        mask[i] = True
        rotation[i] = torch.from_numpy(qvec2rotmat(images[image_id].qvec)).to(rotation)
        translation[i] = torch.from_numpy(images[image_id].tvec).to(translation)

    # fill the camera data
    for i, image_id in enumerate(sorted(images.keys())):
        if cameras[images[image_id].camera_id].model == "SIMPLE_PINHOLE":
            focal[i] = cameras[images[image_id].camera_id].params[0]
        elif cameras[images[image_id].camera_id].model == "PINHOLE":
            focal[i] = (
                cameras[images[image_id].camera_id].params[0]
                + cameras[images[image_id].camera_id].params[1]
            ) / 2.0
        elif cameras[images[image_id].camera_id].model == "SIMPLE_RADIAL":
            focal[i] = cameras[images[image_id].camera_id].params[0]
            k1[i] = cameras[images[image_id].camera_id].params[-1]
        else:
            raise ValueError(
                f"Unsupported camera model: {cameras[images[image_id].camera_id].model}"
            )

    # fill the 3D points data
    for i, point3D_id in enumerate(sorted(points3D.keys())):
        point3D = points3D[point3D_id]
        xyz[i] = torch.from_numpy(point3D.xyz).to(xyz)
        rgb[i] = torch.from_numpy(point3D.rgb).to(rgb)

    # make sure all the slots are filled
    assert not torch.any(torch.isnan(rotation))
    assert not torch.any(torch.isnan(translation))
    assert not torch.any(torch.isnan(focal))
    assert not torch.any(torch.isnan(xyz))

    # make sure the size of names is the same as num_images
    assert len(names) == num_images

    model = ColmapModel(
        names=names,
        mask=mask,
        rotation=rotation,
        translation=translation,
        focal=focal,
        k1=k1,
        points3d=xyz,
        rgb=rgb,
    )

    return model


def write_model(
    save_dir: str,
    images: ImagesContainer,
    cameras: CamerasContainer,
    points3d: Points3DContainer,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
):
    """Write results to binary files. It will have the same naming convention as the COLMAP output:
        1. cameras.bin
        2. images.bin
        3. points3D.bin
    Args:
        save_dir (str): Directory to save the results.
        images (ImagesContainer): ImagesContainer object.
        cameras (CamerasContainer): CamerasContainer object.
        points3d (Points3DContainer): Points3DContainer object.
        R_w2c (torch.Tensor): Rotation matrix from world to camera.
        t_w2c (torch.Tensor): Translation vector from world to camera.
    """
    # create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving the results to {save_dir}")

    # convert from CamerasContainer to Dict[int, Camera]
    cameras_dict = {}
    for i in range(cameras.num_cameras):
        # camera_id is 1-based
        camera_id = i + 1

        # always write in this model (even for pinhole cameras, where k1 is 0)
        model_name = "SIMPLE_RADIAL"

        # get width and height (make sure images with the same camera indeed have the same size)
        image_mask = cameras.camera_idx == i  # (num_images,)
        image_widths = images.widths[image_mask]  # (num_images_with_this_camera,)
        image_heights = images.heights[image_mask]  # (num_images_with_this_camera,)
        assert torch.all(image_widths == image_widths[0])
        assert torch.all(image_heights == image_heights[0])
        width = image_widths[0].item()
        width = typing.cast(int, width)
        height = image_heights[0].item()
        height = typing.cast(int, height)
        del image_mask, image_widths, image_heights

        # get the parameters
        focal = cameras.focal[i].item()
        cx = cameras.cx[i].item()
        cy = cameras.cy[i].item()
        k1 = cameras.k1[i].item()
        params = [focal, cx, cy, k1]

        # add to the dict
        cameras_dict[camera_id] = Camera(
            id=camera_id,
            model=model_name,
            width=width,
            height=height,
            params=np.array(params),
        )

    # convert from ImagesContainer to Dict[int, Image]
    images_dict = {}
    for i in range(images.num_images):
        # only save registered images
        if not images.mask[i]:
            continue

        # image_id is 1-based
        image_id = i + 1

        # get rotation and translation parameters in the form of qvec and tvec
        qvec = rotmat2qvec(R_w2c[i].cpu().numpy())  # array (4,)
        tvec = t_w2c[i].cpu().numpy()  # array (3,)
        assert qvec.shape == (4,)
        assert tvec.shape == (3,)

        # get camera_id (1-based)
        camera_id = cameras.camera_idx[i].item() + 1
        camera_id = typing.cast(int, camera_id)

        # get image name
        image_name = images.names[i]

        # add to the dict
        images_dict[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=None,  # TODO: correctly write the 2D points
            point3D_ids=None,  # TODO: correctly write the 3D points
        )

    # convert from Points3DContainer to Dict[int, Point3D]
    points3d_dict = {}
    for i in range(points3d.num_points):
        point3D_id = i + 1  # point3D_id is 1-based

        # get data
        xyz = points3d.xyz[i].cpu().numpy()  # array (3,)
        rgb = points3d.rgb[i].cpu().numpy()  # array (3,)
        error = points3d.error[i].cpu().numpy()  # array (,)

        # get image_ids and point2D_idxs
        image_ids = 1 + np.array(
            points3d.track_image_idx[i]
        )  # array, image_ids is 1-based
        point2D_idxs = np.array(
            points3d.track_keypoint_idx[i]
        )  # array, point2D_idxs is 0-based

        points3d_dict[point3D_id] = Point3D(
            id=point3D_id,
            xyz=xyz,
            rgb=rgb,
            error=error,
            image_ids=image_ids,
            point2D_idxs=point2D_idxs,
        )

    # write the cameras
    camera_path = os.path.join(save_dir, "cameras.bin")
    if os.path.exists(camera_path):
        raise FileExistsError(f"File {camera_path} already exists.")
    write_cameras_binary(cameras=cameras_dict, path=camera_path)
    logger.info(f"Cameras are written to {camera_path}")

    # write the images
    image_path = os.path.join(save_dir, "images.bin")
    if os.path.exists(image_path):
        raise FileExistsError(f"File {image_path} already exists.")
    write_images_binary(images=images_dict, path=image_path)
    logger.info(f"Images are written to {image_path}")

    # write the 3D points
    points3d_path = os.path.join(save_dir, "points3D.bin")
    if os.path.exists(points3d_path):
        raise FileExistsError(f"File {points3d_path} already exists.")
    write_points3D_binary(points3D=points3d_dict, path=points3d_path)
    logger.info(f"3D points are written to {points3d_path}")
