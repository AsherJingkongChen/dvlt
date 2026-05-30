# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test the camera classes. Adapted from nerfstudio.
"""

import dataclasses
from itertools import product

import torch

from dvlt.struct.cameras import Cameras, CameraType
from dvlt.struct.rays import RayBundle


BATCH_SIZE = 2
H_W = 800
FX_Y = 10.0
CX_Y = H_W / 2.0
C2W_FLAT = torch.eye(4)[:3, :]
CAMERA_TO_WORLDS = [
    C2W_FLAT,
    torch.stack([C2W_FLAT] * BATCH_SIZE),
    torch.stack([torch.stack([C2W_FLAT] * BATCH_SIZE)] * BATCH_SIZE),
]
FX_YS = [
    FX_Y,
    torch.tensor([1]).float() * FX_Y,
    torch.ones(BATCH_SIZE, 1) * FX_Y,
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)) * FX_Y,
]
H_WS = [
    None,
    H_W,
    torch.tensor([1]) * H_W,
    torch.ones(BATCH_SIZE, 1).int() * H_W,
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)).int() * H_W,
]
CX_YS = [
    CX_Y,
    torch.tensor([1]).float() * CX_Y,
    torch.ones(BATCH_SIZE, 1) * CX_Y,
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)) * CX_Y,
]
DISTORTION_PARAMS = [None, torch.zeros(6), torch.zeros((BATCH_SIZE, 6)), torch.zeros((BATCH_SIZE, BATCH_SIZE, 6))]
camera_types = [
    1,
    torch.tensor([1]),
    torch.ones(BATCH_SIZE, 1).int(),
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)).int(),
]
C = Cameras(
    CAMERA_TO_WORLDS[1],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C0 = Cameras(
    CAMERA_TO_WORLDS[0],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C1 = Cameras(
    CAMERA_TO_WORLDS[1],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C2 = Cameras(
    CAMERA_TO_WORLDS[2],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C2_DIST = Cameras(
    CAMERA_TO_WORLDS[2],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[1],
    camera_types[0],
)


def test_pinhole_camera():
    """Test that the pinhole camera model works."""
    c2w = torch.eye(4)[None, :3, :]
    pinhole_camera = Cameras(cx=400.0, cy=400.0, fx=10.0, fy=10.0, camera_to_worlds=c2w)
    camera_ray_bundle = pinhole_camera.generate_rays(camera_indices=0)
    assert isinstance(camera_ray_bundle, RayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))

    # Test generate rays on 1D input
    num_rays = 10
    coords = torch.ones(num_rays, 2)
    pinhole_camera.generate_rays(camera_indices=0, coords=coords)

    # Additional checks for ray directions in OpenCV coordinates (+x right, +y down, +z forward)
    directions = camera_ray_bundle.directions
    # Compute expected direction for the central pixel (row = col = CX_Y)
    # Pixel centres are offset by +0.5 in the implementation
    row_col = int(CX_Y)
    offset = 0.5
    dx = ((row_col + offset) - CX_Y) / 10.0  # (x-cx)/fx
    dy = ((row_col + offset) - CX_Y) / 10.0  # (y-cy)/fy
    expected = torch.tensor([dx, dy, 1.0])
    expected = expected / expected.norm()
    assert torch.allclose(directions[row_col, row_col], expected, atol=1e-4, rtol=1e-3)

    # Top-left pixel should have components: x negative (left), y negative (up), z positive (forward)
    top_left_dir = directions[0, 0]
    assert top_left_dir[0] < 0  # left
    assert top_left_dir[1] < 0  # up (negative y)
    assert top_left_dir[2] > 0  # forward


def test_orthophoto_camera():
    """Test that the orthographic camera model works."""
    c2w = torch.eye(4)[None, :3, :]
    # apply R and T.
    R = torch.Tensor(
        [
            [0.5, -0.14644661, 0.85355339],
            [0.5, 0.85355339, -0.14644661],
            [-0.70710678, 0.5, 0.5],
        ]
    ).unsqueeze(0)
    T = torch.Tensor([[0.5, 0, -0.5]])
    c2w[..., :3, :3] = R
    c2w[..., :3, 3] = T

    ortho_cam = Cameras(cx=1.5, cy=1.5, fx=1.0, fy=1.0, camera_to_worlds=c2w, camera_type=CameraType.ORTHOPHOTO)
    ortho_rays = ortho_cam.generate_rays(camera_indices=0)
    # campare with `PERSPECTIVE` to validate `ORTHOPHOTO`.
    pinhole_cam = Cameras(cx=1.5, cy=1.5, fx=1.0, fy=1.0, camera_to_worlds=c2w, camera_type=CameraType.PERSPECTIVE)
    pinhole_rays = pinhole_cam.generate_rays(camera_indices=0)

    assert ortho_rays.shape == pinhole_rays.shape
    # `ortho_rays.directions` should equal to the center ray of `pinhole_rays.directions`.
    assert torch.allclose(
        ortho_rays.directions, pinhole_rays.directions[1, 1].broadcast_to(ortho_rays.directions.shape)
    )
    # `ortho_rays.origins` should be grid points with a mean value of `pinhole_rays.origins`.
    assert torch.allclose(ortho_rays.origins.mean(dim=(0, 1)), pinhole_rays.origins[1, 1])

    # Additional robustness checks ---------------------
    # 1. All direction vectors for an orthophoto image should be identical (parallel rays).
    dir0 = ortho_rays.directions[0, 0]
    assert torch.allclose(ortho_rays.directions, dir0.expand_as(ortho_rays.directions))

    # 2. Origin grid lies in a plane orthogonal to the shared direction.
    diff_x = ortho_rays.origins[:, 1:, :] - ortho_rays.origins[:, :-1, :]
    diff_y = ortho_rays.origins[1:, :, :] - ortho_rays.origins[:-1, :, :]
    # Dot-product of these in-plane vectors with direction should be ~0.
    assert torch.allclose((diff_x * dir0).sum(-1), torch.zeros_like(diff_x[..., 0]), atol=1e-5)
    assert torch.allclose((diff_y * dir0).sum(-1), torch.zeros_like(diff_y[..., 0]), atol=1e-5)


def test_multi_camera_type():
    """Test that the orthographic camera model works."""
    # here we test two different camera types.
    num_cams = [2]
    c2w = torch.eye(4)[None, :3, :].broadcast_to(*num_cams, 3, 4)
    cx = torch.Tensor([20]).broadcast_to(*num_cams, 1)
    cy = torch.Tensor([10]).broadcast_to(*num_cams, 1)
    fx = torch.Tensor([10]).broadcast_to(*num_cams, 1)
    fy = torch.Tensor([10]).broadcast_to(*num_cams, 1)
    h = torch.Tensor([40]).long().broadcast_to(*num_cams, 1)
    w = torch.Tensor([20]).long().broadcast_to(*num_cams, 1)
    camera_type = [CameraType.PERSPECTIVE, CameraType.ORTHOPHOTO]
    multitype_cameras = Cameras(c2w, fx, fy, cx, cy, w, h, camera_type=camera_type)

    # test `generate_rays`, 1 cam.
    ray0 = multitype_cameras.generate_rays(camera_indices=0)
    assert ray0.shape == torch.Size([40, 20])

    # test `generate_rays`, multiple cams.
    num_rays = [30, 30]
    camera_indices = torch.randint(0, 2, [*num_rays, len(num_cams)])  # (*num_rays, num_cameras_batch_dims)
    ray1 = multitype_cameras.generate_rays(camera_indices=camera_indices)
    assert ray1.shape == torch.Size([40, 20, *num_rays])

    # test `_generate_rays_from_coords`, 1 cam.
    coords = torch.randint(0, 10, [*num_rays, 2]).float()  # (*num_rays 2)
    ray2 = multitype_cameras.generate_rays(camera_indices=0, coords=coords)
    assert ray2.shape == torch.Size([*num_rays])

    # test `_generate_rays_from_coords`, multiple cam.
    ray3 = multitype_cameras.generate_rays(camera_indices=camera_indices, coords=coords)
    assert ray3.shape == torch.Size([*num_rays])

    # Extra: verify that the orthophoto camera in the batch has +z directions.
    ortho_dirs = multitype_cameras.generate_rays(camera_indices=1).directions
    assert torch.allclose(ortho_dirs, torch.tensor([0.0, 0.0, 1.0]).expand_as(ortho_dirs))


def test_equirectangular_camera():
    """Test that the equirectangular camera model works."""
    height = 100  # width is twice the height
    c2w = torch.eye(4)[None, :3, :]
    equirectangular_camera = Cameras(
        cx=float(height),
        cy=0.5 * float(height),
        fx=float(height),
        fy=float(height),
        camera_to_worlds=c2w,
        camera_type=CameraType.EQUIRECTANGULAR,
    )
    camera_ray_bundle = equirectangular_camera.generate_rays(camera_indices=0)
    assert isinstance(camera_ray_bundle, RayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))

    # Check that the directions are mostly correct in local camera coordinates
    # using the OpenCV convention: +x right, +y down, +z forward.
    directions = camera_ray_bundle.directions
    threshold = 0.9

    # Basis vectors in OpenCV coordinates
    x = torch.tensor([1.0, 0.0, 0.0])  # right
    y_down = torch.tensor([0.0, 1.0, 0.0])  # down
    y_up = -y_down  # up
    z = torch.tensor([0.0, 0.0, 1.0])  # forward
    z_back = -z  # backward

    # top pixels point up (negative y direction)
    assert directions[0, 0] @ y_up > threshold
    assert directions[0, height] @ y_up > threshold
    assert directions[0, -1] @ y_up > threshold

    # middle pixels: check the four cardinal directions and front/back
    # leftmost pixel points backward
    assert directions[height // 2, 0] @ z_back > threshold
    # quarter width pixel points left
    assert directions[height // 2, height // 2] @ -x > threshold
    # centre pixel points forward
    assert directions[height // 2, height] @ z > threshold
    # three-quarter width pixel points right
    assert directions[height // 2, 3 * height // 2] @ x > threshold
    # rightmost pixel points backward again
    assert directions[height // 2, -1] @ z_back > threshold

    # bottom pixels point down (positive y direction)
    assert directions[-1, 0] @ y_down > threshold
    assert directions[-1, height] @ y_down > threshold
    assert directions[-1, -1] @ y_down > threshold


def test_camera_as_tensordataclass():
    """Test that the camera class move to Tensordataclass works."""
    _ = C2_DIST[torch.tensor([0]), torch.tensor([0])]

    assert C0.shape == ()
    assert C1[...].shape == torch.Size([BATCH_SIZE])

    assert _check_cam_shapes(C0, ())
    assert _check_cam_shapes(C1, (2,))
    assert _check_cam_shapes(C2_DIST, (2, 2))

    assert C0.generate_rays(0).shape == (H_W, H_W)
    assert C0.generate_rays(0, coords=torch.ones(10, 2)).shape == (10,)
    C1.generate_rays(0)
    C1.generate_rays(torch.tensor([0, 1]).unsqueeze(-1))

    # Make sure rays generated are same when distortion params are identity (all zeros) and None
    assert C2.shape == C2_DIST.shape

    c2_rays = C2.generate_rays(torch.tensor([0, 0]))
    c_dist_rays = C2_DIST.generate_rays(torch.tensor([0, 0]))
    assert _check_dataclass_allclose(c2_rays, c_dist_rays)
    assert c2_rays.shape == (H_W, H_W)
    assert c_dist_rays.shape == (H_W, H_W)

    camera_indices = torch.tensor([[0, 0]])
    assert C2.generate_rays(camera_indices).shape == (H_W, H_W, 1)

    for args in product(
        CAMERA_TO_WORLDS,
        FX_YS,
        FX_YS[:1],
        CX_YS,
        CX_YS[:1],
        H_WS,
        H_WS[:1],
        DISTORTION_PARAMS[:-1],
        camera_types[:-1],
    ):
        _ = Cameras(*args)


def check_generate_rays_shape():
    """Checking the output shapes from Cameras.generate_rays"""
    coord = torch.tensor([1, 1])
    combos = [
        (0, None, torch.Size((800, 800))),  # First camera, all pixels
        (
            0,
            coord,
            torch.Size(()),
        ),  # First camera, selected pixels in coords, output zero dimensional since no extra batch dim in coord
        (
            torch.zeros(1, 1),
            coord.broadcast_to(1, 2),
            torch.Size((1,)),
        ),  # [0]th camera and selected coords, output is [1] dimensional
        (
            0,
            coord.broadcast_to(1, 2),
            torch.Size((1,)),
        ),  # First camera and selected coords, output is [1] dimensional since one extra batch dim in coords
        (
            torch.zeros(1),
            None,
            torch.Size((800, 800)),
        ),  # [0]th camera, all pixels, output is [HxW] dimensional since no extra output batch dim
        (
            torch.zeros(5, 1),
            coord.broadcast_to(5, 2),
            torch.Size((5,)),
        ),  # [0]th camera and selected coords, output is [5] dimensional since extra batch dim is provided in inputs
        (0, coord.broadcast_to(5, 2), torch.Size((5,))),  # First camera and selected coords, output is [5] dimensional
        (
            torch.zeros(5, 1),
            None,
            torch.Size((800, 800, 5)),
        ),  # [0]th camera and all pixels, HxWx5 since coords is none and our extra batch dim in our inputs is [5]
        (
            torch.zeros(11, 5, 1),
            coord.broadcast_to(11, 5, 2),
            torch.Size((11, 5)),
        ),  # [0]th camera and selected coords since inputs have 2 extra batch dims of [11,5]
        (
            0,
            coord.broadcast_to(11, 5, 2),
            torch.Size((11, 5)),
        ),  # [0]th camera and selected coords since coords have 2 extra batch dims of [11, 5]
        (
            torch.zeros(11, 5, 1),
            None,
            torch.Size((800, 800, 11, 5)),
        ),  # [0]th camera and all pixels since coords is none but we still have 2 extra batch dims of [11, 5] in coords
    ]
    for camera_indices, coords, output_size in combos:
        shape = C0.generate_rays(camera_indices=camera_indices, coords=coords).shape
        assert shape == output_size

    assert C1.shape == (2,)
    for camera_indices, coords, output_size in combos:
        shape = C1.generate_rays(camera_indices=camera_indices, coords=coords).shape
        assert shape == output_size

    # camera_indices can't be an int anymore since our cameras object have 2 batch dimensions
    # camera_indices last dim needs to be (..., 2) since len(cameras.shape) == 2
    combos = [
        (torch.zeros(2), coord, ()),
        (torch.zeros(1, 2), coord.broadcast_to(1, 2), (1,)),
        (torch.zeros(2), None, (800, 800)),
        (torch.zeros(5, 2), coord.broadcast_to(5, 2), (5,)),
        (torch.zeros(5, 2), None, (800, 800, 5)),
        (torch.zeros(11, 5, 2), coord.broadcast_to(11, 5, 2), (11, 5)),
        (torch.zeros(11, 5, 2), None, (800, 800, 11, 5)),
    ]
    for camera_indices, coords, output_size in combos:
        shape = C2.generate_rays(camera_indices=camera_indices, coords=coords).shape
        assert shape == output_size


def _check_dataclass_allclose(ipt, other):
    for field in dataclasses.fields(ipt):
        if getattr(ipt, field.name) is not None:
            if isinstance(getattr(ipt, field.name), dict):
                ipt_dict = getattr(ipt, field.name)
                other_dict = getattr(other, field.name)
                for k, v in ipt_dict.items():
                    assert k in other_dict
                    assert torch.allclose(v, other_dict[k])
            else:
                assert torch.allclose(getattr(ipt, field.name), getattr(other, field.name))
    return True


def _check_cam_shapes(cam: Cameras, _batch_size):
    if _batch_size:
        assert len(cam) == _batch_size[0]
    assert cam.shape == (*_batch_size,)
    assert cam.camera_to_worlds.shape == (*_batch_size, 3, 4)
    assert cam.fx.shape == (*_batch_size, 1)
    assert cam.fy.shape == (*_batch_size, 1)
    assert cam.cx.shape == (*_batch_size, 1)
    assert cam.cy.shape == (*_batch_size, 1)
    assert cam.height.shape == (*_batch_size, 1)
    assert cam.width.shape == (*_batch_size, 1)
    assert cam.distortion_params is None or cam.distortion_params.shape == (*_batch_size, 6)
    assert cam.camera_type.shape == (*_batch_size, 1)
    return True


def test_json_export_import(tmp_path):
    num_images = 5
    poses = torch.rand((num_images, 3, 4))
    dist_params = torch.rand(num_images, 6)
    times = torch.rand(num_images, 1)
    metadata = {
        "something1": torch.rand((num_images, 1)),
        "something4": torch.rand((num_images, 4)),
        "somethingbool": torch.rand((num_images, 1)).bool(),
        "somethingint": torch.rand((num_images, 1)).long(),
    }
    fx, fy, cx, cy = 300.0, 300.0, 64.0, 64.0
    width, height = 128, 128
    cameras = Cameras(
        camera_to_worlds=poses,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        distortion_params=dist_params,
        camera_type=CameraType.PERSPECTIVE,
        times=times,
        metadata=metadata,
    )

    cameras.to_json(tmp_path / "cameras.json")
    exported_cams = Cameras.from_json(tmp_path / "cameras.json")

    assert torch.allclose(exported_cams.camera_to_worlds, poses)
    assert (exported_cams.fx == 300.0).all()
    assert (exported_cams.fy == 300.0).all()
    assert (exported_cams.cx == 64.0).all()
    assert (exported_cams.cy == 64.0).all()
    assert (exported_cams.width == 128).all()
    assert (exported_cams.height == 128).all()
    assert (exported_cams.camera_type == 1).all()
    assert torch.allclose(exported_cams.times, times)
    assert set(metadata.keys()) == set(exported_cams.metadata.keys())
    for key, value in metadata.items():
        assert value.dtype == exported_cams.metadata[key].dtype
        assert value.shape == exported_cams.metadata[key].shape
        assert torch.allclose(value, exported_cams.metadata[key])


def test_cameras_indexing():
    """Test that the cameras class can be indexed."""
    cameras = Cameras(
        camera_to_worlds=torch.rand((10, 3, 4)),
        fx=torch.rand((10, 1)),
        fy=torch.rand((10, 1)),
        cx=torch.rand((10, 1)),
        cy=torch.rand((10, 1)),
        width=torch.randint(100, 1000, (10, 1)),
        height=torch.randint(100, 1000, (10, 1)),
        distortion_params=torch.rand((10, 6)),
        camera_type=torch.randint(0, 2, (10, 1)),
        times=torch.rand((10, 1)),
    )
    # Test indexing by camera_indices
    camera_indices = [0, 3, 6]
    indexed_cameras = cameras[camera_indices]
    assert indexed_cameras.shape == (3,)
    assert torch.allclose(indexed_cameras.camera_to_worlds, cameras.camera_to_worlds[camera_indices])
    assert torch.allclose(indexed_cameras.fx, cameras.fx[camera_indices])
    assert torch.allclose(indexed_cameras.fy, cameras.fy[camera_indices])
    assert torch.allclose(indexed_cameras.cx, cameras.cx[camera_indices])
    assert torch.allclose(indexed_cameras.cy, cameras.cy[camera_indices])
    assert torch.allclose(indexed_cameras.width, cameras.width[camera_indices])
    assert torch.allclose(indexed_cameras.height, cameras.height[camera_indices])
    assert torch.allclose(indexed_cameras.distortion_params, cameras.distortion_params[camera_indices])
    assert torch.allclose(indexed_cameras.camera_type, cameras.camera_type[camera_indices])
    assert torch.allclose(indexed_cameras.times, cameras.times[camera_indices])


def test_fisheye_camera():
    """Basic checks for fisheye camera ray directions (OpenCV coords)."""
    height = 50
    c2w = torch.eye(4)[None, :3, :]
    fisheye_camera = Cameras(
        cx=float(height),
        cy=float(height),
        fx=float(height),
        fy=float(height),
        camera_to_worlds=c2w,
        camera_type=CameraType.FISHEYE,
    )

    rb = fisheye_camera.generate_rays(camera_indices=0)
    dirs = rb.directions

    # Central pixel looks forward (+z)
    centre = dirs[height, height]
    assert centre[2] > 0.99

    # Extreme top pixel points up (negative y component)
    assert dirs[0, height][1] < -0.2

    # Extreme left pixel points left (negative x component)
    assert dirs[height, 0][0] < -0.2

    # All direction vectors must be unit-length (within tolerance)
    norms = torch.linalg.norm(dirs.view(-1, 3), dim=-1)
    assert torch.allclose(norms.mean(), torch.tensor(1.0), atol=1e-3)


def _central_pixel(height: int):
    """Return (row, col) index of forward-facing pixel for equirect-like panoramas."""
    return height // 2, height  # as used in previous tests (width = 2*height)


def test_fisheye624_camera():
    """Check basic properties of the FISHEYE624 model."""
    height = 60
    c2w = torch.eye(4)[None, :3, :]
    # 12-D distortion params: 6 radial, 2 tangential, 4 thin-prism
    dist = torch.zeros(1, 12)
    cam = Cameras(
        cx=float(height),
        cy=float(height),
        fx=float(height),
        fy=float(height),
        camera_to_worlds=c2w,
        camera_type=CameraType.FISHEYE624,
        distortion_params=dist,
    )

    rb = cam.generate_rays(camera_indices=0)
    dirs = rb.directions

    row, col = height, height  # central pixel in square fisheye crop
    assert dirs[row, col][2] > 0.95  # roughly forward
    # Unit-norm check
    assert torch.allclose(torch.linalg.norm(dirs.view(-1, 3), dim=-1).mean(), torch.tensor(1.0), atol=1e-2)
