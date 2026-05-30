# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test the ray classes. Adapted from nerfstudio.
"""

import torch

from dvlt.struct.rays import RayBundle


def test_ray_bundle_creation():
    """Test creation of ray bundle with required fields."""
    batch_size = (2, 3)
    origins = torch.ones((*batch_size, 3))
    directions = torch.zeros((*batch_size, 3))
    pixel_area = torch.ones((*batch_size, 1))

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
    )

    assert ray_bundle.shape == batch_size
    assert ray_bundle.origins.shape == (*batch_size, 3)
    assert ray_bundle.directions.shape == (*batch_size, 3)
    assert ray_bundle.pixel_area.shape == (*batch_size, 1)
    assert ray_bundle.camera_indices is None
    assert ray_bundle.nears is None
    assert ray_bundle.fars is None

    # Test length calculation
    assert len(ray_bundle) == origins.numel() // 3


def test_ray_bundle_set_camera_indices():
    """Test setting camera indices for all rays."""
    batch_size = (2, 3)
    origins = torch.ones((*batch_size, 3))
    directions = torch.zeros((*batch_size, 3))
    pixel_area = torch.ones((*batch_size, 1))

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
    )

    camera_index = 5
    ray_bundle.set_camera_indices(camera_index)

    assert ray_bundle.camera_indices is not None
    assert ray_bundle.camera_indices.shape == (*batch_size, 1)
    assert torch.all(ray_bundle.camera_indices == camera_index)


def test_ray_bundle_with_optional_fields():
    """Test creation of ray bundle with all fields."""
    batch_size = (2, 3)
    origins = torch.ones((*batch_size, 3))
    directions = torch.zeros((*batch_size, 3))
    pixel_area = torch.ones((*batch_size, 1))
    camera_indices = torch.zeros((*batch_size, 1), dtype=torch.long)
    nears = torch.zeros((*batch_size, 1))
    fars = torch.ones((*batch_size, 1)) * 10.0
    times = torch.linspace(0, 1, batch_size[0] * batch_size[1]).reshape((*batch_size, 1))
    metadata = {"semantic": torch.randint(0, 10, (*batch_size, 5))}

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars,
        times=times,
        metadata=metadata,
    )

    assert ray_bundle.shape == batch_size
    assert ray_bundle.origins.shape == (*batch_size, 3)
    assert ray_bundle.directions.shape == (*batch_size, 3)
    assert ray_bundle.pixel_area.shape == (*batch_size, 1)
    assert ray_bundle.camera_indices.shape == (*batch_size, 1)
    assert ray_bundle.nears.shape == (*batch_size, 1)
    assert ray_bundle.fars.shape == (*batch_size, 1)
    assert ray_bundle.times.shape == (*batch_size, 1)
    assert "semantic" in ray_bundle.metadata
    assert ray_bundle.metadata["semantic"].shape == (*batch_size, 5)


def test_ray_bundle_slicing():
    """Test slicing of ray bundle."""
    batch_size = (2, 3, 4)
    origins = torch.ones((*batch_size, 3))
    directions = torch.zeros((*batch_size, 3))
    pixel_area = torch.ones((*batch_size, 1))

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
    )

    # Test single index
    sliced = ray_bundle[0]
    assert sliced.shape == batch_size[1:]
    assert sliced.origins.shape == (*batch_size[1:], 3)

    # Test multiple indices
    sliced = ray_bundle[0, 1]
    assert sliced.shape == batch_size[2:]
    assert sliced.origins.shape == (*batch_size[2:], 3)

    # Test with ellipsis
    sliced = ray_bundle[..., 0]
    assert sliced.shape == batch_size[:-1]
    assert sliced.origins.shape == (*batch_size[:-1], 3)


def test_ray_bundle_reshape_and_flatten():
    """Test reshaping and flattening of ray bundle."""
    batch_size = (2, 3, 4)
    origins = torch.ones((*batch_size, 3))
    directions = torch.zeros((*batch_size, 3))
    pixel_area = torch.ones((*batch_size, 1))

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
    )

    # Test reshape
    new_shape = (6, 4)
    reshaped = ray_bundle.reshape(new_shape)
    assert reshaped.shape == new_shape
    assert reshaped.origins.shape == (*new_shape, 3)

    # Test flatten
    flattened = ray_bundle.flatten()
    assert flattened.shape == (batch_size[0] * batch_size[1] * batch_size[2],)
    assert flattened.origins.shape == (batch_size[0] * batch_size[1] * batch_size[2], 3)


def test_ray_bundle_get_row_major_sliced():
    """Test get_row_major_sliced_ray_bundle method."""
    batch_size = (2, 3, 4)
    origins = torch.ones((*batch_size, 3))
    directions = torch.zeros((*batch_size, 3))
    pixel_area = torch.ones((*batch_size, 1))

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
    )

    # Get slice
    start_idx = 5
    end_idx = 15
    sliced = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)

    assert len(sliced) == end_idx - start_idx
    assert sliced.origins.shape[0] == end_idx - start_idx
    assert sliced.directions.shape[0] == end_idx - start_idx
    assert sliced.pixel_area.shape[0] == end_idx - start_idx
