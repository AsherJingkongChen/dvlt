# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from dvlt.common.projection import (
    create_meshgrid,
    depth_to_points,
    generate_depth_map,
    inverse_pinhole,
    points_inside_image,
    project_points,
    unproject_points,
)


def test_inverse_pinhole():
    # Test with a single intrinsic matrix
    intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    inv = inverse_pinhole(intrinsics)

    # Expected inverse
    torch.testing.assert_close(inv, intrinsics.inverse())

    # Test with a batch of intrinsic matrices
    intrinsics2 = intrinsics.clone()
    intrinsics2[:2] *= 2.0
    batch_intrinsics = torch.stack([intrinsics, intrinsics2])
    batch_inv = inverse_pinhole(batch_intrinsics)

    # Expected batch inverse
    expected_batch = torch.stack([intrinsics.inverse(), intrinsics2.inverse()])
    torch.testing.assert_close(batch_inv, expected_batch)


def test_points_inside_image():
    # Create test points
    points = torch.tensor(
        [
            [10.0, 10.0],  # inside
            [5.0, 5.0],  # inside
            [-1.0, 10.0],  # outside (x < 0)
            [10.0, -1.0],  # outside (y < 0)
            [100.0, 10.0],  # outside (x > w-1)
            [10.0, 100.0],  # outside (y > h-1)
        ]
    )

    depths = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    img_hw = (50, 80)  # height=50, width=80

    mask = points_inside_image(points, depths, img_hw)

    expected_mask = torch.tensor([True, True, False, False, False, False])
    torch.testing.assert_close(mask, expected_mask)

    # Test with negative depth
    depths[0] = -1.0
    mask = points_inside_image(points, depths, img_hw)
    expected_mask = torch.tensor([False, True, False, False, False, False])
    torch.testing.assert_close(mask, expected_mask)

    # Test with tensor image dimensions
    img_hw_tensor = torch.tensor([50, 80])
    mask = points_inside_image(points, depths, img_hw_tensor)
    torch.testing.assert_close(mask, expected_mask)


def test_project_points():
    # Test points (3D)
    points = torch.tensor(
        [
            [10.0, 20.0, 5.0],
            [4.0, 8.0, 2.0],
        ]
    )

    # Intrinsic matrix
    intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

    # Project points to 2D
    projected = project_points(points, intrinsics)

    # Expected: (X/Z, Y/Z) * intrinsics
    expected = torch.tensor(
        [
            [4.0, 7.0],  # (10/5 = 2, 20/5 = 4) -> (2*1 + 2, 4*1 + 3)
            [4.0, 7.0],  # (4/2 = 2, 8/2 = 4) -> (2*1 + 2, 4*1 + 3)
        ]
    )
    torch.testing.assert_close(projected, expected)

    # Test with batch
    batch_points = torch.stack([points, points * 2])
    batch_intrinsics = torch.stack([intrinsics, intrinsics])
    batch_projected = project_points(batch_points, batch_intrinsics)

    expected_batch = torch.stack(
        [
            expected,
            torch.tensor(
                [
                    [4.0, 7.0],  # (20/10 = 2, 40/10 = 4) -> (2*1 + 2, 4*1 + 3)
                    [4.0, 7.0],  # (8/4 = 2, 16/4 = 4) -> (2*1 + 2, 4*1 + 3)
                ]
            ),
        ]
    )
    torch.testing.assert_close(batch_projected, expected_batch)

    # Test with 4D tensors (B, S, N, 3) - like motion model case
    B, S, N = 2, 3, 2
    points_4d = points.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, N, 3]
    intrinsics_4d = intrinsics.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, 3, 3]

    projected_4d = project_points(points_4d, intrinsics_4d)
    expected_4d = expected.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, N, 2]

    assert projected_4d.shape == (B, S, N, 2)
    torch.testing.assert_close(projected_4d, expected_4d)

    # Test with arbitrary dimensions and single intrinsics (broadcasting)
    points_multi = torch.randn(2, 3, 4, 3)
    points_multi[..., 2] = torch.abs(points_multi[..., 2]) + 1  # Ensure positive depth
    intrinsics_single = torch.eye(3) * 100
    intrinsics_single[2, 2] = 1

    projected_multi = project_points(points_multi, intrinsics_single)
    assert projected_multi.shape == (2, 3, 4, 2)

    # Test broadcasting with different intrinsics shapes
    B, N = 2, 3
    points_bn3 = torch.tensor(
        [[[10.0, 20.0, 5.0], [6.0, 12.0, 3.0], [2.0, 4.0, 1.0]], [[8.0, 16.0, 2.0], [12.0, 24.0, 4.0], [4.0, 8.0, 2.0]]]
    )  # [B, N, 3]

    # Test 1: Single intrinsics (3, 3) - should broadcast to all points
    base_intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    projected_single = project_points(points_bn3, base_intrinsics)
    assert projected_single.shape == (B, N, 2), f"Expected shape ({B}, {N}, 2), got {projected_single.shape}"

    # Test 2: Per-batch intrinsics (B, 3, 3)
    intrinsics_b33 = base_intrinsics.unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]
    intrinsics_b33[1, 0, 0] = 1.1  # Modify fx for batch 1
    projected_batch = project_points(points_bn3, intrinsics_b33)
    assert projected_batch.shape == (B, N, 2), f"Expected shape ({B}, {N}, 2), got {projected_batch.shape}"

    # Test 3: Per-point intrinsics (B, N, 3, 3)
    intrinsics_bn33 = base_intrinsics.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # [B, N, 3, 3]
    intrinsics_bn33[0, 1, 0, 0] = 1.1  # Modify fx for batch 0, point 1
    intrinsics_bn33[1, 2, 1, 1] = 1.2  # Modify fy for batch 1, point 2
    projected_per_point = project_points(points_bn3, intrinsics_bn33)
    assert projected_per_point.shape == (B, N, 2), f"Expected shape ({B}, {N}, 2), got {projected_per_point.shape}"

    # Verify correctness for the per-point case
    # For batch 0, point 0: [10.0, 20.0, 5.0] with base intrinsics
    hom_0_0 = points_bn3[0, 0] / points_bn3[0, 0, 2]  # [2.0, 4.0, 1.0]
    expected_0_0 = (hom_0_0 @ intrinsics_bn33[0, 0].T)[:2]  # Apply intrinsics and take first 2 coords
    torch.testing.assert_close(projected_per_point[0, 0], expected_0_0, rtol=1e-4, atol=1e-4)


def test_unproject_points():
    # 2D points
    points_2d = torch.tensor(
        [
            [4.0, 7.0],
            [4.0, 7.0],
        ]
    )

    # Depths
    depths = torch.tensor([5.0, 2.0])

    # Intrinsic matrix
    intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

    # Unproject points to 3D
    unprojected = unproject_points(points_2d, depths, intrinsics)

    # Expected: Use inverse intrinsics to project back
    expected = torch.tensor(
        [
            [10.0, 20.0, 5.0],
            [4.0, 8.0, 2.0],
        ]
    )
    torch.testing.assert_close(unprojected, expected, rtol=1e-4, atol=1e-4)

    # Test with depths as column vector
    depths_col = depths.unsqueeze(-1)
    unprojected_col = unproject_points(points_2d, depths_col, intrinsics)
    torch.testing.assert_close(unprojected_col, expected, rtol=1e-4, atol=1e-4)

    # Test with batch
    batch_points_2d = torch.stack([points_2d, points_2d])
    batch_depths = torch.stack([depths, depths * 2])
    batch_intrinsics = torch.stack([intrinsics, intrinsics])

    batch_unprojected = unproject_points(batch_points_2d, batch_depths, batch_intrinsics)

    expected_batch = torch.stack([expected, expected * 2])
    torch.testing.assert_close(batch_unprojected, expected_batch, rtol=1e-4, atol=1e-4)

    # Test with 4D tensors (B, S, N, 2)
    B, S, N = 2, 3, 2
    points_4d = points_2d.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, N, 2]
    depths_4d = depths.unsqueeze(0).unsqueeze(0).repeat(B, S, 1)  # [B, S, N]
    intrinsics_4d = intrinsics.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, 3, 3]

    unprojected_4d = unproject_points(points_4d, depths_4d, intrinsics_4d)
    expected_4d = expected.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, N, 3]

    assert unprojected_4d.shape == (B, S, N, 3)
    torch.testing.assert_close(unprojected_4d, expected_4d, rtol=1e-4, atol=1e-4)

    # Test with arbitrary dimensions and single intrinsics (broadcasting)
    points_multi = torch.tensor([[[4.0, 7.0]], [[4.0, 7.0]]])  # [2, 1, 2]
    depths_multi = torch.tensor([[5.0], [2.0]])  # [2, 1]
    intrinsics_single = intrinsics

    unprojected_multi = unproject_points(points_multi, depths_multi, intrinsics_single)
    expected_multi = torch.tensor([[[10.0, 20.0, 5.0]], [[4.0, 8.0, 2.0]]])  # [2, 1, 3]

    assert unprojected_multi.shape == (2, 1, 3)
    torch.testing.assert_close(unprojected_multi, expected_multi, rtol=1e-4, atol=1e-4)

    # Test broadcasting with different intrinsics shapes
    B, N = 2, 3
    points_bn2 = torch.tensor(
        [[[4.0, 7.0], [6.0, 9.0], [2.0, 5.0]], [[8.0, 11.0], [10.0, 13.0], [4.0, 7.0]]]
    )  # [B, N, 2]
    depths_bn = torch.tensor([[5.0, 3.0, 1.0], [2.0, 4.0, 6.0]])  # [B, N]

    # Test 1: Single intrinsics (3, 3) - should broadcast to all points
    base_intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    unprojected_single = unproject_points(points_bn2, depths_bn, base_intrinsics)
    assert unprojected_single.shape == (B, N, 3), f"Expected shape ({B}, {N}, 3), got {unprojected_single.shape}"

    # Test 2: Per-batch intrinsics (B, 3, 3)
    intrinsics_b33 = base_intrinsics.unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]
    intrinsics_b33[1, 0, 0] = 1.1  # Modify fx for batch 1
    unprojected_batch = unproject_points(points_bn2, depths_bn, intrinsics_b33)
    assert unprojected_batch.shape == (B, N, 3), f"Expected shape ({B}, {N}, 3), got {unprojected_batch.shape}"

    # Test 3: Per-point intrinsics (B, N, 3, 3)
    intrinsics_bn33 = base_intrinsics.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # [B, N, 3, 3]
    intrinsics_bn33[0, 1, 0, 0] = 1.1  # Modify fx for batch 0, point 1
    intrinsics_bn33[1, 2, 1, 1] = 1.2  # Modify fy for batch 1, point 2
    unprojected_per_point = unproject_points(points_bn2, depths_bn, intrinsics_bn33)
    assert unprojected_per_point.shape == (B, N, 3), f"Expected shape ({B}, {N}, 3), got {unprojected_per_point.shape}"

    # Verify correctness for the per-point case
    # For batch 0, point 0: [4.0, 7.0] with depth 5.0 and base intrinsics
    hom_0_0 = torch.tensor([4.0, 7.0, 1.0])
    inv_intrinsics_0_0 = inverse_pinhole(intrinsics_bn33[0, 0])
    expected_0_0 = (hom_0_0 @ inv_intrinsics_0_0.T) * depths_bn[0, 0]
    torch.testing.assert_close(unprojected_per_point[0, 0], expected_0_0, rtol=1e-4, atol=1e-4)


def test_create_meshgrid():
    # Test with normalized coordinates
    height, width = 3, 4
    grid = create_meshgrid(height, width, normalized_coordinates=True)

    # Expected: grid with shape (3, 4, 2) with normalized coordinates
    # X should go from -1 to 1 horizontally
    # Y should go from -1 to 1 vertically
    assert grid.shape == (3, 4, 2)
    torch.testing.assert_close(grid[0, 0, 0], torch.tensor(-1.0))  # Top-left x
    torch.testing.assert_close(grid[0, -1, 0], torch.tensor(1.0))  # Top-right x
    torch.testing.assert_close(grid[0, 0, 1], torch.tensor(-1.0))  # Top-left y
    torch.testing.assert_close(grid[-1, 0, 1], torch.tensor(1.0))  # Bottom-left y

    # Test with unnormalized coordinates
    grid = create_meshgrid(height, width, normalized_coordinates=False)
    assert grid.shape == (3, 4, 2)
    torch.testing.assert_close(grid[0, 0, 0], torch.tensor(0.0))  # Top-left x
    torch.testing.assert_close(grid[0, -1, 0], torch.tensor(3.0))  # Top-right x
    torch.testing.assert_close(grid[0, 0, 1], torch.tensor(0.0))  # Top-left y
    torch.testing.assert_close(grid[-1, 0, 1], torch.tensor(2.0))  # Bottom-left y


@pytest.mark.parametrize("batch_size", [1, 2])
def test_depth_to_points(batch_size):
    # Create a small depth map
    height, width = 3, 4
    if batch_size == 1:
        depth_map = torch.ones((height, width))
        intrinsics = torch.tensor([[1.0, 0.0, width / 2], [0.0, 1.0, height / 2], [0.0, 0.0, 1.0]])
    else:
        depth_map = torch.ones((batch_size, height, width))
        intrinsics = torch.stack(
            [torch.tensor([[1.0, 0.0, width / 2], [0.0, 1.0, height / 2], [0.0, 0.0, 1.0]])] * batch_size
        )

    # Get 3D points
    points = depth_to_points(depth_map, intrinsics)

    # Check shape - now returns (..., H, W, 3) preserving spatial structure
    expected_shape = (height, width, 3) if batch_size == 1 else (batch_size, height, width, 3)
    assert points.shape == expected_shape

    # For unit focal length and depth=1, the z component should be 1 everywhere
    z_values = points[..., 2]
    torch.testing.assert_close(z_values, torch.ones_like(z_values))

    # Test with different depth values
    if batch_size == 1:
        depth_map = torch.full((height, width), 2.0)
    else:
        depth_map = torch.full((batch_size, height, width), 2.0)

    points = depth_to_points(depth_map, intrinsics)
    z_values = points[..., 2]
    torch.testing.assert_close(z_values, torch.full_like(z_values, 2.0))


def test_generate_depth_map():
    # Create 3D points
    points = torch.tensor(
        [
            [10.0, 20.0, 5.0],  # Projects to (4, 7)
            [4.0, 8.0, 2.0],  # Projects to (4, 7) - depth 2.0 is smaller than 5.0
            [30.0, 15.0, 6.0],  # Projects to (7, 6)
        ]
    )

    # Intrinsic matrix
    intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

    img_hw = (10, 15)  # height=10, width=15

    # Generate depth map
    depth_map = generate_depth_map(points, intrinsics, img_hw)

    # Check shape
    assert depth_map.shape == img_hw

    # Check specific values - note: pixel coordinates are (y, x) with depth_map indexing
    # For pixels with multiple points, the smallest depth should be kept
    assert depth_map[7, 4] == 2.0  # Smallest depth between first and second point at (4, 7)
    assert depth_map[6, 7] == 6.0  # From third point at (7, 6)

    # All other values should be 0
    mask = torch.ones_like(depth_map, dtype=torch.bool)
    mask[7, 4] = False
    mask[6, 7] = False
    assert torch.all(depth_map[mask] == 0)
