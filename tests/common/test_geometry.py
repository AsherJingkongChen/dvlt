# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dvlt.common.geometry import (
    _compute_residual_and_jacobian,
    normalize_with_norm,
    radial_and_tangential_undistort,
    transform_points,
)


def test_transform_points():
    # Test with 3D points and a single transform
    points = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    # Create a transformation matrix that translates by (10, 20, 30)
    transform = torch.tensor(
        [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
    )

    transformed = transform_points(points, transform)

    expected = torch.tensor(
        [
            [11.0, 22.0, 33.0],
            [14.0, 25.0, 36.0],
        ]
    )
    torch.testing.assert_close(transformed, expected)

    # Test with 3x4 transform matrix (should be automatically converted to 4x4)
    points_simple = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    # 3x4 transform matrix that translates by (5, 10, 15)
    transform_3x4 = torch.tensor([[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 10.0], [0.0, 0.0, 1.0, 15.0]])

    transformed_3x4 = transform_points(points_simple, transform_3x4)

    expected_3x4 = torch.tensor([[6.0, 12.0, 18.0], [5.0, 10.0, 15.0]])
    torch.testing.assert_close(transformed_3x4, expected_3x4)

    # Test with 3D points and a batch of transforms
    points_3d = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    # Create two transformation matrices
    transforms = torch.stack(
        [
            torch.tensor([[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0], [0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0, -5.0], [0.0, 1.0, 0.0, -10.0], [0.0, 0.0, 1.0, -15.0], [0.0, 0.0, 0.0, 1.0]]),
        ]
    )

    # Batch of points
    batch_points = torch.stack([points_3d, points_3d])

    transformed = transform_points(batch_points, transforms)

    expected = torch.stack(
        [
            torch.tensor(
                [
                    [11.0, 22.0, 33.0],
                    [14.0, 25.0, 36.0],
                ]
            ),
            torch.tensor(
                [
                    [-4.0, -8.0, -12.0],
                    [-1.0, -5.0, -9.0],
                ]
            ),
        ]
    )
    torch.testing.assert_close(transformed, expected)


def test_compute_residual_and_jacobian():
    # Create undistorted coordinates
    x = torch.tensor([0.1, 0.2])
    y = torch.tensor([0.3, 0.4])

    # Create distorted coordinates (just using the same for simplicity in test)
    xd = x.clone()
    yd = y.clone()

    # No distortion parameters
    distortion_params = torch.zeros(2, 6)

    # Compute residuals and jacobians
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(x, y, xd, yd, distortion_params)

    # With no distortion, residuals should be zero
    torch.testing.assert_close(fx, torch.zeros_like(fx))
    torch.testing.assert_close(fy, torch.zeros_like(fy))

    # With no distortion, jacobians should be identity-like
    torch.testing.assert_close(fx_x, torch.ones_like(fx_x))
    torch.testing.assert_close(fy_y, torch.ones_like(fy_y))
    torch.testing.assert_close(fx_y, torch.zeros_like(fx_y))
    torch.testing.assert_close(fy_x, torch.zeros_like(fy_x))

    # Test with non-zero distortion
    # k1 = 0.1 (first parameter)
    distortion_params = torch.zeros(2, 6)
    distortion_params[:, 0] = 0.1

    # Compute with distortion
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(x, y, xd, yd, distortion_params)

    # Should have non-zero residuals now
    assert torch.any(fx != 0)
    assert torch.any(fy != 0)


def test_radial_and_tangential_undistort():
    # Create distorted coordinates
    coords = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    # No distortion case
    distortion_params = torch.zeros(2, 6)

    # Undistort
    undistorted = radial_and_tangential_undistort(coords, distortion_params)

    # With no distortion, should return original coordinates
    torch.testing.assert_close(undistorted, coords)

    # Test with distortion (simple case with only k1)
    distortion_params = torch.zeros(1, 6)
    distortion_params[0, 0] = 0.1  # k1 = 0.1

    # Create a pre-distorted point
    # For a point (x,y), with r²=x²+y², the distorted point with only k1 is:
    # (x * (1 + k1*r²), y * (1 + k1*r²))
    x, y = 0.5, 0.5
    r_squared = x**2 + y**2
    distortion_factor = 1 + 0.1 * r_squared
    distorted_x = x * distortion_factor
    distorted_y = y * distortion_factor

    distorted_coords = torch.tensor([[distorted_x, distorted_y]])

    # Undistort
    undistorted = radial_and_tangential_undistort(distorted_coords, distortion_params)

    # Should recover the original point
    expected = torch.tensor([[0.5, 0.5]])
    torch.testing.assert_close(undistorted, expected, rtol=1e-4, atol=1e-4)


def test_normalize_with_norm():
    # Test 1D normalization
    x = torch.tensor([3.0, 4.0])
    normalized, norm = normalize_with_norm(x, dim=0)

    expected_norm = torch.tensor([5.0])
    expected_normalized = torch.tensor([3.0 / 5.0, 4.0 / 5.0])

    torch.testing.assert_close(norm, expected_norm)
    torch.testing.assert_close(normalized, expected_normalized)

    # Test 2D normalization along rows
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    normalized, norm = normalize_with_norm(x, dim=1)

    expected_norm = torch.tensor([[5.0], [13.0]])
    expected_normalized = torch.tensor([[3.0 / 5.0, 4.0 / 5.0], [5.0 / 13.0, 12.0 / 13.0]])

    torch.testing.assert_close(norm, expected_norm)
    torch.testing.assert_close(normalized, expected_normalized)

    # Test with zeros (should use epsilon)
    x = torch.zeros(3)
    normalized, norm = normalize_with_norm(x, dim=0)

    # Should use epsilon value from constants
    # We'll just check that the norm is small but positive
    assert norm > 0
    assert norm < 1e-6
