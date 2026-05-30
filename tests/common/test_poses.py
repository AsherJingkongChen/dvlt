# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dvlt.common.pose import inverse_pose, multiply_poses, rotation_matrix_between, to4x4


def test_rotation_matrix_between():
    # Test with simple vectors
    a = torch.tensor([1.0, 0.0, 0.0])  # x-axis
    b = torch.tensor([0.0, 1.0, 0.0])  # y-axis

    # Rotation from x to y should be a 90-degree rotation around z
    rot = rotation_matrix_between(a, b)

    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(rot, expected, rtol=1e-5, atol=1e-5)

    # Test with non-unit vectors (should be normalized internally)
    a = torch.tensor([2.0, 0.0, 0.0])
    b = torch.tensor([0.0, 3.0, 0.0])

    rot = rotation_matrix_between(a, b)
    torch.testing.assert_close(rot, expected, rtol=1e-5, atol=1e-5)

    # Test with parallel vectors (should handle special case)
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([2.0, 0.0, 0.0])

    rot = rotation_matrix_between(a, b)

    # Should be identity matrix for parallel vectors
    expected = torch.eye(3)
    torch.testing.assert_close(rot, expected, rtol=1e-5, atol=1e-5)


def test_to4x4():
    # Create a 3x4 pose matrix
    pose_3x4 = torch.rand(3, 4)

    # Convert to 4x4
    pose_4x4 = to4x4(pose_3x4)

    # Check shape
    assert pose_4x4.shape == (4, 4)

    # Check that the top-left 3x4 is the same as the input
    torch.testing.assert_close(pose_4x4[:3, :], pose_3x4)

    # Check that the bottom row is [0, 0, 0, 1]
    torch.testing.assert_close(pose_4x4[3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    # Test with batch
    batch_pose = torch.rand(2, 3, 4)
    batch_4x4 = to4x4(batch_pose)

    # Check shape
    assert batch_4x4.shape == (2, 4, 4)

    # Check values
    for i in range(2):
        torch.testing.assert_close(batch_4x4[i, :3, :], batch_pose[i])
        torch.testing.assert_close(batch_4x4[i, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_inverse_pose():
    # Create a 3x4 pose matrix (rotation + translation)
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = torch.tensor([1.0, 2.0, 3.0])

    pose = torch.zeros(3, 4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    # Get inverse
    inv_pose = inverse_pose(pose)

    # Expected: R^T and -R^T * t
    expected_R = R.transpose(0, 1)
    expected_t = -expected_R @ t

    expected = torch.zeros(3, 4)
    expected[:3, :3] = expected_R
    expected[:3, 3] = expected_t

    torch.testing.assert_close(inv_pose, expected, rtol=1e-5, atol=1e-5)

    # Test with 4x4 pose
    pose_4x4 = to4x4(pose)
    inv_pose_4x4 = inverse_pose(pose_4x4)

    # Expected should also be 4x4
    expected_4x4 = torch.zeros(4, 4)
    expected_4x4[:3, :] = expected
    expected_4x4[3, 3] = 1.0

    torch.testing.assert_close(inv_pose_4x4, expected_4x4, rtol=1e-5, atol=1e-5)

    # Test with batch
    batch_pose = torch.stack([pose, pose])
    batch_inv = inverse_pose(batch_pose)

    # Check shape
    assert batch_inv.shape == (2, 3, 4)

    # Check values
    for i in range(2):
        torch.testing.assert_close(batch_inv[i], expected, rtol=1e-5, atol=1e-5)


def test_multiply_poses():
    # Create two 3x4 pose matrices
    R1 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t1 = torch.tensor([1.0, 2.0, 3.0])

    pose_a = torch.zeros(3, 4)
    pose_a[:3, :3] = R1
    pose_a[:3, 3] = t1

    R2 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t2 = torch.tensor([4.0, 5.0, 6.0])

    pose_b = torch.zeros(3, 4)
    pose_b[:3, :3] = R2
    pose_b[:3, 3] = t2

    # Multiply poses
    result = multiply_poses(pose_a, pose_b)

    # Expected: R1 @ R2 and R1 @ t2 + t1
    expected_R = R1 @ R2
    expected_t = R1 @ t2 + t1

    expected = torch.zeros(3, 4)
    expected[:3, :3] = expected_R
    expected[:3, 3] = expected_t

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    # Test with batch
    batch_a = torch.stack([pose_a, pose_a])
    batch_b = torch.stack([pose_b, pose_b])

    batch_result = multiply_poses(batch_a, batch_b)

    # Check shape
    assert batch_result.shape == (2, 3, 4)

    # Check values
    for i in range(2):
        torch.testing.assert_close(batch_result[i], expected, rtol=1e-5, atol=1e-5)
