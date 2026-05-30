# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch import Tensor

import dvlt.metric.pose
from dvlt.metric.pose import (
    batched_all_pairs,
    calculate_auc,
    compare_translation_by_angle,
    compute_pose_metrics,
    compute_rel_deg,
    rotation_angle,
    translation_angle,
)


def create_test_poses(rot_angle_deg: float) -> tuple[Tensor, Tensor]:
    """
    Create two poses with a known relative rotation and translation angle.

    Args:
        rot_angle_deg: Relative rotation angle in degrees

    Returns:
        tuple: (pose1, pose2) as 4x4 tensors
    """
    # Convert angles to radians
    rot_angle_rad = rot_angle_deg * np.pi / 180.0

    # Create first pose (identity)
    pose1 = torch.eye(4)

    # Create second pose
    pose2 = torch.eye(4)
    pose2[0, 0] = torch.cos(torch.tensor(rot_angle_rad))
    pose2[0, 2] = torch.sin(torch.tensor(rot_angle_rad))
    pose2[2, 0] = -torch.sin(torch.tensor(rot_angle_rad))
    pose2[2, 2] = torch.cos(torch.tensor(rot_angle_rad))
    return pose1, pose2


@pytest.mark.parametrize("angle", [0, 1, 5, 10, 45, 90, 180])
def test_rotation_angle(angle):
    # Create test rotations with known angles
    pose1, pose2 = create_test_poses(angle)

    # Measure the rotation directly
    rot1 = pose1[:3, :3].unsqueeze(0)  # Add batch dimension
    rot2 = pose2[:3, :3].unsqueeze(0)  # Add batch dimension

    computed_angle = rotation_angle(rot1, rot2).item()
    assert computed_angle == pytest.approx(angle, abs=1e-2)


@pytest.mark.parametrize("angle", [0, 1, 5, 10, 45, 90, 180])
def test_translation_angle(angle):
    # Create normalized translation vectors directly
    t1 = torch.tensor([[1.0, 0.0, 0.0]])  # Reference translation (x-axis)

    # Create a vector at the specified angle from x-axis in x-z plane
    angle_rad = angle * np.pi / 180.0
    t2 = torch.tensor([[np.cos(angle_rad), 0.0, np.sin(angle_rad)]])

    # Use the imported translation_angle function
    computed_angle = translation_angle(t1, t2).item()
    if angle > 90:
        assert computed_angle == pytest.approx(180 - angle, abs=1e-2)
    else:
        assert computed_angle == pytest.approx(angle, abs=1e-2)


def test_compare_translation_by_angle():
    # Test vectors with known angles
    t1 = torch.tensor([[1.0, 0.0, 0.0]])  # x-axis

    # Test various angles
    angles = [0, 30, 45, 60, 90]
    for angle_deg in angles:
        angle_rad = angle_deg * np.pi / 180.0
        t2 = torch.tensor([[np.cos(angle_rad), 0.0, np.sin(angle_rad)]])

        computed_angle_rad = compare_translation_by_angle(t1, t2).item()
        computed_angle_deg = computed_angle_rad * 180.0 / np.pi

        assert computed_angle_deg == pytest.approx(angle_deg, abs=1e-2)


def test_batched_all_pairs_single_batch():
    # Test with B=1, N=3
    i1, i2 = batched_all_pairs(1, 3)
    assert len(i1) == 3  # 3C2 = 3 pairs
    assert len(i2) == 3

    expected_i1 = torch.tensor([0, 0, 1])
    expected_i2 = torch.tensor([1, 2, 2])
    assert torch.allclose(i1, expected_i1)
    assert torch.allclose(i2, expected_i2)


def test_batched_all_pairs_multiple_batches():
    # Test with B=2, N=3
    i1, i2 = batched_all_pairs(2, 3)
    assert len(i1) == 6  # 2 batches * 3C2 = 6 pairs
    assert len(i2) == 6

    expected_i1 = torch.tensor([0, 0, 1, 3, 3, 4])
    expected_i2 = torch.tensor([1, 2, 2, 4, 5, 5])
    assert torch.allclose(i1, expected_i1)
    assert torch.allclose(i2, expected_i2)


def test_calculate_auc():
    # Create sample error distributions
    r_error = torch.tensor([2.0, 5.0, 8.0, 12.0, 18.0])
    t_error = torch.tensor([3.0, 7.0, 9.0, 15.0, 20.0])

    # Test with max_threshold=20
    auc = calculate_auc(r_error, t_error, 20)

    # The exact value depends on binning, but we can verify it's in a reasonable range
    assert 0 < auc.item() < 1


def test_compute_pose_metrics():
    """Test the compute_pose_metrics function."""
    # Create simple rotation and translation errors
    r_error = torch.tensor([5.0, 12.0, 18.0])
    t_error = torch.tensor([8.0, 12.0, 15.0])

    # Mock the compute_rel_deg function to return our predefined errors
    original_compute_rel_deg = compute_rel_deg

    try:
        # Define a mock function
        def mock_compute_rel_deg(pred_poses, gt_poses):
            return r_error, t_error

        # Temporarily replace the original function
        dvlt.metric.pose.compute_rel_deg = mock_compute_rel_deg

        # Create dummy poses (content doesn't matter as we've mocked the compute_rel_deg function)
        dummy_poses = torch.eye(4).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 4, 4]

        # Compute metrics with custom thresholds
        metrics = compute_pose_metrics(dummy_poses, dummy_poses, thresholds=(5, 10, 20))

        # Check the mean errors
        assert metrics["R_error"] == pytest.approx(r_error.mean(), abs=1e-6)
        assert metrics["T_error"] == pytest.approx(t_error.mean(), abs=1e-6)

        # Check that all expected metrics exist
        assert "Racc_5" in metrics
        assert "Racc_10" in metrics
        assert "Racc_20" in metrics
        assert "Tacc_5" in metrics
        assert "Tacc_10" in metrics
        assert "Tacc_20" in metrics
        assert "Auc_5" in metrics
        assert "Auc_10" in metrics
        assert "Auc_20" in metrics

    finally:
        # Restore the original function
        dvlt.metric.pose.compute_rel_deg = original_compute_rel_deg
