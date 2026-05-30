# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from dvlt.metric.pointcloud import (
    _chamfer_distance,
    compute_chamfer_distance,
    compute_point_cloud_metrics,
    compute_point_cloud_metrics_multi_threshold,
)
from tests.utils.fixtures import get_optimal_device as get_device


def test_gpu_compute_point_cloud_metrics_identical_points() -> None:
    """Test metrics when predicted and ground truth points are identical."""
    device = get_device()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )

    metrics = compute_point_cloud_metrics(points, points)

    # With identical point clouds, all metrics should be 1.0
    assert metrics["precision"].item() == pytest.approx(1.0, abs=1e-6)
    assert metrics["recall"].item() == pytest.approx(1.0, abs=1e-6)
    assert metrics["f_score"].item() == pytest.approx(1.0, abs=1e-6)


def test_gpu_compute_point_cloud_metrics_with_threshold() -> None:
    """Test metrics with different thresholds."""
    device = get_device()
    pred_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
    )

    # Slightly offset points from predictions
    gt_points = torch.tensor(
        [
            [0.005, 0.005, 0.005],  # Within 0.01 threshold
            [1.02, 0.0, 0.0],  # Outside 0.01 threshold
            [0.0, 0.99, 0.0],  # Within 0.01 threshold
        ],
        device=device,
    )

    # With threshold 0.01
    metrics_default = compute_point_cloud_metrics(pred_points, gt_points, threshold=0.01)

    # 2 of 3 points are within threshold for both precision and recall
    assert metrics_default["precision"].item() == pytest.approx(2 / 3, abs=1e-6)
    assert metrics_default["recall"].item() == pytest.approx(2 / 3, abs=1e-6)
    assert metrics_default["f_score"].item() == pytest.approx(2 / 3, abs=1e-6)

    # With higher threshold, all points should be considered correct
    metrics_high = compute_point_cloud_metrics(pred_points, gt_points, threshold=0.05)
    assert metrics_high["precision"].item() == pytest.approx(1.0, abs=1e-6)
    assert metrics_high["recall"].item() == pytest.approx(1.0, abs=1e-6)
    assert metrics_high["f_score"].item() == pytest.approx(1.0, abs=1e-6)


def test_gpu_compute_chamfer_distance_identical_points() -> None:
    """Test Chamfer distance with identical point clouds."""
    device = get_device()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
    )

    distance = compute_chamfer_distance(points, points)

    # Identical point clouds should have zero distance
    assert distance.item() == pytest.approx(0.0, abs=1e-6)


def test_gpu_compute_chamfer_distance_unidirectional() -> None:
    """Test unidirectional Chamfer distance."""
    device = get_device()
    pred_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        device=device,
    )

    gt_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Different point
        ],
        device=device,
    )

    # Bidirectional
    distance_bi = compute_chamfer_distance(pred_points, gt_points, bidirectional=True)

    # Unidirectional
    distance_uni = compute_chamfer_distance(pred_points, gt_points, bidirectional=False)

    # The unidirectional distance should be different from bidirectional
    assert distance_bi != distance_uni
    assert distance_bi > 0.0
    assert distance_uni > 0.0


def test_gpu_compute_point_cloud_metrics_multi_threshold() -> None:
    """Test metrics with multiple thresholds."""
    device = get_device()
    pred_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
    )

    # Points at varying distances from predictions
    gt_points = torch.tensor(
        [
            [0.004, 0.004, 0.004],  # Within 0.01 threshold
            [1.015, 0.0, 0.0],  # Within 0.02 threshold but outside 0.01
            [0.0, 0.975, 0.0],  # Within 0.03 threshold but outside 0.01
        ],
        device=device,
    )

    # Only evaluate two thresholds (0.01 m and 0.02 m)
    thresholds = [0.01, 0.02]
    metrics = compute_point_cloud_metrics_multi_threshold(pred_points, gt_points, thresholds=thresholds)

    # For the 0.01 m threshold, only one point is within range
    assert metrics["precision@0.010m"].item() == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["recall@0.010m"].item() == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["f_score@0.010m"].item() == pytest.approx(1 / 3, abs=1e-6)

    # For the 0.02 m threshold, two points should be within range
    assert metrics["precision@0.020m"].item() == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["recall@0.020m"].item() == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["f_score@0.020m"].item() == pytest.approx(2 / 3, abs=1e-6)

    # Check average F-score across the evaluated thresholds
    avg_f_score = (metrics["f_score@0.010m"].item() + metrics["f_score@0.020m"].item()) / 2
    assert metrics["f_score_avg"].item() == pytest.approx(avg_f_score, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_gpu_device_handling() -> None:
    """Test device handling logic."""
    # Create points on CPU and CUDA
    points_cpu = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    points_cuda = points_cpu.to("cuda")

    # Test with CPU tensors
    metrics_cpu = compute_point_cloud_metrics(points_cpu, points_cpu)
    assert metrics_cpu["f_score"].item() == pytest.approx(1.0, abs=1e-6)

    # Test with CUDA tensors
    metrics_cuda = compute_point_cloud_metrics(points_cuda, points_cuda)
    assert metrics_cuda["f_score"].item() == pytest.approx(1.0, abs=1e-6)

    # Test mixed case (should raise an assertion error)
    with pytest.raises(AssertionError):
        compute_point_cloud_metrics(points_cpu, points_cuda)

    # Test our _chamfer_distance implementation on different devices
    points_cpu_batch = points_cpu.unsqueeze(0)
    points_cuda_batch = points_cuda.unsqueeze(0)

    result_cpu = _chamfer_distance(points_cpu_batch, points_cpu_batch)
    result_cuda = _chamfer_distance(points_cuda_batch, points_cuda_batch)

    assert result_cpu.item() == pytest.approx(result_cuda.item(), abs=1e-6)
