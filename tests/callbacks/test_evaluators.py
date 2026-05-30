# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator callbacks."""

import tempfile
from unittest.mock import Mock

import pytest
import torch
from accelerate import Accelerator

from dvlt.callbacks import DepthEvaluator, IntrinsicsEvaluator, PointmapEvaluator, PoseEvaluator
from dvlt.callbacks.base import Callback
from dvlt.common.constants import DataField, PredictionField
from dvlt.struct.cameras import Cameras
from dvlt.struct.util import extri_intri_to_cameras


@pytest.fixture
def mock_trainer():
    """Create a mock trainer for testing."""
    trainer = Mock()
    # Create Accelerator - it will use the state initialized by the autouse fixture
    trainer.accelerator = Accelerator()

    # Override gather_for_metrics to return input as-is for testing
    trainer.accelerator.gather_for_metrics = lambda x: x
    return trainer


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    B, S, H, W = 1, 3, 64, 64
    return {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: torch.rand(B, S, 3, H, W),
        DataField.DEPTHS: torch.rand(B, S, H, W),
        DataField.WORLD_POINTS: torch.randn(B, S, H, W, 3),
        DataField.POINT_MASKS: torch.ones(B, S, H, W, dtype=torch.bool),
        DataField.EXTRINSICS_C2W: torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1),
        DataField.INTRINSICS: torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1),
        DataField.SCALE_FACTOR: torch.ones(B),  # [B] tensor as per documentation
    }


@pytest.fixture
def sample_predictions(sample_batch):
    """Create sample predictions matching the batch."""
    B = sample_batch[DataField.IMAGES].shape[0]
    S, H, W = sample_batch[DataField.IMAGES].shape[1], 64, 64
    extr = sample_batch[DataField.EXTRINSICS_C2W]
    intr = sample_batch[DataField.INTRINSICS]
    # Flatten batch and sequence dimensions for camera creation, then reshape
    extr_flat = extr.view(-1, 4, 4)
    intr_flat = intr.view(-1, 3, 3)
    cameras_flat = extri_intri_to_cameras(extr_flat, intr_flat, (H, W))
    # Reshape all camera attributes to [B, S, ...]
    cameras = Cameras(
        camera_to_worlds=cameras_flat.camera_to_worlds.view(B, S, 3, 4),
        fx=cameras_flat.fx.view(B, S, 1),
        fy=cameras_flat.fy.view(B, S, 1),
        cx=cameras_flat.cx.view(B, S, 1),
        cy=cameras_flat.cy.view(B, S, 1),
        width=cameras_flat.width.view(B, S, 1) if cameras_flat.width.dim() > 0 else cameras_flat.width,
        height=cameras_flat.height.view(B, S, 1) if cameras_flat.height.dim() > 0 else cameras_flat.height,
    )

    return {
        PredictionField.CAMERAS: cameras,
        PredictionField.DEPTHS: torch.rand(B, S, H, W),
        PredictionField.DEPTHS_CONF: torch.rand(B, S, H, W),
        PredictionField.WORLD_POINTS: torch.randn(B, S, H, W, 3),
    }


def test_depth_evaluator(mock_trainer, sample_batch, sample_predictions):
    """Test DepthEvaluator functionality."""
    evaluator = DepthEvaluator()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize
        evaluator.on_test_dataset_start(mock_trainer, "test_dataset")

        # Process batch
        metrics = evaluator.on_test_batch(sample_batch, sample_predictions, temp_dir, 0, mock_trainer)

        # Check metrics were computed
        assert isinstance(metrics, dict)
        assert len(metrics) > 0  # Should have depth metrics
        assert "AbsRel" in metrics or len(metrics) == 0  # Either has metrics or GT missing

        # Aggregate
        agg_metrics = evaluator.on_test_dataset_end(mock_trainer, temp_dir)
        assert agg_metrics is not None or len(evaluator.metrics_list) == 0


def test_pose_evaluator(mock_trainer, sample_batch, sample_predictions):
    """Test PoseEvaluator functionality."""
    evaluator = PoseEvaluator()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize
        evaluator.on_test_dataset_start(mock_trainer, "test_dataset")

        # Process batch
        metrics = evaluator.on_test_batch(sample_batch, sample_predictions, temp_dir, 0, mock_trainer)

        # Check metrics
        assert isinstance(metrics, dict)

        # Aggregate
        agg_metrics = evaluator.on_test_dataset_end(mock_trainer, temp_dir)
        assert agg_metrics is not None or len(evaluator.metrics_list) == 0


def test_intrinsics_evaluator(mock_trainer, sample_batch, sample_predictions):
    """Test IntrinsicsEvaluator functionality."""
    evaluator = IntrinsicsEvaluator()

    with tempfile.TemporaryDirectory() as temp_dir:
        evaluator.on_test_dataset_start(mock_trainer, "test_dataset")
        metrics = evaluator.on_test_batch(sample_batch, sample_predictions, temp_dir, 0, mock_trainer)
        assert isinstance(metrics, dict)
        agg_metrics = evaluator.on_test_dataset_end(mock_trainer, temp_dir)
        assert agg_metrics is not None or len(evaluator.metrics_list) == 0


def test_pointmap_evaluator(mock_trainer, sample_batch, sample_predictions):
    """Test PointmapEvaluator functionality."""
    evaluator = PointmapEvaluator()

    with tempfile.TemporaryDirectory() as temp_dir:
        evaluator.on_test_dataset_start(mock_trainer, "test_dataset")
        metrics = evaluator.on_test_batch(sample_batch, sample_predictions, temp_dir, 0, mock_trainer)
        assert isinstance(metrics, dict)
        agg_metrics = evaluator.on_test_dataset_end(mock_trainer, temp_dir)
        assert agg_metrics is not None or len(evaluator.metrics_list) == 0


def test_evaluator_preprocessing(mock_trainer, sample_batch, sample_predictions):
    """Test that evaluators handle preprocessing correctly."""
    evaluator = DepthEvaluator(preprocess_batch=True, scale_gt=True, align_predictions=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        evaluator.on_test_dataset_start(mock_trainer, "test_dataset")

        # Should not raise errors even with preprocessing enabled
        metrics = evaluator.on_test_batch(sample_batch, sample_predictions, temp_dir, 0, mock_trainer)

        assert isinstance(metrics, dict)


def test_trainer_parameter_usage(mock_trainer):
    """Test that callbacks can access trainer state."""

    class TestCallback(Callback):
        def on_test_batch(self, batch, predictions, output_dir, batch_idx, trainer, **kwargs):
            # Should be able to access trainer attributes
            assert hasattr(trainer, "accelerator")
            assert hasattr(trainer, "model")
            return {"test_metric": 1.0}

    callback = TestCallback()

    with tempfile.TemporaryDirectory() as temp_dir:
        metrics = callback.on_test_batch({}, {}, temp_dir, 0, mock_trainer)
        assert metrics == {"test_metric": 1.0}
