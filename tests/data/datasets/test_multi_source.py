# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pytest

from dvlt.common.constants import DataField
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.train import TrainDataset


class MockDataset:
    """Minimal dataset that exposes get_data/available_data_fields for validation tests."""

    def __init__(self, batch_data):
        self.batch_data = batch_data

    def __len__(self):
        return 1

    def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
        return self.batch_data

    def available_data_fields(self):
        # Claim default fields are available even if missing in batch, so runtime validation triggers
        return {
            DataField.IMAGES,
            DataField.EXTRINSICS_C2W,
            DataField.INTRINSICS,
            DataField.DEPTHS,
            DataField.WORLD_POINTS,
            DataField.POINT_MASKS,
        } | set(self.batch_data.keys())


class DummyDataset(TrainDataset):
    """Minimal TrainDataset that yields a tiny well-formed batch."""

    def __len__(self):
        return 1

    def available_data_fields(self):
        return {
            DataField.SEQ_NAME,
            DataField.IDS,
            DataField.IMAGES,
            DataField.DEPTHS,
            DataField.EXTRINSICS_C2W,
            DataField.INTRINSICS,
            DataField.WORLD_POINTS,
            DataField.POINT_MASKS,
            DataField.SCALE_FACTOR,
        }

    def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
        S, H, W = 2, 4, 4
        sample = {
            DataField.SEQ_NAME: "dummy",
            DataField.IDS: np.array([0, 1]),
            DataField.IMAGES: [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(S)],
            DataField.DEPTHS: [np.zeros((H, W), dtype=np.float32) for _ in range(S)],
            DataField.EXTRINSICS_C2W: [np.eye(4, dtype=np.float32) for _ in range(S)],
            DataField.INTRINSICS: [np.eye(3, dtype=np.float32) for _ in range(S)],
            DataField.WORLD_POINTS: [np.zeros((H, W, 3), dtype=np.float32) for _ in range(S)],
            DataField.POINT_MASKS: [np.ones((H, W), dtype=bool) for _ in range(S)],
            DataField.SCALE_FACTOR: 1.0,
        }
        return sample


def test_valid_minimal_batch():
    """Test that a valid minimal batch passes validation."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ],
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES])

    # Should not raise any validation errors
    sample = composed[0]
    assert sample[DataField.SEQ_NAME] == "test_seq"
    assert sample[DataField.IMAGES].shape == (2, 3, 100, 100)  # [S, C, H, W]


def test_missing_required_field():
    """Test that missing required fields raise validation errors."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        # Missing IMAGES field
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES])

    with pytest.raises(ValueError, match="Failed to get a valid sample"):
        composed[0]


def test_wrong_dtype_validation():
    """Test that wrong dtypes are caught by validation."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ],
        DataField.DEPTHS: [np.random.random((100, 100)).astype(np.float64)],  # Wrong dtype, should be float32
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES, DataField.DEPTHS])

    with pytest.raises(ValueError, match="Failed to get a valid sample"):
        composed[0]


def test_required_field_dependency():
    """Test that field dependencies are enforced."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ],
        DataField.EXTRINSICS_C2W: [np.eye(4, dtype=np.float32)],
        DataField.WORLD_POINTS: [np.random.random((100, 100, 3)).astype(np.float32)],
        # Missing POINT_MASKS - should be required when extrinsics + world_points present
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(
        mock_dataset,
        load_data_fields=[DataField.IMAGES, DataField.EXTRINSICS_C2W, DataField.WORLD_POINTS],
    )

    with pytest.raises(ValueError, match="Failed to get a valid sample"):
        composed[0]


def test_image_range_validation():
    """Test that image validation works correctly."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),  # Valid image
        ],
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES])

    # This should pass with valid uint8 images
    sample = composed[0]
    assert sample[DataField.SEQ_NAME] == "test_seq"


def test_image_wrong_dtype():
    """Test that images with wrong dtype are caught."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.int32),  # Wrong dtype, should be uint8
        ],
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES])

    with pytest.raises(ValueError, match="Failed to get a valid sample"):
        composed[0]


def test_shape_validation():
    """Test that wrong shapes are caught by validation."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ],
        DataField.DEPTHS: [
            np.random.random((100, 100)).astype(np.float32),
            np.random.random((50, 50)).astype(np.float32),  # Wrong height/width
        ],
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES, DataField.DEPTHS])

    with pytest.raises(ValueError, match="Failed to get a valid sample"):
        composed[0]


def test_list_length_validation():
    """Test that wrong list lengths are caught."""
    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ],
        DataField.DEPTHS: [
            np.random.random((100, 100)).astype(np.float32),
            # Missing second depth array - should have S=2 elements
        ],
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES, DataField.DEPTHS])

    with pytest.raises(ValueError, match="Failed to get a valid sample"):
        composed[0]


def test_image_value_range_validation():
    """Test that images with invalid values are caught."""
    # Create an image with out-of-range values
    invalid_image = np.array([[[300, 150, 100]]], dtype=np.float32)  # Values > 255

    batch = {
        DataField.SEQ_NAME: "test_seq",
        DataField.IMAGES: [invalid_image.astype(np.uint8)],  # Convert to uint8 (will wrap values)
    }

    mock_dataset = MockDataset(batch)
    composed = MultiSourceDataset(mock_dataset, load_data_fields=[DataField.IMAGES])

    # This should pass because uint8 values are always valid (0-255) after wrapping
    sample = composed[0]
    assert sample[DataField.SEQ_NAME] == "test_seq"


def test_training_vs_eval_mode():
    """Test critical differences between training and evaluation modes."""
    base_ds = DummyDataset()

    # Training mode
    train_ds = MultiSourceDataset(base_ds, training=True, load_data_fields=None)
    eval_ds = MultiSourceDataset(base_ds, training=False, load_data_fields=None)

    # Length should be padded in training mode
    assert len(train_ds) == MultiSourceDataset.MIN_LEN  # 1000
    assert len(eval_ds) == 1  # original length

    # Augmentation should only be set up in training mode
    assert train_ds.image_aug is not None
    assert eval_ds.image_aug is None


def test_index_resolution_training():
    """Training mode should rescale padded indices to the underlying dataset size and honor tuple args."""

    # Create a dataset that triggers padding and records the seq_index used
    class TinyDataset(TrainDataset):
        def __init__(self):
            super().__init__()
            self.last_seq_index = None

        def __len__(self):
            return 10

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            self.last_seq_index = seq_index
            return DummyDataset().get_data(seq_index, data_fields, img_per_seq, aspect_ratio)

    tiny_ds = TinyDataset()
    comp_ds_padded = MultiSourceDataset(tiny_ds, training=True, load_data_fields=None, enable_color_jitter=False)

    # Length should be padded
    assert len(comp_ds_padded) == MultiSourceDataset.MIN_LEN

    # Access a high padded index and ensure it is mapped into [0, len(tiny_ds))
    _ = comp_ds_padded[(999, 2, 1.0)]
    assert 0 <= tiny_ds.last_seq_index < 10


def test_index_resolution_eval():
    """Evaluation mode should not use padding and ignores tuple payload beyond seq_index."""

    class SpyDataset(TrainDataset):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __len__(self):
            return 3

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            self.calls.append((seq_index, img_per_seq, aspect_ratio))
            return DummyDataset().get_data(seq_index, data_fields, img_per_seq, aspect_ratio)

    spy = SpyDataset()
    comp_ds = MultiSourceDataset(spy, training=False, load_data_fields=None)

    _ = comp_ds[0]
    _ = comp_ds[(0, 2, 1.0)]

    # First call from int index
    assert spy.calls[0] == (0, None, 1.0)
    # Second call should ignore img_per_seq/aspect_ratio in eval mode
    assert spy.calls[1] == (0, None, 1.0)


def test_retry_logic():
    """Test that retry logic works correctly in training mode."""
    base_ds = DummyDataset()

    # Mock check_all_finite to fail first few times
    with patch("dvlt.data.datasets.multi_source.check_all_finite") as mock_check:
        mock_check.side_effect = [False, False, True]  # fail twice, then succeed

        comp_ds = MultiSourceDataset(base_ds, training=True, load_data_fields=None, enable_color_jitter=False)
        sample = comp_ds[(0, 2, 1.0)]  # Should succeed after retries

        assert mock_check.call_count == 3
        assert sample is not None

    # Test max retries exceeded
    with patch("dvlt.data.datasets.multi_source.check_all_finite") as mock_check:
        mock_check.return_value = False  # always fail

        comp_ds = MultiSourceDataset(base_ds, training=True, load_data_fields=None, enable_color_jitter=False)

        try:
            comp_ds[(0, 2, 1.0)]
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "Failed to get a valid sample" in str(e)


def test_retry_resamples_index_and_preserves_tuple_args():
    """On retry, seq_index is resampled (may or may not be different) and tuple payload should be preserved."""

    class MultiSeqDataset(TrainDataset):
        def __init__(self):
            super().__init__()
            self.calls = []  # record (seq_index, img_per_seq, aspect_ratio)

        def __len__(self):
            return 100

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            # record the call for later assertions
            self.calls.append((seq_index, img_per_seq, aspect_ratio))
            return DummyDataset().get_data(seq_index, data_fields, img_per_seq, aspect_ratio)

    base_ds = MultiSeqDataset()

    # Run multiple trials to test probabilistic behavior
    different_count = 0
    total_trials = 10

    for _ in range(total_trials):
        base_ds.calls.clear()

        # Fail once, then succeed so we have at least two get_data calls
        with patch("dvlt.data.datasets.multi_source.check_all_finite") as mock_check:
            mock_check.side_effect = [False, True]

            comp_ds = MultiSourceDataset(base_ds, training=True, load_data_fields=None, enable_color_jitter=False)
            _ = comp_ds[(0, 2, 1.0)]

            # We should have made at least two get_data calls
            assert len(base_ds.calls) >= 2
            first_seq, first_img_per_seq, first_aspect = base_ds.calls[0]
            second_seq, second_img_per_seq, second_aspect = base_ds.calls[1]

            # Count cases where seq_index differs (most should be different due to sampling with replacement)
            if first_seq != second_seq:
                different_count += 1

            # tuple payload should always be preserved across retries
            assert first_img_per_seq == 2 and first_aspect == 1.0
            assert second_img_per_seq == 2 and second_aspect == 1.0

    # With 100 sequences, we expect most retries to use different indices
    # Allow some tolerance for the probabilistic nature
    assert (
        different_count >= total_trials * 0.7
    ), f"Expected most retries to use different seq_index, got {different_count}/{total_trials}"


def test_multi_dataset_index_mapping_and_boundaries():
    """Indices should map correctly across concatenated datasets and boundaries."""

    class DsA(TrainDataset):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __len__(self):
            return 2

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            self.calls.append(seq_index)
            return DummyDataset().get_data(seq_index, data_fields, img_per_seq, aspect_ratio)

    class DsB(TrainDataset):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __len__(self):
            return 3

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            self.calls.append(seq_index)
            return DummyDataset().get_data(seq_index, data_fields, img_per_seq, aspect_ratio)

    a, b = DsA(), DsB()
    comp = MultiSourceDataset({"A": a, "B": b}, training=False, load_data_fields=None)

    _ = comp[0]  # A[0]
    _ = comp[1]  # A[1]
    _ = comp[2]  # B[0]
    _ = comp[4]  # B[2]

    assert a.calls == [0, 1]
    assert b.calls == [0, 2]  # only indices 0 and 2 called in B


def test_point_masks_invalidation_when_first_frame_empty():
    """If first frame has no valid points, all frames' masks should be cleared in training."""

    class MaskDataset(TrainDataset):
        def __len__(self):
            return 1

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            S, H, W = 2, 4, 4
            return {
                DataField.SEQ_NAME: "masks",
                DataField.IDS: np.array([0, 1]),
                DataField.IMAGES: [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(S)],
                DataField.DEPTHS: [np.zeros((H, W), dtype=np.float32) for _ in range(S)],
                DataField.EXTRINSICS_C2W: [np.eye(4, dtype=np.float32) for _ in range(S)],
                DataField.INTRINSICS: [np.eye(3, dtype=np.float32) for _ in range(S)],
                DataField.WORLD_POINTS: [np.zeros((H, W, 3), dtype=np.float32) for _ in range(S)],
                DataField.POINT_MASKS: [np.zeros((H, W), dtype=bool), np.ones((H, W), dtype=bool)],
            }

    ds = MaskDataset()
    comp = MultiSourceDataset(ds, training=True, load_data_fields=None, enable_color_jitter=False)
    sample = comp[0]
    assert sample[DataField.POINT_MASKS].sum() == 0


def test_pose_based_scale_fallback_when_no_valid_points():
    """When all point masks are False, scale should fall back to mean relative translation magnitude."""

    class FallbackScaleDataset(TrainDataset):
        def __len__(self):
            return 1

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            # Use multiple frames to ensure the fallback aggregates over many translations
            S, H, W = 5, 4, 4
            extrinsics = []
            for i in range(S):
                extr = np.eye(4, dtype=np.float32)
                extr[0, 3] = 2.0 * i  # translate along +x by 2*i meters
                extrinsics.append(extr)
            return {
                DataField.SEQ_NAME: "fallback_scale",
                DataField.IDS: np.arange(S, dtype=np.int64),
                DataField.IMAGES: [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(S)],
                DataField.DEPTHS: [np.zeros((H, W), dtype=np.float32) for _ in range(S)],
                DataField.EXTRINSICS_C2W: extrinsics,
                DataField.INTRINSICS: [np.eye(3, dtype=np.float32) for _ in range(S)],
                DataField.WORLD_POINTS: [np.zeros((H, W, 3), dtype=np.float32) for _ in range(S)],
                # All masks are False so the fallback (pose-based) path is taken
                DataField.POINT_MASKS: [np.zeros((H, W), dtype=bool) for _ in range(S)],
            }

    ds = FallbackScaleDataset()
    comp = MultiSourceDataset(ds, training=False, load_data_fields=None)
    sample = comp[0]
    # Expect mean of translations [2, 4, 6, 8] = 5.0 for S=5
    expected = float(np.mean([2.0 * i for i in range(1, 5)]))
    sf = sample[DataField.SCALE_FACTOR]
    assert sf.numel() == 1, f"scale_factor should be scalar-like, got shape {tuple(sf.shape)}"
    assert abs(float(sf.item()) - expected) < 1e-6


def test_exception_handling_retries_and_logs(caplog):
    """Dataset exceptions should be caught, logged, and retried until success."""

    class FlakyDataset(TrainDataset):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __len__(self):
            return 5

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("simulated failure")
            return DummyDataset().get_data(seq_index, data_fields, img_per_seq, aspect_ratio)

    flaky = FlakyDataset()
    comp = MultiSourceDataset({"DS": flaky}, training=True, load_data_fields=None, enable_color_jitter=False)

    import logging

    with caplog.at_level(logging.ERROR, logger="dvlt.data.datasets.multi_source"):
        sample = comp[(0, 2, 1.0)]

    # Should have succeeded after retries
    assert sample is not None
    assert flaky.calls >= 3

    # Logs should contain our error with dataset label and request payload
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Data loading error" in m for m in messages)
    assert any("DS" in m for m in messages)
    assert any("request=" in m for m in messages)


def test_exception_max_retries_raises(caplog):
    """If all retries fail due to exceptions, a ValueError should be raised."""

    class AlwaysFail(TrainDataset):
        def __len__(self):
            return 3

        def available_data_fields(self):
            return DummyDataset().available_data_fields()

        def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
            raise IOError("persistent failure")

    ds = AlwaysFail()
    comp = MultiSourceDataset({"AF": ds}, training=True, load_data_fields=None, enable_color_jitter=False)

    import logging

    with caplog.at_level(logging.ERROR, logger="dvlt.data.datasets.multi_source"):
        with pytest.raises(ValueError, match="Failed to get a valid sample"):
            _ = comp[(0, 2, 1.0)]

    # There should be at least one logged error message
    assert any("Data loading error" in rec.getMessage() for rec in caplog.records)
