# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pytest

from dvlt.data.datasets.train import TrainDataset
from dvlt.data.datasets.util import rank_views_by_pose_similarity, rotate_array_90, rotate_frame_90


class MockTrainDataset(TrainDataset):
    """Concrete implementation of TrainDataset for testing purposes."""

    def __init__(self, *args, **kwargs):
        # Default to "index" view ranking to avoid pose-based requirements in basic tests
        if "view_ranking" not in kwargs:
            kwargs["view_ranking"] = "index"
        super().__init__(*args, **kwargs)
        self.data = []

    def __len__(self):
        return len(self.data)

    def get_data(self, seq_index=None, seq_name=None, img_per_seq=None, aspect_ratio=1.0):
        return {"test_key": "test_value"}


# Import centralized fixtures
pytest_plugins = ["tests.utils.fixtures"]


@pytest.fixture
def test_dataset():
    """Fixture providing a test dataset instance."""
    dataset = MockTrainDataset()
    dataset.set_image_params(img_size=512, patch_size=16)
    return dataset


def test_getitem(test_dataset):
    """Test the __getitem__ method."""
    with patch.object(test_dataset, "get_data", return_value={"test": "data"}) as mock_get_data:
        # Test with index only
        result = test_dataset[5]
        mock_get_data.assert_called_with(seq_index=5, data_fields=None, img_per_seq=None, aspect_ratio=1.0)
        assert result == {"test": "data"}

        # Test with tuple
        result = test_dataset[(10, 2, 1.5)]
        mock_get_data.assert_called_with(seq_index=10, data_fields=None, img_per_seq=2, aspect_ratio=1.5)
        assert result == {"test": "data"}


def test_get_target_hw(test_dataset):
    """Test the get_target_hw method."""
    # Test with aspect ratio 1.0
    hw = test_dataset.get_target_hw(1.0)
    assert hw[0] == 512
    assert hw[1] == 512

    # Test with aspect ratio 0.5
    hw = test_dataset.get_target_hw(0.5)
    assert hw[0] == 256
    assert hw[1] == 512

    # Test that output is divisible by patch_size
    test_dataset.patch_size = 32
    hw = test_dataset.get_target_hw(1.7)
    assert hw[0] % 32 == 0
    assert hw[1] % 32 == 0


def test_pick_neighbor_views_with_index_ranking():
    """Test the pick_neighbor_views method with index-based view ranking."""
    # Test with default window_size (256)
    dataset = MockTrainDataset(view_ranking="index")
    dataset.set_image_params(img_size=512, patch_size=16)
    ids = dataset.pick_neighbor_views(start_idx=50, total_ids=5, full_seq_num=100)
    assert len(ids) == 5
    assert ids[0] == 50

    # Test with explicit window_size_ratio
    dataset = MockTrainDataset(view_ranking="index", window_size_ratio=3.0)
    dataset.set_image_params(img_size=512, patch_size=16)
    ids = dataset.pick_neighbor_views(start_idx=50, total_ids=3, full_seq_num=100)
    assert len(ids) == 3
    assert ids[0] == 50

    # Test with explicit window_size
    dataset = MockTrainDataset(view_ranking="index", window_size=10)
    dataset.set_image_params(img_size=512, patch_size=16)
    ids = dataset.pick_neighbor_views(start_idx=50, total_ids=4, full_seq_num=100)
    assert len(ids) == 4
    assert ids[0] == 50
    for id_val in ids[1:]:
        assert 40 <= id_val <= 60

    # Test boundary conditions
    dataset = MockTrainDataset(view_ranking="index", window_size=10)
    dataset.set_image_params(img_size=512, patch_size=16)
    ids = dataset.pick_neighbor_views(start_idx=5, total_ids=3, full_seq_num=100)
    assert len(ids) == 3
    assert ids[0] == 5
    for id_val in ids[1:]:
        assert 0 <= id_val <= 15


def test_pick_neighbor_views_with_pose_ranking():
    """Test the pick_neighbor_views method with pose-based view ranking."""
    dataset = MockTrainDataset(view_ranking="pose", window_size=4)
    dataset.set_image_params(img_size=512, patch_size=16)

    # Create sample extrinsics for 5 views
    extrinsics = np.zeros((5, 4, 4))
    positions = np.array(
        [
            [0, 0, 0],
            [10, 10, 0],
            [1, 0, 0],
            [0, 1, 0],
            [5, 5, 0],
        ]
    )
    for i in range(5):
        extrinsics[i, :3, 3] = positions[i]  # Set position
        extrinsics[i, :3, 2] = [0, 0, 1]  # Set forward direction
        extrinsics[i, 3, 3] = 1  # Set homogeneous coordinate

    # Test with pose-based ranking - should require extrinsics_c2w
    with patch("numpy.random.choice", return_value=np.array([0, 3])):
        ids = dataset.pick_neighbor_views(
            start_idx=2,
            total_ids=3,
            full_seq_num=5,
            extrinsics_c2w=extrinsics,
        )

        assert len(ids) == 3
        assert ids[0] == 2  # Start index should be preserved


def test_view_sampling_random():
    """Test that random view sampling does not sort IDs."""
    dataset = MockTrainDataset(view_ranking="index", view_sampling="random", window_size=10)
    dataset.set_image_params(img_size=512, patch_size=16)

    with patch("numpy.random.choice", return_value=np.array([30, 10, 20])):
        ids = dataset.pick_neighbor_views(start_idx=15, total_ids=4, full_seq_num=100)

        assert len(ids) == 4
        assert ids[0] == 15  # Start index at beginning
        # The remaining IDs should be in the order returned by random.choice (not sorted)
        assert np.array_equal(ids[1:], [30, 10, 20])


def test_view_sampling_random_ordered():
    """Test that random_ordered view sampling sorts IDs."""
    dataset = MockTrainDataset(view_ranking="index", view_sampling="random_ordered", window_size=10)
    dataset.set_image_params(img_size=512, patch_size=16)

    with patch("numpy.random.choice", return_value=np.array([30, 10, 20])):
        ids = dataset.pick_neighbor_views(start_idx=15, total_ids=4, full_seq_num=100)

        assert len(ids) == 4
        # All IDs should be sorted
        expected_sorted = np.sort([15, 30, 10, 20])
        assert np.array_equal(ids, expected_sorted)


def test_invalid_view_ranking():
    """Test that invalid view ranking raises ValueError."""
    dataset = MockTrainDataset(view_ranking="invalid_method")
    dataset.set_image_params(img_size=512, patch_size=16)

    with pytest.raises(ValueError, match="Unknown view_ranking"):
        dataset.pick_neighbor_views(start_idx=50, total_ids=3, full_seq_num=100)


def test_invalid_view_sampling():
    """Test that invalid view sampling raises ValueError."""
    dataset = MockTrainDataset(view_ranking="index", view_sampling="invalid_method")
    dataset.set_image_params(img_size=512, patch_size=16)

    with pytest.raises(ValueError, match="Unknown view_sampling"):
        dataset.pick_neighbor_views(start_idx=50, total_ids=3, full_seq_num=100)


def test_rank_views_by_pose_similarity():
    """Test the rank_views_by_pose_similarity utility function."""
    # Create sample extrinsics for 5 views
    extrinsics = np.zeros((5, 4, 4))

    # Set positions and forward directions
    positions = np.array(
        [
            [0, 0, 0],  # Reference position
            [10, 10, 0],  # Farthest from reference
            [1, 0, 0],  # Close to reference
            [0, 1, 0],  # Close to reference
            [5, 5, 0],  # Far from reference
        ]
    )

    # Set forward directions (just using unit vectors for simplicity)
    forward = np.array(
        [
            [0, 0, 1],  # Reference direction
            [0, 0, 1],  # Same as reference
            [0, 0, 1],  # Same as reference
            [0, 0, 1],  # Same as reference
            [0, 0, 1],  # Same as reference
        ]
    )

    # Set the extrinsics matrices
    for i in range(5):
        extrinsics[i, :3, 3] = positions[i]  # Set position
        extrinsics[i, :3, 2] = forward[i]  # Set forward direction
        extrinsics[i, 3, 3] = 1  # Set homogeneous coordinate

    # Test ranking with reference view 0
    sorted_idxs = rank_views_by_pose_similarity(0, extrinsics)

    # The closest view should be the reference itself, then views at positions [1,0,0] and [0,1,0]
    assert sorted_idxs[0] == 0  # Reference view is closest to itself
    assert sorted_idxs[1] in [2, 3]  # Second closest should be either view 1 or 2
    assert sorted_idxs[2] in [2, 3]  # Third closest should be either view 1 or 2
    assert sorted_idxs[3] == 4  # Fourth closest should be view 3
    assert sorted_idxs[4] == 1  # Furthest should be view 4


@patch("dvlt.data.datasets.train.crop_around_principal_point")
@patch("dvlt.data.datasets.train.resize_with_overshoot")
@patch("dvlt.data.datasets.train.rotate_frame_90")
@patch("dvlt.data.datasets.train.depth_to_world_coords_points")
def test_prepare_view(
    mock_depth_to_points,
    mock_rotate,
    mock_resize,
    mock_crop,
    test_dataset,
    sample_image,
    sample_depth,
    sample_extrinsics,
    sample_intrinsics,
):
    """Test the prepare_view method."""
    original_hw = np.array([100, 200])
    target_hw = np.array([64, 128])

    # Configure mocks
    mock_crop.side_effect = lambda img, depth, intri, hw, filepath=None, strict=False: (img, depth, intri)
    mock_resize.side_effect = lambda img, depth, intri, target, original, safe_bound=None, scale_aug_factor=None: (
        img,
        depth,
        intri,
    )
    mock_rotate.side_effect = lambda img, depth, extri, intri, clockwise=None: (img, depth, extri, intri)
    mock_depth_to_points.return_value = (
        np.zeros((100, 200, 3)),
        np.zeros((100, 200, 3)),
        np.ones((100, 200), dtype=bool),
    )

    # Test normal processing flow
    with patch("numpy.random.rand", return_value=0.4):  # Below 0.5 to not trigger rotation
        result = test_dataset.prepare_view(
            sample_image, sample_depth, sample_extrinsics, sample_intrinsics, original_hw, target_hw
        )

    # Verify result structure
    assert len(result) == 7

    # Verify mock calls
    assert mock_crop.call_count >= 1
    assert mock_resize.call_count == 1
    assert mock_depth_to_points.call_count == 1

    # Reset mock call counts
    mock_crop.reset_mock()
    mock_resize.reset_mock()
    mock_rotate.reset_mock()
    mock_depth_to_points.reset_mock()

    # Test with landscape check and rotation
    test_dataset.allow_orientation_swap = True

    # Important: the first crop operation updates original_hw, so we need to mock
    # crop_around_principal_point to set the image shape to be portrait
    mock_crop.side_effect = lambda img, depth, intri, hw, filepath=None, strict=False: (
        np.ones((300, 100, 3)),  # portrait image: height > 1.2 * width
        np.ones((300, 100)),
        intri,
    )

    # Mock the random calls to ensure rotation is triggered
    # First ensure we have np.random.rand() > 0.5 for the rotation condition
    with patch("numpy.random.rand", side_effect=[0.6, 0.7]):
        result = test_dataset.prepare_view(
            sample_image,
            sample_depth,
            sample_extrinsics,
            sample_intrinsics,
            np.array([300, 100]),  # original portrait orientation
            target_hw,
        )

    # Verify that rotation was called
    assert mock_rotate.call_count >= 1


def test_rotate_frame_90(sample_image, sample_depth, sample_extrinsics, sample_intrinsics):
    """Test the complete rotate_frame_90 function."""
    # Clockwise rotation
    (rotated_image, rotated_depth, rotated_extrinsics, rotated_intrinsics) = rotate_frame_90(
        sample_image, sample_depth, sample_extrinsics, sample_intrinsics, clockwise=True
    )

    assert rotated_image.shape == (200, 100, 3)
    assert rotated_depth.shape == (200, 100)
    assert np.all(rotated_image == 128)
    assert np.array_equal(rotated_depth, rotate_array_90(sample_depth, clockwise=True))

    H, W = sample_image.shape[0], sample_image.shape[1]
    fx, fy, cx, cy = sample_intrinsics[0, 0], sample_intrinsics[1, 1], sample_intrinsics[0, 2], sample_intrinsics[1, 2]
    expected_intrinsics_cw = np.array([[fy, 0, H - cy], [0, fx, cx], [0, 0, 1]], dtype=sample_intrinsics.dtype)
    assert np.allclose(rotated_intrinsics, expected_intrinsics_cw)

    expected_R_cw = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=sample_extrinsics.dtype)
    assert np.allclose(rotated_extrinsics[:3, :3], expected_R_cw)
    assert np.allclose(rotated_extrinsics[:3, 3], np.zeros(3, dtype=sample_extrinsics.dtype))
    assert np.allclose(rotated_extrinsics[3], np.array([0, 0, 0, 1], dtype=sample_extrinsics.dtype))

    # Counter-clockwise rotation
    (rotated_image_ccw, rotated_depth_ccw, rotated_extrinsics_ccw, rotated_intrinsics_ccw) = rotate_frame_90(
        sample_image, sample_depth, sample_extrinsics, sample_intrinsics, clockwise=False
    )

    assert rotated_image_ccw.shape == (200, 100, 3)
    assert rotated_depth_ccw.shape == (200, 100)
    assert np.array_equal(rotated_depth_ccw, rotate_array_90(sample_depth, clockwise=False))

    expected_intrinsics_ccw = np.array([[fy, 0, cy], [0, fx, W - cx], [0, 0, 1]], dtype=sample_intrinsics.dtype)
    assert np.allclose(rotated_intrinsics_ccw, expected_intrinsics_ccw)

    expected_R_ccw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=sample_extrinsics.dtype)
    assert rotated_extrinsics_ccw.shape == (4, 4)
    assert np.allclose(rotated_extrinsics_ccw[:3, :3], expected_R_ccw)
    assert np.allclose(rotated_extrinsics_ccw[:3, 3], np.zeros(3, dtype=sample_extrinsics.dtype))
    assert np.allclose(rotated_extrinsics_ccw[3], np.array([0, 0, 0, 1], dtype=sample_extrinsics.dtype))

    # No-depth input
    (_, rotated_depth_none, _, _) = rotate_frame_90(
        sample_image, None, sample_extrinsics, sample_intrinsics, clockwise=True
    )
    assert rotated_depth_none is None


def test_prepare_view_with_actual_rotation(
    test_dataset, sample_image, sample_depth, sample_extrinsics, sample_intrinsics
):
    """Test the prepare_view method with actual rotation (not mocked)."""

    # Create a portrait image to trigger rotation
    portrait_image = np.ones((300, 200, 3), dtype=np.uint8) * 128
    portrait_depth = np.ones((300, 200), dtype=np.float32)

    # Create appropriate intrinsics for the portrait image
    portrait_intrinsics = np.array(
        [
            [150, 0, 100],  # fx, 0, cx (centered horizontally)
            [0, 150, 150],  # 0, fy, cy (centered vertically)
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    original_hw = np.array([300, 200])
    target_hw = np.array([64, 64])

    # Enable landscape check
    test_dataset.allow_orientation_swap = True

    # Mock depth_to_world_coords_points to avoid complex setup
    with patch("dvlt.data.datasets.train.depth_to_world_coords_points") as mock_depth_to_points:
        mock_depth_to_points.return_value = (
            np.zeros((64, 64, 3)),
            np.zeros((64, 64, 3)),
            np.ones((64, 64), dtype=bool),
        )

        # Mock random calls to ensure rotation is triggered
        with patch("numpy.random.rand", side_effect=[0.6, 0.7]):  # Both > 0.5 to trigger rotation
            result = test_dataset.prepare_view(
                portrait_image, portrait_depth, sample_extrinsics, portrait_intrinsics, original_hw, target_hw
            )

        # Verify result structure
        assert len(result) == 7
        (
            image,
            depth_map,
            extri,
            intri,
            world_coords,
            cam_coords,
            point_mask,
        ) = result

        # The image should be processed to target_hw
        assert image.shape[:2] == tuple(target_hw)
        if depth_map is not None:
            assert depth_map.shape == tuple(target_hw)

        # Verify that matrices have correct shapes
        assert extri.shape in ((3, 4), (4, 4))
        assert intri.shape == (3, 3)
