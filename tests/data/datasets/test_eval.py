# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from dvlt.data.datasets.eval import EvalDataset


class MockEvalDataset(EvalDataset):
    """Concrete implementation of EvalDataset for testing purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return {"test_key": "test_value"}


# Import centralized fixtures
pytest_plugins = ["tests.utils.fixtures"]


@pytest.fixture
def test_dataset():
    """Fixture providing a test eval dataset instance."""
    dataset = MockEvalDataset(mode="crop")
    dataset.set_image_params(img_size=518, patch_size=14)
    return dataset


@pytest.fixture
def test_dataset_pad():
    """Fixture providing a test eval dataset instance with pad mode."""
    dataset = MockEvalDataset(mode="pad")
    dataset.set_image_params(img_size=518, patch_size=14)
    return dataset


@pytest.fixture
def sample_images():
    """Fixture providing sample images as numpy arrays."""
    return [
        np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
        np.random.randint(0, 255, (150, 180, 3), dtype=np.uint8),
    ]


@pytest.fixture
def sample_depths():
    """Fixture providing sample depth maps."""
    return [
        np.random.rand(100, 200).astype(np.float32),
        np.random.rand(150, 180).astype(np.float32),
    ]


@pytest.fixture
def sample_intrinsics_list():
    """Fixture providing sample intrinsic matrices."""
    return [
        np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]], dtype=np.float32),
        np.array([[120, 0, 90], [0, 120, 75], [0, 0, 1]], dtype=np.float32),
    ]


def test_init(test_dataset):
    """Test the initialization of EvalDataset."""
    assert test_dataset.img_size == 518
    assert test_dataset.patch_size == 14
    assert test_dataset.mode == "crop"
    assert test_dataset.max_frames is None
    assert test_dataset.max_depth_thresh is None


def test_init_with_parameters():
    """Test initialization with all parameters."""
    dataset = MockEvalDataset(
        max_frames=10,
        mode="pad",
        max_depth_thresh=50.0,
    )
    dataset.set_image_params(img_size=256, patch_size=16)
    assert dataset.max_frames == 10
    assert dataset.img_size == 256
    assert dataset.patch_size == 16
    assert dataset.mode == "pad"
    assert dataset.max_depth_thresh == 50.0


def test_preprocess_images_only(test_dataset, sample_images):
    """Test preprocessing with images only."""
    images, depths, intrinsics = test_dataset._preprocess_images_depths_intrinsics(sample_images)

    assert isinstance(images, list)
    assert depths is None
    assert intrinsics is None

    first_img = images[0]
    assert isinstance(first_img, np.ndarray)
    assert first_img.ndim == 3  # H, W, C
    assert first_img.shape[2] == 3  # RGB channels
    assert first_img.dtype == np.uint8


def test_preprocess_all_inputs(test_dataset, sample_images, sample_depths, sample_intrinsics_list):
    """Test preprocessing with all inputs of different sizes and data types."""
    images, depths, intrinsics = test_dataset._preprocess_images_depths_intrinsics(
        sample_images, depths=sample_depths, intrinsics=sample_intrinsics_list
    )

    assert isinstance(images, list)
    assert isinstance(depths, list)
    assert isinstance(intrinsics, list)

    assert isinstance(depths[0], np.ndarray)
    assert depths[0].dtype == np.float32
    assert depths[0].ndim == 2  # H, W

    first_K = intrinsics[0]
    assert isinstance(first_K, np.ndarray)
    assert first_K.shape == (3, 3)

    n_images = len(sample_images)
    assert len(images) == n_images
    assert len(depths) == n_images
    assert len(intrinsics) == n_images

    for img, depth_map in zip(images, depths, strict=False):
        assert img.shape[0] == depth_map.shape[0]
        assert img.shape[1] == depth_map.shape[1]


def test_preprocess_pad_mode(test_dataset_pad, sample_images):
    """Test preprocessing in pad mode."""
    images, depths, intrinsics = test_dataset_pad._preprocess_images_depths_intrinsics(sample_images)

    # Check that each image is padded to square
    for img in images:
        assert img.shape[0] == img.shape[1] == test_dataset_pad.img_size
        # Depths should be None in this test (only images provided)


def test_preprocess_empty_images(test_dataset):
    """Test preprocessing with empty image list."""
    with pytest.raises(ValueError, match="At least 1 image is required"):
        test_dataset._preprocess_images_depths_intrinsics([])


def test_preprocess_invalid_mode():
    """Test initialization with invalid mode."""
    with pytest.raises(ValueError, match="Mode must be either 'crop' or 'pad'"):
        dataset = MockEvalDataset(mode="invalid")
        dataset._preprocess_images_depths_intrinsics([np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)])


def test_preprocess_rgba_image(test_dataset):
    """Test preprocessing with RGBA image."""
    # Create RGBA image with transparency
    rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    rgba_image[:, :, 3] = 128  # Semi-transparent

    images, _, _ = test_dataset._preprocess_images_depths_intrinsics([rgba_image])

    # Should be converted to RGB and processed normally
    assert isinstance(images, list)
    assert images[0].shape[2] == 3


def test_preprocess_max_depth_thresh():
    """Test preprocessing with max depth threshold."""
    dataset_with_thresh = MockEvalDataset(max_depth_thresh=1.0)
    dataset_with_thresh.set_image_params(img_size=518, patch_size=14)
    dataset_without_thresh = MockEvalDataset(max_depth_thresh=None)
    dataset_without_thresh.set_image_params(img_size=518, patch_size=14)

    # Create a test case with mixed depth values
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Create depth with values both above and below threshold
    test_depth = np.random.uniform(0.5, 1.5, (100, 100)).astype(np.float32)
    test_depth[30:70, 30:70] = 2.0  # Large region above threshold

    # Process with threshold
    _, depths_with_thresh, _ = dataset_with_thresh._preprocess_images_depths_intrinsics(
        [test_image], depths=[test_depth.copy()]
    )

    # Process without threshold
    _, depths_without_thresh, _ = dataset_without_thresh._preprocess_images_depths_intrinsics(
        [test_image], depths=[test_depth.copy()]
    )

    # The thresholded version should have much lower mean depth
    # (values above 1.0 should be set to 0 before resizing)
    assert depths_with_thresh[0].mean() < depths_without_thresh[0].mean()

    # Additionally, test that all values above threshold are actually set to 0 before processing
    test_depth_high = np.full((50, 50), 2.0, dtype=np.float32)  # All above threshold
    _, depths_all_high, _ = dataset_with_thresh._preprocess_images_depths_intrinsics(
        [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)], depths=[test_depth_high]
    )

    # Should be very close to 0 after thresholding (allowing for small interpolation artifacts)
    assert depths_all_high[0].max() < 0.1


def test_preprocess_single_image(test_dataset):
    """Test preprocessing with single image to ensure correct shapes."""
    single_image = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]

    images, _, _ = test_dataset._preprocess_images_depths_intrinsics(single_image)

    # Should return a single image
    assert len(images) == 1
    assert images[0].ndim == 3  # H, W, C


def test_get_indices():
    """Test the _get_indices method with concrete examples to verify ranking and sampling functionality."""

    # Test 1: Index ranking + First sampling
    dataset_index_first = MockEvalDataset(view_ranking="index", view_sampling="first")

    # Without max_frames - should return all indices in order
    indices = dataset_index_first._get_indices(10)
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # With max_frames - should return first N indices
    indices = dataset_index_first._get_indices(10, max_frames=5)
    assert indices == [0, 1, 2, 3, 4]

    # Test 2: Index ranking + Uniform sampling
    dataset_index_uniform = MockEvalDataset(view_ranking="index", view_sampling="uniform")

    # Uniform sampling from 10 frames, taking 5
    indices = dataset_index_uniform._get_indices(10, max_frames=5)
    expected = [0, 2, 4, 6, 8]  # np.linspace(0, 10, 5, endpoint=False, dtype=int)
    assert indices == expected

    # Uniform sampling from 12 frames, taking 4
    indices = dataset_index_uniform._get_indices(12, max_frames=4)
    expected = [0, 3, 6, 9]  # np.linspace(0, 12, 4, endpoint=False, dtype=int)
    assert indices == expected

    # Test 3: Pose ranking + First sampling
    dataset_pose_first = MockEvalDataset(view_ranking="pose", view_sampling="first")

    # Create predictable extrinsics accounting for the normalization step in pose similarity:
    # The function normalizes positions by avg_scale = mean(||pos||) before computing distances
    # Distance = r_dists + lambda_t * t_dists
    # where r_dists = rotation_angle_deg / 180.0, t_dists = ||normalized_pos - normalized_pos[0]||

    extrinsics = np.zeros((4, 4, 4))

    # Frame 0: reference frame at origin, identity rotation (distance = 0)
    extrinsics[0] = np.eye(4)

    # Frame 1: translation that will be affected by normalization
    extrinsics[1] = np.eye(4)
    extrinsics[1, :3, 3] = [1.0, 0, 0]  # Will be normalized

    # Frame 2: larger translation that will be heavily penalized after normalization
    extrinsics[2] = np.eye(4)
    extrinsics[2, :3, 3] = [5.0, 0, 0]  # Will be heavily penalized

    # Frame 3: 90-degree rotation only (no translation)
    extrinsics[3] = np.eye(4)
    extrinsics[3, :3, 3] = [0, 0, 0]  # no translation
    # 90-degree rotation around Y-axis: r_dist = 90/180 = 0.5
    extrinsics[3, 0, 0] = 0  # cos(90°) = 0
    extrinsics[3, 0, 2] = 1  # sin(90°) = 1
    extrinsics[3, 2, 0] = -1  # -sin(90°) = -1
    extrinsics[3, 2, 2] = 0  # cos(90°) = 0

    indices = dataset_pose_first._get_indices(4, extrinsics_c2w=extrinsics)
    assert len(indices) == 4
    assert indices[0] == 0  # Reference frame always first
    assert indices[1] == 3  # Frame 3 (pure rotation) should be most similar after normalization
    assert set(indices) == {0, 1, 2, 3}  # All frames present

    # Test first sampling with max_frames
    indices = dataset_pose_first._get_indices(4, max_frames=2, extrinsics_c2w=extrinsics)
    assert len(indices) == 2
    assert indices == [0, 3]  # Should get reference frame and most similar frame (rotation only)

    # Test 4: Pose ranking + Uniform sampling
    dataset_pose_uniform = MockEvalDataset(view_ranking="pose", view_sampling="uniform")

    # Create 6 frames with simple, predictable ranking even with normalization
    # Frame 0: reference at origin
    # Frames 1-5: small equal translations that maintain order after normalization
    extrinsics_large = np.zeros((6, 4, 4))
    for i in range(6):
        extrinsics_large[i] = np.eye(4)
        # Small, equal increments so normalization doesn't change relative order
        extrinsics_large[i, :3, 3] = [i * 0.1, 0, 0]

    # After normalization, the relative ordering remains [0, 1, 2, 3, 4, 5]
    # because all are just scaled by the same factor
    # Test uniform sampling: from 6 frames, sample 3
    # np.linspace(0, 6, 3, endpoint=False) = [0, 2, 4]
    indices = dataset_pose_uniform._get_indices(6, max_frames=3, extrinsics_c2w=extrinsics_large)
    assert len(indices) == 3
    # After pose ranking [0,1,2,3,4,5], uniform sampling should pick [0,2,4]
    assert indices == [0, 2, 4]

    # Test 5: Edge cases
    # Max frames greater than video length
    indices = dataset_index_first._get_indices(3, max_frames=10)
    assert len(indices) == 3
    assert indices == [0, 1, 2]

    # Single frame
    indices = dataset_index_first._get_indices(1)
    assert indices == [0]

    # Test 6: Error cases
    # Pose ranking without extrinsics
    with pytest.raises(AssertionError, match="Extrinsics are required for pose ranking"):
        dataset_pose_first._get_indices(5)

    # Invalid view ranking
    invalid_ranking = MockEvalDataset(view_ranking="invalid")
    with pytest.raises(ValueError, match="Invalid view ranking method: invalid"):
        invalid_ranking._get_indices(5)

    # Invalid view sampling
    invalid_sampling = MockEvalDataset(view_ranking="index", view_sampling="invalid")
    with pytest.raises(ValueError, match="Invalid view sampling method: invalid"):
        invalid_sampling._get_indices(10, max_frames=5)


def test_preprocess_intrinsics_scaling(test_dataset, sample_images, sample_intrinsics_list):
    """Test that intrinsics are properly scaled during preprocessing."""
    original_intrinsics = sample_intrinsics_list[0].copy()

    images, _, intrinsics = test_dataset._preprocess_images_depths_intrinsics(
        sample_images[:1], intrinsics=sample_intrinsics_list[:1]
    )

    # Check that intrinsics were modified (scaled)
    assert not np.allclose(intrinsics[0], original_intrinsics)

    # The scaling should maintain the structure but adjust focal lengths and principal points
    assert intrinsics[0][2, 2] == 1.0
    assert intrinsics[0][0, 1] == 0.0
    assert intrinsics[0][1, 0] == 0.0


def test_preprocess_different_image_shapes():
    """Test preprocessing with images of various sizes and aspect ratios."""
    dataset = MockEvalDataset(mode="crop")
    dataset.set_image_params(img_size=200, patch_size=10)

    # Create images with very different aspect ratios and sizes
    images = [
        np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8),  # Wide
        np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),  # Tall
        np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8),  # Small
        np.random.randint(0, 255, (150, 80, 3), dtype=np.uint8),  # Tall and narrow
    ]

    processed_images, _, _ = dataset._preprocess_images_depths_intrinsics(images)

    # All processed images should have the same shape
    assert all(img.shape == processed_images[0].shape for img in processed_images)

    # Width should be the target size
    assert processed_images[0].shape[1] == dataset.img_size

    # Height should be consistent across all images (padded to max) and divisible by patch_size
    assert processed_images[0].shape[0] % dataset.patch_size == 0


def test_preprocess_crop_mode_large_height(test_dataset):
    """Test crop mode when height exceeds target size."""
    # Create image with height much larger than img_size
    large_image = [np.random.randint(0, 255, (1000, 518, 3), dtype=np.uint8)]

    images, _, _ = test_dataset._preprocess_images_depths_intrinsics(large_image)

    # Height should be cropped to img_size
    assert images[0].shape[0] == test_dataset.img_size


def test_abstract_methods_not_implemented():
    """Test that abstract methods are not implemented in base class."""
    dataset = EvalDataset()
    dataset.set_image_params(img_size=518, patch_size=14)

    with pytest.raises(NotImplementedError):
        dataset.get_data(0)

    with pytest.raises(NotImplementedError):
        dataset.__len__()


def test_preprocess_modes_consistency():
    """Test preprocessing in pad/crop modes produces consistent output shapes."""
    pad_dataset = MockEvalDataset(mode="pad")
    pad_dataset.set_image_params(img_size=256, patch_size=14)
    crop_dataset = MockEvalDataset(mode="crop")
    crop_dataset.set_image_params(img_size=256, patch_size=14)

    different_images = [
        np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
        np.random.randint(0, 255, (150, 180, 3), dtype=np.uint8),
    ]

    # Pad mode: outputs should be square at img_size.
    pad_images, _, _ = pad_dataset._preprocess_images_depths_intrinsics(different_images)
    assert pad_images[0].shape[0] == pad_images[0].shape[1] == pad_dataset.img_size

    # Crop mode: width should be img_size, all heights should match.
    crop_images, _, _ = crop_dataset._preprocess_images_depths_intrinsics(different_images)
    assert crop_images[0].shape[1] == crop_dataset.img_size
    assert all(img.shape == crop_images[0].shape for img in crop_images)
