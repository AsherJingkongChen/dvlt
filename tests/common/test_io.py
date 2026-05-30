# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from dvlt.common.io import _load_16bit_png_depth, read_depth, read_image_cv2


@patch("os.path.exists")
@patch("os.path.getsize")
@patch("cv2.imread")
@patch("cv2.cvtColor")
def test_read_image_cv2_rgb(mock_cvtcolor, mock_imread, mock_getsize, mock_exists):
    # Set up mocks
    mock_exists.return_value = True
    mock_getsize.return_value = 100  # Non-empty file

    # Mock a color image
    mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img

    # Mock the cvtColor to return a transformed image
    transformed_img = np.ones((10, 10, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = transformed_img

    # Call the function
    result = read_image_cv2("test.jpg", rgb=True)

    # Verify the function behavior
    mock_exists.assert_called_once_with("test.jpg")
    mock_getsize.assert_called_once_with("test.jpg")
    mock_imread.assert_called_once_with("test.jpg")
    mock_cvtcolor.assert_called_once_with(mock_img, cv2.COLOR_BGR2RGB)

    # Verify the result
    assert np.array_equal(result, transformed_img)


@patch("os.path.exists")
@patch("os.path.getsize")
@patch("cv2.imread")
def test_read_image_cv2_non_rgb(mock_imread, mock_getsize, mock_exists):
    # Set up mocks
    mock_exists.return_value = True
    mock_getsize.return_value = 100  # Non-empty file

    # Mock a color image
    mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img

    # Call the function
    result = read_image_cv2("test.jpg", rgb=False)

    # Verify the function behavior
    mock_exists.assert_called_once_with("test.jpg")
    mock_getsize.assert_called_once_with("test.jpg")
    mock_imread.assert_called_once_with("test.jpg")

    # Verify the result
    assert np.array_equal(result, mock_img)


@patch("os.path.exists")
@patch("os.path.getsize")
@patch("cv2.imread")
@patch("cv2.cvtColor")
def test_read_image_cv2_grayscale(mock_cvtcolor, mock_imread, mock_getsize, mock_exists):
    # Set up mocks
    mock_exists.return_value = True
    mock_getsize.return_value = 100  # Non-empty file

    # Mock a grayscale image
    mock_img = np.zeros((10, 10), dtype=np.uint8)
    mock_imread.return_value = mock_img

    # Mock the cvtColor to return a transformed image
    transformed_img = np.ones((10, 10, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = transformed_img

    # Call the function
    result = read_image_cv2("test.jpg", rgb=True)

    # Verify the function behavior
    mock_cvtcolor.assert_called_once_with(mock_img, cv2.COLOR_GRAY2BGR)

    # Verify the result
    assert np.array_equal(result, transformed_img)


@patch("os.path.exists")
@patch("os.path.getsize")
def test_read_image_cv2_file_not_exist(mock_getsize, mock_exists):
    # Set up mocks
    mock_exists.return_value = False

    # Call the function
    result = read_image_cv2("test.jpg")

    # Verify the function behavior
    mock_exists.assert_called_once_with("test.jpg")
    mock_getsize.assert_not_called()

    # Verify the result
    assert result is None


@patch("cv2.imread")
def test_read_depth_exr(mock_imread):
    # Mock the imread function to return a fake depth map
    mock_depth = np.ones((10, 10, 3), dtype=np.float32)
    mock_imread.return_value = mock_depth

    # Call the function
    result = read_depth("test.exr")

    # Verify the function behavior
    mock_imread.assert_called_once_with("test.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Verify the result
    assert np.array_equal(result, mock_depth[..., 0])


@patch("dvlt.common.io._load_16bit_png_depth")
def test_read_depth_png(mock_load_16bit):
    # Mock the load_16bit_png_depth function
    mock_depth = np.ones((10, 10), dtype=np.float32)
    mock_load_16bit.return_value = mock_depth

    # Call the function
    result = read_depth("test.png")

    # Verify the function behavior
    mock_load_16bit.assert_called_once_with("test.png")

    # Verify the result
    assert np.array_equal(result, mock_depth)


def test_read_depth_invalid_extension():
    # Call the function with an invalid extension
    with pytest.raises(ValueError):
        read_depth("test.jpg")


@patch("PIL.Image.open")
def test_load_16bit_png_depth(mock_open):
    # Mock the PIL.Image.open context manager
    mock_img = MagicMock()
    mock_img.size = (10, 10)
    mock_open.return_value.__enter__.return_value = mock_img

    # Mock the np.array call inside the function
    mock_array = np.zeros((10, 10), dtype=np.uint16)
    mock_img.__array__ = lambda dtype, copy=None: mock_array

    # Create a patch for np.frombuffer
    with patch("numpy.frombuffer") as mock_frombuffer:
        # Mock the return value of frombuffer
        mock_frombuffer.return_value = np.ones(100, dtype=np.float16)

        # Call the function
        result = _load_16bit_png_depth("test.png")

        # Verify the function behavior
        mock_open.assert_called_once_with("test.png")
        mock_frombuffer.assert_called_once()

        # Verify the result shape
        assert result.shape == (10, 10)
        assert result.dtype == np.float32
