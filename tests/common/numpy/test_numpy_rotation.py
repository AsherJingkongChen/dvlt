# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.testing as npt

from dvlt.common.numpy.rotation import mat_to_quat, quat_to_mat, standardize_quaternion


def test_quat_to_mat():
    # Test identity quaternion [0, 0, 0, 1] (w-last format)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    mat = quat_to_mat(quat)
    expected = np.eye(3)
    npt.assert_allclose(mat, expected)

    # Test 90-degree rotation around X axis: [1, 0, 0, 0] (w-last format)
    quat = np.array([1.0, 0.0, 0.0, 0.0]) / np.sqrt(1.0)
    mat = quat_to_mat(quat)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    npt.assert_allclose(mat, expected, rtol=1e-5, atol=1e-5)

    # Test batch of quaternions
    quats = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])  # Identity  # 90-degree X rotation
    mats = quat_to_mat(quats)
    expected = np.array([np.eye(3), np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])])
    npt.assert_allclose(mats, expected, rtol=1e-5, atol=1e-5)


def test_mat_to_quat():
    # Test identity matrix
    mat = np.eye(3)
    quat = mat_to_quat(mat)
    expected = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    npt.assert_allclose(quat, expected)

    # Test 90-degree rotation around X axis
    mat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    quat = mat_to_quat(mat)
    # For 90-degree X-axis rotation, we expect [±0.7071, 0, 0, 0.7071]
    expected = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    npt.assert_allclose(np.abs(quat), np.abs(expected), rtol=1e-5, atol=1e-5)

    # Test batch of matrices
    mats = np.array([np.eye(3), np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])])
    quats = mat_to_quat(mats)
    expected = np.array([[0.0, 0.0, 0.0, 1.0], [0.7071068, 0.0, 0.0, 0.7071068]])  # Identity  # 90-degree X rotation
    # Handle possible sign flips
    actual_abs = np.abs(quats)
    expected_abs = np.abs(expected)
    npt.assert_allclose(actual_abs, expected_abs, rtol=1e-5, atol=1e-5)


def test_round_trip_conversion():
    # Test round trip conversion (quat -> mat -> quat)
    quats = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # Identity
            [0.5, 0.5, 0.5, 0.5],  # 120-degree rotation around [1,1,1]
            [0.0, 0.0, 0.7071, 0.7071],  # 90-degree rotation around Z
        ]
    )

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    # Convert to matrices and back
    mats = quat_to_mat(quats)
    quats_recovered = mat_to_quat(mats)

    # Since quaternions can be negated and still represent the same rotation,
    # we need to check if q or -q matches our original
    quats_standardized = standardize_quaternion(quats)
    quats_recovered_standardized = standardize_quaternion(quats_recovered)

    npt.assert_allclose(quats_standardized, quats_recovered_standardized, rtol=1e-5, atol=1e-5)


def test_standardize_quaternion():
    # Test with quaternions having positive real part
    quats = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # Identity, real part is positive (should remain unchanged)
            [0.5, 0.5, 0.5, 0.5],  # Real part is positive (should remain unchanged)
        ]
    )
    result = standardize_quaternion(quats)
    expected = quats
    npt.assert_allclose(result, expected)

    # Test with quaternions having negative real part
    quats = np.array(
        [
            [0.0, 0.0, 0.0, -1.0],  # Identity with negative real part (should be negated)
            [0.5, 0.5, 0.5, -0.5],  # Negative real part (should be negated)
        ]
    )
    result = standardize_quaternion(quats)
    expected = -quats  # All quaternions should be negated
    npt.assert_allclose(result, expected)
