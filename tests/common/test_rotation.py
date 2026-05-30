# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from dvlt.common.rotation import mat_to_quat, quat_to_mat, quaternion_slerp, so3_relative_angle, standardize_quaternion


def test_quat_to_mat():
    # Test identity quaternion [0, 0, 0, 1] (w-last format)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    mat = quat_to_mat(quat)
    expected = torch.eye(3)
    torch.testing.assert_close(mat, expected)

    # Test 90-degree rotation around X axis: [1, 0, 0, 0] (w-last format)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]) / torch.sqrt(torch.tensor(1.0))
    mat = quat_to_mat(quat)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    torch.testing.assert_close(mat, expected, rtol=1e-5, atol=1e-5)

    # Test batch of quaternions
    quats = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])  # Identity  # 90-degree X rotation
    mats = quat_to_mat(quats)
    expected = torch.stack([torch.eye(3), torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])])
    torch.testing.assert_close(mats, expected, rtol=1e-5, atol=1e-5)


def test_mat_to_quat():
    # Test identity matrix
    mat = torch.eye(3)
    quat = mat_to_quat(mat)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    torch.testing.assert_close(quat, expected)

    # Test 90-degree rotation around X axis
    mat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    quat = mat_to_quat(mat)
    # For 90-degree X-axis rotation, we expect [±0.7071, 0, 0, 0.7071]
    expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
    torch.testing.assert_close(quat.abs(), expected.abs(), rtol=1e-5, atol=1e-5)

    # Test batch of matrices
    mats = torch.stack([torch.eye(3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])])
    quats = mat_to_quat(mats)
    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.7071068, 0.0, 0.0, 0.7071068]]  # Identity  # 90-degree X rotation
    )
    # Handle possible sign flips
    actual_abs = quats.abs()
    expected_abs = expected.abs()
    torch.testing.assert_close(actual_abs, expected_abs, rtol=1e-5, atol=1e-5)


def test_round_trip_conversion():
    # Test round trip conversion (quat -> mat -> quat)
    quats = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # Identity
            [0.5, 0.5, 0.5, 0.5],  # 120-degree rotation around [1,1,1]
            [0.0, 0.0, 0.7071, 0.7071],  # 90-degree rotation around Z
        ]
    )

    # Normalize quaternions
    quats = quats / torch.norm(quats, dim=1, keepdim=True)

    # Convert to matrices and back
    mats = quat_to_mat(quats)
    quats_recovered = mat_to_quat(mats)

    # Since quaternions can be negated and still represent the same rotation,
    # we need to check if q or -q matches our original
    quats_standardized = standardize_quaternion(quats)
    quats_recovered_standardized = standardize_quaternion(quats_recovered)

    torch.testing.assert_close(quats_standardized, quats_recovered_standardized, rtol=1e-5, atol=1e-5)


def test_standardize_quaternion():
    # Test with quaternions having positive real part
    quats = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # Identity, real part is positive (should remain unchanged)
            [0.5, 0.5, 0.5, 0.5],  # Real part is positive (should remain unchanged)
        ]
    )
    result = standardize_quaternion(quats)
    expected = quats
    torch.testing.assert_close(result, expected)

    # Test with quaternions having negative real part
    quats = torch.tensor(
        [
            [0.0, 0.0, 0.0, -1.0],  # Identity with negative real part (should be negated)
            [0.5, 0.5, 0.5, -0.5],  # Negative real part (should be negated)
        ]
    )
    result = standardize_quaternion(quats)
    expected = -quats  # All quaternions should be negated
    torch.testing.assert_close(result, expected)


def test_quaternion_slerp_matches_scipy():
    batch_size = 12
    num_t = 7

    # Generate random unit quaternions (scalar-last) and standardize
    def random_quat(n: int) -> torch.Tensor:
        q = torch.randn(n, 4, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        return standardize_quaternion(q).to(torch.float64)

    q1 = random_quat(batch_size)
    q2 = random_quat(batch_size)

    # Avoid near-degenerate pairs where |dot| ~ 1 which can be ambiguous
    dots = (q1 * q2).sum(dim=-1).abs()
    degenerate = dots > 0.999999
    if degenerate.any():
        q2[degenerate] = random_quat(int(degenerate.sum().item()))

    # Interpolation parameters in [0, 1]
    t_vals = torch.linspace(0.0, 1.0, steps=num_t, dtype=torch.float64)

    # Our interpolation (broadcasted over t)
    q1b = q1[:, None, :].expand(-1, num_t, -1)
    q2b = q2[:, None, :].expand(-1, num_t, -1)
    tb = t_vals[None, :, None].expand(batch_size, -1, 1)
    ours = quaternion_slerp(q1b, q2b, tb).to(torch.float64)

    # SciPy interpolation
    scipy_mats = []
    key_times = np.array([0.0, 1.0])
    for b in range(batch_size):
        key_rots = R.from_quat(np.stack([q1[b].numpy(), q2[b].numpy()], axis=0))
        slerp = Slerp(key_times, key_rots)
        rots = slerp(t_vals.numpy())
        scipy_mats.append(torch.from_numpy(rots.as_matrix()))  # (num_t, 3, 3)
    scipy_mats = torch.stack(scipy_mats, dim=0).to(torch.float64)  # (B, T, 3, 3)

    # Convert ours to rotation matrices
    ours_mats = quat_to_mat(ours.reshape(-1, 4)).reshape(batch_size, num_t, 3, 3)

    # Compare via relative rotation angle
    diff_angles = so3_relative_angle(ours_mats.reshape(-1, 3, 3), scipy_mats.reshape(-1, 3, 3))

    # Expect very small angular difference (in radians)
    assert torch.all(diff_angles < 1e-5), f"Max angle diff: {diff_angles.max().item()}"

    # Endpoint checks: t=0 -> q1, t=1 -> q2 (up to sign)
    torch.testing.assert_close(
        standardize_quaternion(ours[:, 0, :]), standardize_quaternion(q1.to(ours.dtype)), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        standardize_quaternion(ours[:, -1, :]), standardize_quaternion(q2.to(ours.dtype)), rtol=1e-6, atol=1e-6
    )
