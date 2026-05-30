# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ray utilities and ray-to-pose conversion.

Tests the round-trip conversion:
1. GT camera params -> world rays (compute_world_rays)
2. world rays -> recovered camera params (rays_to_pose)
3. Compare recovered params with original GT
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from dvlt.common.rays import compute_world_rays, rays_to_pose


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def synthetic_camera_batch(device):
    """
    Create synthetic camera parameters for testing.

    Returns a batch with:
    - extrinsics_c2w: (B, S, 4, 4) camera-to-world matrices
    - intrinsics: (B, S, 3, 3) intrinsic matrices
    - H, W: image dimensions
    """
    B, S = 2, 4
    H, W = 256, 384

    # Create random but valid camera poses
    extrinsics_c2w = torch.eye(4, device=device, dtype=torch.float32)
    extrinsics_c2w = extrinsics_c2w.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).clone()

    # Add some variation to camera poses
    for b in range(B):
        for s in range(S):
            # Random rotation (small angles for realistic cameras)
            angles = np.random.uniform(-0.3, 0.3, size=3)  # radians
            R = torch.from_numpy(Rotation.from_euler("xyz", angles).as_matrix()).float().to(device)

            # Random translation
            t = torch.randn(3, device=device) * 0.5  # Small translations

            extrinsics_c2w[b, s, :3, :3] = R
            extrinsics_c2w[b, s, :3, 3] = t

    # Create typical intrinsics (fx, fy, cx, cy)
    intrinsics = torch.zeros(B, S, 3, 3, device=device, dtype=torch.float32)
    fx = 600.0
    fy = 600.0
    cx = W / 2.0
    cy = H / 2.0

    intrinsics[:, :, 0, 0] = fx
    intrinsics[:, :, 1, 1] = fy
    intrinsics[:, :, 0, 2] = cx
    intrinsics[:, :, 1, 2] = cy
    intrinsics[:, :, 2, 2] = 1.0

    return {
        "extrinsics_c2w": extrinsics_c2w,
        "intrinsics": intrinsics,
        "H": H,
        "W": W,
    }


# =============================================================================
# Helper functions
# =============================================================================


def rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation error in degrees between two rotation matrices.

    Args:
        R_pred: (B, S, 3, 3) predicted rotation matrices
        R_gt: (B, S, 3, 3) ground truth rotation matrices

    Returns:
        error: (B, S) rotation error in degrees
    """
    R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle_rad = torch.acos(cos_angle)
    return torch.rad2deg(angle_rad)


def translation_error(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """Compute translation error (L2 norm)."""
    return torch.norm(t_pred - t_gt, dim=-1)


def focal_length_error_percent(f_pred: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:
    """Compute focal length error as percentage."""
    rel_error = torch.abs(f_pred - f_gt) / (f_gt + 1e-8)
    return rel_error.mean(dim=-1) * 100


def principal_point_error(pp_pred: torch.Tensor, pp_gt: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Compute principal point error as percentage of image diagonal."""
    diag = torch.sqrt(torch.tensor(H**2 + W**2, dtype=pp_pred.dtype, device=pp_pred.device))
    error = torch.norm(pp_pred - pp_gt, dim=-1) / diag
    return error * 100


# =============================================================================
# Tests
# =============================================================================


class TestComputeWorldRays:
    """Tests for compute_world_rays function."""

    def test_output_shape(self, synthetic_camera_batch):
        """Test that output has correct shape."""
        batch = synthetic_camera_batch
        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            batch["H"],
            batch["W"],
        )

        B, S = batch["extrinsics_c2w"].shape[:2]
        H, W = batch["H"], batch["W"]

        assert rays.shape == (B, S, H, W, 6)

    def test_ray_directions_unnormalized(self, synthetic_camera_batch):
        """Test that ray directions are unnormalized (DA3 convention)."""
        batch = synthetic_camera_batch
        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            batch["H"],
            batch["W"],
        )

        ray_dirs = rays[..., :3]
        norms = torch.norm(ray_dirs, dim=-1)

        # Rays should NOT be unit normalized
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
        assert norms.min() > 0.5
        assert norms.max() < 10.0

    def test_ray_origins_match_camera_position(self, synthetic_camera_batch):
        """Test that ray origins match camera positions."""
        batch = synthetic_camera_batch
        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            batch["H"],
            batch["W"],
        )

        ray_origins = rays[..., 3:]
        cam_positions = batch["extrinsics_c2w"][:, :, :3, 3]

        for b in range(rays.shape[0]):
            for s in range(rays.shape[1]):
                expected = cam_positions[b, s]
                actual = ray_origins[b, s, 0, 0]
                assert torch.allclose(actual, expected, atol=1e-6)


class TestRaysToPose:
    """Tests for rays_to_pose function."""

    def test_output_shapes(self, synthetic_camera_batch):
        """Test that outputs have correct shapes."""
        batch = synthetic_camera_batch
        H, W = batch["H"], batch["W"]

        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            H,
            W,
        )

        B, S = rays.shape[:2]
        rays_conf = torch.ones(B, S, H, W, device=rays.device)

        extrinsics_c2w, intrinsics = rays_to_pose(rays, rays_conf, H, W, patch_size=16)

        assert extrinsics_c2w.shape == (B, S, 4, 4)
        assert intrinsics.shape == (B, S, 3, 3)


class TestRoundTrip:
    """Tests for round-trip conversion: GT -> rays -> recovered params."""

    def test_rotation_recovery(self, synthetic_camera_batch):
        """Test rotation recovery with strict threshold."""
        batch = synthetic_camera_batch
        H, W = batch["H"], batch["W"]

        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            H,
            W,
        )

        B, S = rays.shape[:2]
        rays_conf = torch.ones(B, S, H, W, device=rays.device)

        extrinsics_c2w_pred, _ = rays_to_pose(rays, rays_conf, H, W, patch_size=16)

        R_gt = batch["extrinsics_c2w"][:, :, :3, :3]
        R_pred = extrinsics_c2w_pred[:, :, :3, :3]

        rot_error = rotation_error_deg(R_pred, R_gt)
        mean_error = rot_error.mean().item()
        max_error = rot_error.max().item()

        assert mean_error < 0.05, f"Mean rotation error {mean_error:.6f}° exceeds 0.05°"
        assert max_error < 0.1, f"Max rotation error {max_error:.6f}° exceeds 0.1°"

    def test_translation_recovery(self, synthetic_camera_batch):
        """Test translation recovery with strict threshold."""
        batch = synthetic_camera_batch
        H, W = batch["H"], batch["W"]

        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            H,
            W,
        )

        B, S = rays.shape[:2]
        rays_conf = torch.ones(B, S, H, W, device=rays.device)

        extrinsics_c2w_pred, _ = rays_to_pose(rays, rays_conf, H, W, patch_size=16)

        t_gt = batch["extrinsics_c2w"][:, :, :3, 3]
        t_pred = extrinsics_c2w_pred[:, :, :3, 3]

        trans_error = translation_error(t_pred, t_gt)
        mean_error = trans_error.mean().item()
        max_error = trans_error.max().item()

        assert mean_error < 1e-6, f"Mean translation error {mean_error:.8f} exceeds 1e-6"
        assert max_error < 1e-5, f"Max translation error {max_error:.8f} exceeds 1e-5"

    def test_intrinsics_recovery(self, synthetic_camera_batch):
        """Test intrinsics recovery with strict threshold."""
        batch = synthetic_camera_batch
        H, W = batch["H"], batch["W"]

        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            H,
            W,
        )

        B, S = rays.shape[:2]
        rays_conf = torch.ones(B, S, H, W, device=rays.device)

        _, intrinsics_pred = rays_to_pose(rays, rays_conf, H, W, patch_size=16)

        # Focal length check
        f_gt = torch.stack([batch["intrinsics"][:, :, 0, 0], batch["intrinsics"][:, :, 1, 1]], dim=-1)
        f_pred = torch.stack([intrinsics_pred[:, :, 0, 0], intrinsics_pred[:, :, 1, 1]], dim=-1)

        focal_error = focal_length_error_percent(f_pred, f_gt)
        assert focal_error.mean().item() < 0.01, f"Focal length error {focal_error.mean():.4f}% exceeds 0.01%"

        # Principal point check
        pp_gt = torch.stack([batch["intrinsics"][:, :, 0, 2], batch["intrinsics"][:, :, 1, 2]], dim=-1)
        pp_pred = torch.stack([intrinsics_pred[:, :, 0, 2], intrinsics_pred[:, :, 1, 2]], dim=-1)

        pp_error = principal_point_error(pp_pred, pp_gt, H, W)
        assert pp_error.mean().item() < 0.01, f"Principal point error {pp_error.mean():.4f}% exceeds 0.01%"


class TestWorldPointReconstruction:
    """Tests for world point <-> ray/depth conversion."""

    def test_depth_reconstruction(self, synthetic_camera_batch):
        """Test that world_point = ray_origin + depth * ray_direction works correctly."""
        batch = synthetic_camera_batch
        H, W = batch["H"], batch["W"]
        device = batch["extrinsics_c2w"].device

        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            H,
            W,
        )

        B, S = rays.shape[:2]
        ray_dirs = rays[..., :3]
        ray_origins = rays[..., 3:]

        # Create random depths
        depths = torch.rand(B, S, H, W, device=device) * 5.0 + 0.5

        # Reconstruct world points
        world_points = ray_origins + depths.unsqueeze(-1) * ray_dirs

        # Project back to camera space and verify z-depth
        extrinsics_c2w = batch["extrinsics_c2w"]
        R = extrinsics_c2w[:, :, :3, :3]
        t = extrinsics_c2w[:, :, :3, 3:4]
        R_inv = R.transpose(-1, -2)
        t_inv = -torch.matmul(R_inv, t)

        world_points_flat = world_points.reshape(B, S, -1, 3)
        cam_points = torch.matmul(world_points_flat, R_inv.transpose(-1, -2)) + t_inv.transpose(-1, -2)
        cam_points = cam_points.reshape(B, S, H, W, 3)

        recovered_depths = cam_points[..., 2]
        depth_error = torch.abs(recovered_depths - depths)

        assert depth_error.mean().item() < 1e-6, f"Mean depth error {depth_error.mean():.8f} exceeds 1e-6"
        assert depth_error.max().item() < 1e-5, f"Max depth error {depth_error.max():.8f} exceeds 1e-5"

    def test_specific_point_reconstruction(self, device):
        """Test world point reconstruction with known 3D points."""
        B, S = 1, 1
        H, W = 64, 64

        # Camera at z=-5, looking at origin
        extrinsics_c2w = torch.eye(4, device=device, dtype=torch.float32)
        extrinsics_c2w[2, 3] = -5.0
        extrinsics_c2w = extrinsics_c2w.unsqueeze(0).unsqueeze(0)

        intrinsics = torch.zeros(B, S, 3, 3, device=device, dtype=torch.float32)
        fx, fy = 50.0, 50.0
        cx, cy = W / 2, H / 2
        intrinsics[:, :, 0, 0] = fx
        intrinsics[:, :, 1, 1] = fy
        intrinsics[:, :, 0, 2] = cx
        intrinsics[:, :, 1, 2] = cy
        intrinsics[:, :, 2, 2] = 1.0

        rays = compute_world_rays(extrinsics_c2w, intrinsics, H, W)
        ray_dirs = rays[..., :3]
        ray_origins = rays[..., 3:]

        # At center pixel with depth=5, reconstruct origin (0, 0, 0)
        center_h, center_w = H // 2, W // 2
        depth_at_center = torch.tensor(5.0, device=device)
        reconstructed_point = (
            ray_origins[0, 0, center_h, center_w] + depth_at_center * ray_dirs[0, 0, center_h, center_w]
        )

        expected_point = torch.tensor([0.0, 0.0, 0.0], device=device)
        point_error = torch.norm(reconstructed_point - expected_point).item()

        assert point_error < 0.15, f"Point reconstruction error {point_error:.6f} exceeds 0.15"

    def test_world_point_round_trip(self, synthetic_camera_batch):
        """Test full round-trip: world points -> project -> reconstruct."""
        batch = synthetic_camera_batch
        H, W = batch["H"], batch["W"]
        device = batch["extrinsics_c2w"].device

        B, S = batch["extrinsics_c2w"].shape[:2]

        rays = compute_world_rays(
            batch["extrinsics_c2w"],
            batch["intrinsics"],
            H,
            W,
        )
        ray_dirs = rays[..., :3]
        ray_origins = rays[..., 3:]

        # Create random world points in front of cameras
        cam_pos = batch["extrinsics_c2w"][:, :, :3, 3]
        cam_forward = batch["extrinsics_c2w"][:, :, :3, 2]

        num_points = 100
        random_offsets = torch.randn(B, S, num_points, 3, device=device) * 0.5
        random_depths = torch.rand(B, S, num_points, device=device) * 4.0 + 1.0

        world_points = cam_pos.unsqueeze(2) + cam_forward.unsqueeze(2) * random_depths.unsqueeze(-1) + random_offsets

        # Project to camera space
        extrinsics_c2w = batch["extrinsics_c2w"]
        intrinsics = batch["intrinsics"]

        R = extrinsics_c2w[:, :, :3, :3]
        t = extrinsics_c2w[:, :, :3, 3]
        R_inv = R.transpose(-1, -2)
        t_inv = -torch.einsum("bsij,bsj->bsi", R_inv, t)

        cam_points = torch.einsum("bsij,bsnj->bsni", R_inv, world_points) + t_inv.unsqueeze(2)

        z_depths = cam_points[..., 2]

        K = intrinsics
        cam_points_h = cam_points / cam_points[..., 2:3]
        pixels = torch.einsum("bsij,bsnj->bsni", K, cam_points_h)
        pixel_coords = pixels[..., :2]

        # Filter valid points
        valid = (
            (pixel_coords[..., 0] >= 0)
            & (pixel_coords[..., 0] < W)
            & (pixel_coords[..., 1] >= 0)
            & (pixel_coords[..., 1] < H)
            & (z_depths > 0)
        )

        errors = []
        for b in range(B):
            for s in range(S):
                valid_mask = valid[b, s]
                if valid_mask.sum() == 0:
                    continue

                valid_pixels = pixel_coords[b, s, valid_mask]
                valid_depths = z_depths[b, s, valid_mask]
                valid_world_pts = world_points[b, s, valid_mask]

                pixel_u = valid_pixels[:, 0].long().clamp(0, W - 1)
                pixel_v = valid_pixels[:, 1].long().clamp(0, H - 1)

                sampled_dirs = ray_dirs[b, s, pixel_v, pixel_u]
                sampled_origins = ray_origins[b, s, pixel_v, pixel_u]

                reconstructed = sampled_origins + valid_depths.unsqueeze(-1) * sampled_dirs
                point_errors = torch.norm(reconstructed - valid_world_pts, dim=-1)
                errors.append(point_errors)

        all_errors = torch.cat(errors)
        mean_error = all_errors.mean().item()

        # Allow some error due to pixel quantization
        assert mean_error < 0.03, f"Mean reconstruction error {mean_error:.6f} exceeds 0.03"


class TestIdentityCamera:
    """Test edge cases with identity cameras."""

    def test_identity_camera_round_trip(self, device):
        """Test round-trip with identity camera."""
        B, S = 1, 1
        H, W = 256, 384

        extrinsics_c2w = torch.eye(4, device=device, dtype=torch.float32)
        extrinsics_c2w = extrinsics_c2w.unsqueeze(0).unsqueeze(0)

        intrinsics = torch.zeros(B, S, 3, 3, device=device, dtype=torch.float32)
        intrinsics[:, :, 0, 0] = 500  # fx
        intrinsics[:, :, 1, 1] = 500  # fy
        intrinsics[:, :, 0, 2] = W / 2  # cx
        intrinsics[:, :, 1, 2] = H / 2  # cy
        intrinsics[:, :, 2, 2] = 1.0

        rays = compute_world_rays(extrinsics_c2w, intrinsics, H, W)
        rays_conf = torch.ones(B, S, H, W, device=device)

        extrinsics_pred, intrinsics_pred = rays_to_pose(rays, rays_conf, H, W, patch_size=16)

        # Check rotation
        R_pred = extrinsics_pred[0, 0, :3, :3]
        R_gt = extrinsics_c2w[0, 0, :3, :3]
        rot_error = rotation_error_deg(R_pred.unsqueeze(0).unsqueeze(0), R_gt.unsqueeze(0).unsqueeze(0))
        assert rot_error.item() < 0.01, f"Rotation error {rot_error.item():.6f}° exceeds 0.01°"

        # Check translation
        t_pred = extrinsics_pred[0, 0, :3, 3]
        t_gt = extrinsics_c2w[0, 0, :3, 3]
        trans_err = torch.norm(t_pred - t_gt).item()
        assert trans_err < 1e-6, f"Translation error {trans_err:.8f} exceeds 1e-6"

        # Check intrinsics
        fx_error = abs(intrinsics_pred[0, 0, 0, 0].item() - 500) / 500 * 100
        fy_error = abs(intrinsics_pred[0, 0, 1, 1].item() - 500) / 500 * 100
        cx_error = abs(intrinsics_pred[0, 0, 0, 2].item() - W / 2) / W * 100
        cy_error = abs(intrinsics_pred[0, 0, 1, 2].item() - H / 2) / H * 100

        assert fx_error < 0.01, f"fx error {fx_error:.4f}% exceeds 0.01%"
        assert fy_error < 0.01, f"fy error {fy_error:.4f}% exceeds 0.01%"
        assert cx_error < 0.01, f"cx error {cx_error:.4f}% exceeds 0.01%"
        assert cy_error < 0.01, f"cy error {cy_error:.4f}% exceeds 0.01%"
