"""Modular wrappers over :class:`DVLTModel`.

Decomposes :class:`DVLTModel` into ``nn.Module`` sub-units (aggregator and
per-modality heads) that can be plucked off and reused independently.
Sub-module parameters are shared by reference with the host model.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from dvlt.model_components import activate_head

from .heads import _concat_uv
from .model import DVLTModel, _slice_expand_flatten


class DVLT(nn.Module):
    """Top-level container exposing the aggregator and per-modality heads."""

    def __init__(self, model: DVLTModel):
        super().__init__()

        # ==================== Image Encoder ====================
        self.aggregator = Aggregator(model)

        # ==================== Decoder Heads ====================
        self.depth_head = DepthHead(model)
        self.camera_head = CameraHead(model)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """images: (B, S, 3, H, W) -> {depth, depth_conf, pose_enc}."""
        H, W = images.shape[-2:]
        features, patch_start_idx = self.aggregator(images)
        depth, depth_conf = self.depth_head(features, H, W, patch_start_idx)
        pose_enc = self.camera_head(features, H, W)
        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "pose_enc": pose_enc,
        }


class Aggregator(nn.Module):
    """Multi-view image encoder: patch tokens through the looped recurrent block."""

    def __init__(self, model: DVLTModel):
        super().__init__()

        # ==================== Patch Embedder ====================
        self.patch_embed_encoder = model.patch_embed_encoder

        # ==================== Positional Encoding ====================
        self.rope = model.rope
        self.position_getter = model.position_getter

        # ==================== Special Tokens ====================
        self.num_register_tokens = model.num_register_tokens
        self.patch_start_idx = model.patch_start_idx
        self.camera_token = model.camera_token
        self.register_token = model.register_token

        # ==================== Recurrent AA Blocks ====================
        self.recurrent_blocks = model.recurrent_blocks

        # ==================== Host-bound Methods ====================
        self._get_rope_positions = model._get_rope_positions
        self._solve_inference = model._solve_inference

        # ==================== Normalization Constants ====================
        self.register_buffer("_resnet_mean", model._resnet_mean.clone(), persistent=False)
        self.register_buffer("_resnet_std", model._resnet_std.clone(), persistent=False)

    def forward(self, images: Tensor) -> Tuple[List[Tensor], int]:
        """images: (B, S, 3, H, W) -> ([(B, S, 1 + R + H_P * W_P, C)], int)."""
        B, S, _, H, W = images.shape

        x = (images - self._resnet_mean) / self._resnet_std
        x = x.view(B * S, 3, H, W)
        z_0 = self.patch_embed_encoder(x, is_training=True)["x_norm_patchtokens"]

        register_token = self.register_token.expand(B, S, -1, -1).reshape(
            B * S, self.num_register_tokens, -1
        )
        camera_token = _slice_expand_flatten(self.camera_token, B, S)
        x = torch.cat([camera_token, register_token, z_0], dim=1)

        rope_pos = self._get_rope_positions(B * S, H, W, images.device)
        features = self._solve_inference(x, rope_pos, B, S)
        features = features.view(B, S, *features.shape[1:])
        return [features], self.patch_start_idx


class CameraHead(nn.Module):
    """Per-view camera-pose head over the ray-decoder transformer trunk."""

    def __init__(self, model: DVLTModel):
        super().__init__()

        # ==================== Transformer Front-end ====================
        self.proj_in = model.ray_decoder.proj_in
        self.blocks = model.ray_decoder.blocks

        # ==================== Camera Head ====================
        self.camera_head = model.camera_head

        # ==================== Cached Constants ====================
        self.decode_chunk_size = model.decode_chunk_size

        # ==================== Host-bound Methods ====================
        self._get_rope_positions = model._get_rope_positions

    def forward(
        self,
        features: List[Tensor],
        H: int,
        W: int,
    ) -> Tensor:
        """features[0]: (B, S, 1 + R + H_P * W_P, C) -> (B, S, 3 + 4 + 2)."""
        x = features[0]
        B, S = x.shape[:2]
        x = x.reshape(B * S, *x.shape[2:])
        rope_pos = self._get_rope_positions(B * S, H, W, x.device)

        BS = B * S
        chunk = self.decode_chunk_size if self.decode_chunk_size is not None else BS
        cls_chunks: List[Tensor] = []
        for start in range(0, BS, chunk):
            end = min(start + chunk, BS)
            cls_chunks.append(self._forward_impl(x[start:end], rope_pos[start:end]))
        cls = cls_chunks[0] if len(cls_chunks) == 1 else torch.cat(cls_chunks, dim=0)

        return self.camera_head(cls, B, S)

    def _forward_impl(self, x, pos):
        """Per-chunk forward over the ray-decoder trunk; returns the camera-token slot."""
        x = self.proj_in(x)
        for blk in self.blocks:
            x = grad_checkpoint(blk, x, pos=pos, use_reentrant=False)
        return x[:, 0]


class DepthHead(nn.Module):
    """Per-pixel depth and depth-confidence head over the depth-decoder conv path."""

    def __init__(self, model: DVLTModel):
        super().__init__()

        # ==================== Transformer Front-end ====================
        self.proj_in = model.depth_decoder.proj_in
        self.blocks = model.depth_decoder.blocks
        self.norm = model.depth_decoder.norm

        # ==================== Output Stage ====================
        self.upsample_blocks = model.depth_decoder.upsample_blocks
        self.output_block = model.depth_decoder.output_block

        # ==================== Cached Constants ====================
        self.patch_size = model.depth_decoder.patch_size
        self.decode_chunk_size = model.decode_chunk_size
        self.feature_only = False

        # ==================== Host-bound Methods ====================
        self._get_rope_positions = model._get_rope_positions

    def forward(
        self,
        features: List[Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """features[0]: (B, S, 1 + R + H_P * W_P, C) -> ((B, S, H, W, 1), (B, S, H, W, 1))."""
        x = features[0]
        B, S = x.shape[:2]
        x = x.reshape(B * S, *x.shape[2:])
        rope_pos = self._get_rope_positions(B * S, H, W, x.device)

        BS = B * S
        chunk = self.decode_chunk_size if self.decode_chunk_size is not None else BS
        chunks: List[Tensor] = []
        for start in range(0, BS, chunk):
            end = min(start + chunk, BS)
            chunks.append(
                self._forward_impl(x[start:end], H, W, patch_start_idx, rope_pos[start:end])
            )
        x = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

        if self.feature_only:
            return x.view(B, S, *x.shape[1:])

        depth, depth_conf = activate_head(x, activation="exp_clamped", conf_activation="exp_plus_one")
        depth = depth.view(B, S, H, W, 1)
        depth_conf = depth_conf.view(B, S, H, W, 1)
        return depth, depth_conf

    def _forward_impl(self, x, H, W, patch_start_idx, pos):
        """Per-chunk forward over the depth-decoder conv-head path."""
        B = x.shape[0]
        ph, pw = H // self.patch_size, W // self.patch_size

        x = self.proj_in(x)
        for blk in self.blocks:
            x = grad_checkpoint(blk, x, pos=pos, use_reentrant=False)
        x = self.norm(x)[:, patch_start_idx:]

        aspect_ratio = W / H
        x = x.permute(0, 2, 1).reshape(B, -1, ph, pw)
        for block in self.upsample_blocks:
            x = _concat_uv(x, aspect_ratio)
            for layer in block:
                x = grad_checkpoint(layer, x, use_reentrant=False)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        x = _concat_uv(x, aspect_ratio)
        if self.feature_only:
            return x
        with torch.autocast(x.device.type, enabled=False):
            x = self.output_block(x.float())
        return x
