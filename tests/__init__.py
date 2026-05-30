# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite with automatic discovery and intelligent categorization.

This package provides:
- Automatic test discovery and categorization
- Centralized test utilities and fixtures
- Smart hardware detection and execution planning
- Performance-optimized test execution
"""

from .utils.fixtures import DummyDataset, compare_batches, get_optimal_device, sampler_config


__all__ = ["DummyDataset", "sampler_config", "compare_batches", "get_optimal_device"]
