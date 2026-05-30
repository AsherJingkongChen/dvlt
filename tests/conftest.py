# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "distributed: mark test as requiring multiple GPUs")
    config.addinivalue_line("markers", "accelerate: mark test as requiring accelerate launch")

    # Comprehensive warning suppression for cleaner test output
    warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*std\\(\\): degrees of freedom.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*log_with.*no supported trackers.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Detected call of.*lr_scheduler.step.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Mixed precision.*")
    warnings.filterwarnings("ignore", category=pytest.PytestAssertRewriteWarning)

    # Reduce logging verbosity
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("dvlt").setLevel(logging.WARNING)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables and configurations."""
    # Set environment variables for testing
    os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"  # Reduce Accelerate logging during tests
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Reduce transformers logging

    # Set torch settings for consistent behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def multi_gpu_available():
    """Check if multiple GPUs are available for testing."""
    return torch.cuda.device_count() > 1


def pytest_collection_modifyitems(config, items):
    """Skip tests based on their explicit markers and available resources."""
    for _ in items:
        # No automatic marking - rely entirely on explicit markers
        # The markers should be added explicitly in test files
        pass


def pytest_runtest_setup(item):
    """Setup function run before each test."""
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

    # Skip multi-GPU tests if only one GPU available
    if item.get_closest_marker("distributed"):
        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")
