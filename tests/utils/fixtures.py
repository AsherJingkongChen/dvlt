# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized test fixtures to eliminate duplication."""

import io
import sys
import tempfile
from typing import Any, List, Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from accelerate.state import AcceleratorState, PartialState
from torch import nn
from torch.utils.data import Dataset

from dvlt.common.constants import DataField
from dvlt.model.base import Model, Module


# =============================================================================
# Dataset Fixtures
# =============================================================================
class DummyBaseDataset(Dataset):
    """Base dataset for testing purposes."""

    def __init__(self, length: int = 856):
        self.length = length
        self.cumulative_sizes = [length]

    def __len__(self):
        return self.length


class DummyMultiBaseDataset(Dataset):
    """Base dataset for testing purposes."""

    def __init__(self, length: int = 856):
        self.length = length * 3
        self.cumulative_sizes = [length, length * 2, length * 3]
        self.repeat_factors = torch.tensor([0.2, 0.3, 0.5])

    def __len__(self):
        return self.length


class DummyDataset(Dataset):
    """Reusable dataset for testing purposes.

    Consolidates the duplicate DummyDataset classes found across:
    - tests/data/test_sampler.py
    - tests/data/test_loader.py
    - tests/data/test_integration.py
    - tests/data/test_distributed_integration.py
    """

    def __init__(self, length: int = 856, multi: bool = False):
        if multi:
            self.dataset = DummyMultiBaseDataset(length)
        else:
            self.dataset = DummyBaseDataset(length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx

    @property
    def repeat_factors(self):
        """Forward repeat_factors from the wrapped dataset if available."""
        return getattr(self.dataset, "repeat_factors", None)

    @property
    def cumulative_sizes(self):
        """Forward cumulative_sizes from the wrapped dataset."""
        return getattr(self.dataset, "cumulative_sizes", [len(self.dataset)])


# Alias for compatibility with existing tests
@pytest.fixture
def dummy_dataset():
    """Standard dummy dataset for testing."""
    return DummyDataset()


@pytest.fixture
def dummy_multi_dataset():
    """Standard dummy multi dataset for testing."""
    return DummyDataset(multi=True)


# =============================================================================
# Sampler Configuration Fixtures
# =============================================================================


@pytest.fixture
def sampler_config():
    """Standard sampler configuration used across tests.

    Consolidates the duplicate sampler_config fixtures found across:
    - tests/data/test_sampler.py
    - tests/data/test_loader.py
    - tests/data/test_integration.py
    - tests/data/test_distributed_integration.py
    """
    return {
        "aspect_ratio_bounds": (0.33, 1.0),
        "frames_per_sample_range": (2, 24),
        "max_frames_per_gpu": 48,
        "shuffle": True,
        "seed": 42,
        "infinite": False,
    }


# =============================================================================
# Vision/Image Data Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Fixture providing a sample image."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 128


@pytest.fixture
def sample_depth():
    """Fixture providing a sample depth map."""
    return np.ones((100, 200), dtype=np.float32)


@pytest.fixture
def sample_extrinsics():
    """Fixture providing sample extrinsic camera matrix."""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)


@pytest.fixture
def sample_intrinsics():
    """Fixture providing sample intrinsic camera matrix."""
    return np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]], dtype=np.float32)


# =============================================================================
# Model Fixtures
# =============================================================================


class MockTracker:
    """Mock tracker for logging tests."""

    def __init__(self):
        # Mimic the interface of accelerate trackers (e.g. WandB, TensorBoard)
        # so that the production logging code can seamlessly interact with this
        # lightweight mock within the unit-tests.
        #
        # The current implementation of `dvlt.model.base._log_parameter` checks
        # the `name` attribute to decide how to format the logged values
        # (histogram for "wandb", raw tensor for "tensorboard").  Providing a
        # default value of "wandb" is sufficient for the tests – it exercises
        # the histogram branch while staying agnostic to any external
        # dependencies.
        self.name = "tensorboard"
        self.logged_data = []

    def log(self, data, step):
        self.logged_data.append((step, data))


class MockModel(Model):
    """Model for trainer integration testing that supports freezing."""

    def __init__(self, input_size=32, hidden_size=4, output_size=1):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Handle different input shapes
        if x.dim() > 2:  # e.g. [B, S, C, H, W]
            # Flatten to [B*S, features]
            x = x.flatten(start_dim=2).view(-1, x.shape[-1])

        # Regular forward pass through the network
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

    def enable_gradient_checkpointing(self, use_reentrant: bool = False) -> None:
        """Enable gradient checkpointing for the model."""
        pass

    def get_param_groups(self):
        """Return parameter groups for optimizer setup."""
        return {"params": list(self.parameters())}

    def print_summary(self):
        """Print model summary for testing."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("MockModel Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model architecture: {self}")


class MockModule(Module):
    """Mock module for testing trainer integration."""

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 4,
        output_size: int = 1,
        freeze: Optional[List[str]] = None,
        log_params: Optional[List[str]] = None,
        loss_scale: float = 1.0,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_scale = loss_scale
        self.test_step_count = 0
        super().__init__(freeze=freeze, log_params=log_params)

    def build_model(self):
        return MockModel(input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.output_size)

    def train_step(self, batch: dict, step: int, accelerator=None):
        """Mock training step that returns loss and metrics."""
        # Handle both batch formats: DataField.IMAGES (trainer integration) and "input" (base tests)
        if DataField.IMAGES in batch:
            x = batch[DataField.IMAGES]  # [B, S, C, H, W] for trainer integration
            # Make training deterministic by using step as seed for reproducible loss
            torch.manual_seed(step + 42)  # Add offset to avoid seed=0 issues

            # Simple dummy computation to get a loss
            output = self.model(x)
            # Create a dummy target that's deterministic based on step
            target = output.detach() + (torch.randn_like(output) * 0.1)
            loss = nn.MSELoss()(output, target) * self.loss_scale

            # Mock metrics for progress bar
            pbar_logs = {
                "train_loss": loss.detach(),
                "step": torch.tensor(step, dtype=torch.float32, device=loss.device),
            }

            # Mock tracker logs
            tracker_logs = {
                "mse": loss.detach() / self.loss_scale,
                "output_mean": output.mean().detach(),
            }

            predictions = {"predictions": output}
            return loss, pbar_logs, tracker_logs, predictions
        else:
            # Handle base test format with "input"/"target"
            x = batch["input"]
            y = batch["target"]

            output = self.model(x)
            loss = nn.MSELoss()(output, y) * self.loss_scale

            # Mock metrics
            metrics = {"loss": loss}
            logs = {"train_loss": loss.item()}

            predictions = {"predictions": output}
            return loss, metrics, logs, predictions

    def test_step(self, batch: dict, accelerator=None) -> dict:
        """Mock test step for validation.

        Note: Now returns just predictions - callbacks handle metrics.
        """
        self.test_step_count += 1

        # Just predict - callbacks handle metrics
        return self.predict(batch, accelerator)

    def predict(self, batch: dict, accelerator=None) -> dict:
        """Mock predict method."""
        # Handle both batch formats
        if DataField.IMAGES in batch:
            x = batch[DataField.IMAGES]  # [B, S, C, H, W] for trainer integration
        else:
            x = batch["input"]

        with torch.no_grad():
            output = self.model(x)

        return {
            "predictions": output,
            "confidence": torch.ones_like(output) * 0.9,
        }

    def get_model_state_dict(self):
        """Get model parameters for state comparison."""
        # Handle both wrapped (DDP) and unwrapped models
        if hasattr(self.model, "module"):
            # Model is wrapped (e.g., by DDP)
            return {name: param.clone().detach() for name, param in self.model.module.named_parameters()}
        else:
            # Model is not wrapped
            return {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compare_model_states(self, state1: dict, state2: dict, rtol=1e-5):
        """Compare two model state dictionaries."""
        if set(state1.keys()) != set(state2.keys()):
            return False

        for key, value in state1.items():
            # Move both tensors to CPU for comparison to handle device differences
            tensor1 = value.cpu()
            tensor2 = state2[key].cpu()
            if not torch.allclose(tensor1, tensor2, rtol=rtol):
                return False
        return True

    def print_summary(self):
        """Print model summary for testing."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("TrainerMockModule Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model architecture: {self.model}")


@pytest.fixture
def mock_module():
    """Create a trainer mock module instance for integration testing."""
    return MockModule()  # Uses default parameters (32, 4, 1) for trainer integration tests


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    return {"input": torch.randn(4, 10), "target": torch.randn(4, 1)}


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# =============================================================================
# Training Configuration Fixtures
# =============================================================================


@pytest.fixture
def basic_trainer_config():
    """Basic trainer configuration for testing."""
    return {
        "max_train_steps": 20,
        "validation_steps": 100,  # Disable validation by setting high value
        "validation_batches": 3,
        "checkpointing_steps": 100,  # Disable checkpointing for basic test
        "checkpoints_total_limit": 2,
        "print_step_interval": 5,
        "learning_rate": 1e-3,
        "gradient_accumulation_steps": 1,
        "tqdm": False,  # Disable tqdm for cleaner test output
        "experiment_logger": (),  # Disable logging for tests
    }


# =============================================================================
# I/O and Capture Fixtures
# =============================================================================


@pytest.fixture
def capture_stdout():
    """Fixture to capture stdout for testing."""
    original_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    try:
        yield captured_output
    finally:
        sys.stdout = original_stdout


# =============================================================================
# Auto-use Fixtures for State Management
# =============================================================================


@pytest.fixture(autouse=True)
def reset_accelerator_state():
    """Reset AcceleratorState after each test to prevent initialization conflicts.

    For distributed tests, we need to be more careful about cleanup to prevent
    state from one test interfering with another.
    """
    # Clean up any existing state BEFORE the test
    try:
        AcceleratorState._shared_state.clear()
        PartialState._shared_state.clear()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    # Initialize minimal accelerate state for logging
    # (some modules use accelerate.logging.get_logger())
    try:
        PartialState()
    except Exception:
        pass

    yield  # Run the test

    # Clean up after test
    try:
        AcceleratorState._shared_state.clear()
        PartialState._shared_state.clear()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def mock_accelerate_logger():
    """Mock the accelerate logger for progress tests."""
    with patch("dvlt.engine.progress.logger") as mock_logger:
        yield mock_logger


def compare_batches(batch1: Any, batch2: Any) -> bool:
    """Helper function to compare batches handling tensors properly.

    This function handles comparison of batches containing tensors, lists, tuples,
    and other data types commonly found in data loader outputs.

    Args:
        batch1: First batch to compare
        batch2: Second batch to compare

    Returns:
        bool: True if batches are equal, False otherwise
    """
    if isinstance(batch1, torch.Tensor) and isinstance(batch2, torch.Tensor):
        # Move to CPU for comparison
        b1_cpu = batch1.cpu() if batch1.is_cuda else batch1
        b2_cpu = batch2.cpu() if batch2.is_cuda else batch2
        return torch.equal(b1_cpu, b2_cpu)
    elif isinstance(batch1, (list, tuple)) and isinstance(batch2, (list, tuple)):
        if len(batch1) != len(batch2):
            return False
        return all(compare_batches(b1, b2) for b1, b2 in zip(batch1, batch2, strict=False))
    else:
        return batch1 == batch2


def get_optimal_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
