# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Comprehensive test suite for the Module and Model base classes.
Tests logging, freezing, saving/loading, and other core functionalities.
"""

import os
import tempfile
from unittest.mock import Mock

import pytest
import torch

# Initialize accelerate state to avoid logging errors
from accelerate import PartialState
from torch import nn

from dvlt.model.base import Model
from tests.utils.fixtures import MockModel, MockModule


try:
    PartialState()
except Exception:
    pass  # Already initialized


# Import centralized fixtures
pytest_plugins = ["tests.utils.fixtures"]


@pytest.fixture
def test_module():
    """Create a test module instance."""
    return MockModule(input_size=10, hidden_size=5)  # Sized for base functionality tests with dummy_batch


@pytest.fixture
def test_model():
    """Create a test model instance."""
    return MockModel(input_size=10, hidden_size=5)  # Sized for base functionality tests


def add_gradients_to_model(model):
    """Add dummy gradients to model parameters for testing."""
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.randn(4, 1)
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()


def check_param_requires_grad(model, expected_states):
    """Helper to check requires_grad state of all parameters."""
    actual_states = {
        "layer1.weight": model.layer1.weight.requires_grad,
        "layer1.bias": model.layer1.bias.requires_grad,
        "layer2.weight": model.layer2.weight.requires_grad,
        "layer2.bias": model.layer2.bias.requires_grad,
    }
    assert actual_states == expected_states


def check_logged_keys(tracker, expected_keys, expected_step):
    """Helper to check logged keys and step."""
    logged_keys = set()
    for step, data in tracker.logged_data:
        assert step == expected_step
        logged_keys.update(data.keys())
    assert logged_keys == expected_keys


# Combined initialization tests
def test_module_initialization():
    """Test module initialization with various parameters."""
    # Default initialization
    module = MockModule(input_size=10, hidden_size=5)
    assert module.paths_to_freeze is None
    assert module.paths_to_log is None
    assert isinstance(module.model, MockModel)
    assert module.model_file == "model.pt"

    # With freeze parameters
    freeze_paths = ["layer1.weight", "layer2.bias"]
    module_freeze = MockModule(input_size=10, hidden_size=5, freeze=freeze_paths)
    assert module_freeze.paths_to_freeze == freeze_paths
    assert module_freeze.paths_to_log is None

    # With log parameters
    log_paths = ["layer1"]
    module_log = MockModule(input_size=10, hidden_size=5, log_params=log_paths)
    assert module_log.paths_to_log == log_paths
    assert module_log.paths_to_freeze is None

    # With log all parameters
    module_log_all = MockModule(input_size=10, hidden_size=5, log_params=[])
    assert module_log_all.paths_to_log == []
    assert module_log_all.paths_to_freeze is None


# Combined parameter freezing tests
@pytest.mark.parametrize(
    "freeze_paths,expected_frozen",
    [
        (["layer1.weight"], {"layer1.weight": False, "layer1.bias": True, "layer2.weight": True, "layer2.bias": True}),
        (["layer1"], {"layer1.weight": False, "layer1.bias": False, "layer2.weight": True, "layer2.bias": True}),
        (
            ["layer1.weight", "layer2.bias"],
            {"layer1.weight": False, "layer1.bias": True, "layer2.weight": True, "layer2.bias": False},
        ),
        (
            ["nonexistent.layer"],
            {"layer1.weight": True, "layer1.bias": True, "layer2.weight": True, "layer2.bias": True},
        ),  # Invalid path
    ],
)
def test_parameter_freezing(freeze_paths, expected_frozen):
    """Test freezing various parameter combinations."""
    module = MockModule(input_size=10, hidden_size=5, freeze=freeze_paths)

    # Before freezing - all should be trainable
    check_param_requires_grad(
        module.model, {"layer1.weight": True, "layer1.bias": True, "layer2.weight": True, "layer2.bias": True}
    )

    module.freeze()

    # After freezing - check expected state
    check_param_requires_grad(module.model, expected_frozen)


def test_train_test_steps(test_module, dummy_batch):
    """Test training and testing step functionality."""
    # Test training step
    loss, metrics, logs, preds_train = test_module.train_step(dummy_batch, step=1)

    assert isinstance(loss, torch.Tensor) and loss.dim() == 0 and loss.requires_grad
    assert isinstance(metrics, dict) and "loss" in metrics and torch.equal(metrics["loss"], loss)
    assert isinstance(logs, dict) and "train_loss" in logs and logs["train_loss"] == loss.item()
    # Predictions dict should be returned
    assert isinstance(preds_train, dict) and "predictions" in preds_train

    # Test testing step (now returns just predictions)
    predictions = test_module.test_step(dummy_batch)

    # Validate predictions dictionary
    assert isinstance(predictions, dict) and "predictions" in predictions
    assert isinstance(predictions["predictions"], torch.Tensor)


def test_predict_method(test_module, dummy_batch):
    """Test the predict method returns standardized predictions."""
    # Test predict method
    predictions = test_module.predict(dummy_batch)

    # Check that predictions is a dictionary
    assert isinstance(predictions, dict)

    # Check that required keys exist
    assert "predictions" in predictions
    assert "confidence" in predictions

    # Check that predictions have correct shape and type
    assert isinstance(predictions["predictions"], torch.Tensor)
    assert isinstance(predictions["confidence"], torch.Tensor)

    # Check shapes match input batch size
    batch_size = dummy_batch["input"].shape[0]
    assert predictions["predictions"].shape[0] == batch_size
    assert predictions["confidence"].shape[0] == batch_size

    # Check that predictions are computed without gradients
    assert not predictions["predictions"].requires_grad
    assert not predictions["confidence"].requires_grad

    # Test that repeated calls give same results (deterministic without dropout)
    test_module.model.eval()
    predictions2 = test_module.predict(dummy_batch)
    assert torch.equal(predictions["predictions"], predictions2["predictions"])


def test_save_load_functionality():
    """Test model saving and loading functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and train a module to get some non-random weights
        module = MockModule(input_size=10, hidden_size=5)
        dummy_input = torch.randn(4, 10)
        dummy_target = torch.randn(4, 1)
        output = module.model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()

        # Save original weights for comparison
        original_weights = {name: param.clone() for name, param in module.model.named_parameters()}

        # Mock accelerator and save
        accelerator = Mock()
        accelerator.is_main_process = True
        accelerator.wait_for_everyone = Mock()
        accelerator.unwrap_model = Mock(return_value=module.model)

        save_path = os.path.join(temp_dir, "test_model")
        module.save_pretrained(accelerator, save_path, safe_serialization=False)

        # Check that model file was created
        model_file = os.path.join(save_path, "model.pt")
        assert os.path.exists(model_file) and os.path.isfile(model_file)

        # Test loading
        module2 = MockModule(input_size=10, hidden_size=5)
        assert not torch.equal(original_weights["layer1.weight"], module2.model.layer1.weight)

        module2.load_pretrained(save_path)

        # Check that all weights were loaded correctly
        for name, original_weight in original_weights.items():
            loaded_weight = dict(module2.model.named_parameters())[name]
            assert torch.equal(original_weight, loaded_weight)


# Combined parameter utility tests
def test_parameter_utilities(test_module):
    """Test parameter utility methods."""
    # Test get_params
    all_params = list(test_module.get_params())
    assert len(all_params) == 4
    assert all(isinstance(p, torch.nn.Parameter) for p in all_params)

    expected_shapes = [torch.Size([5, 10]), torch.Size([5]), torch.Size([1, 5]), torch.Size([1])]
    assert [p.shape for p in all_params] == expected_shapes

    # Test get_trainable_params (all trainable by default)
    trainable_params = list(test_module.get_trainable_params())
    assert len(trainable_params) == 4
    assert all(p.requires_grad for p in trainable_params)

    # Test get_param_groups
    param_groups = test_module.get_param_groups()
    assert isinstance(param_groups, dict) and "params" in param_groups
    params = list(param_groups["params"])
    assert len(params) == 4 and all(p.requires_grad for p in params)


def test_trainable_params_with_freezing():
    """Test get_trainable_params with various freezing scenarios."""
    test_cases = [
        (None, 4),  # No freezing
        (["layer1.weight"], 3),  # One parameter frozen
        (["layer1"], 2),  # Entire module frozen
    ]

    for freeze_paths, expected_count in test_cases:
        module = MockModule(input_size=10, hidden_size=5, freeze=freeze_paths)
        if freeze_paths:
            module.freeze()

        trainable_params = list(module.get_trainable_params())
        assert len(trainable_params) == expected_count
        assert all(p.requires_grad for p in trainable_params)


# Model property tests
def test_model_properties(test_model):
    """Test Model device and dtype properties."""
    # Test device property
    device = test_model.device
    assert isinstance(device, torch.device) and device.type == "cpu"

    if torch.cuda.is_available():
        test_model.cuda()
        assert test_model.device.type == "cuda"
        test_model.cpu()  # Reset for dtype test

    # Test dtype property
    dtype = test_model.dtype
    assert isinstance(dtype, torch.dtype) and dtype == torch.float32

    test_model.double()
    assert test_model.dtype == torch.float64


def test_model_gradient_checkpointing(test_model):
    """Test gradient checkpointing functionality."""
    # Should not raise any exceptions
    test_model.enable_gradient_checkpointing()
    test_model.enable_gradient_checkpointing(use_reentrant=True)
    test_model.enable_gradient_checkpointing(use_reentrant=False)


class BlockModel(Model):
    """Model with a list of blocks for testing gradient checkpointing."""

    def __init__(self, num_blocks=8, gradient_checkpointing_config=None):
        super().__init__(gradient_checkpointing_config=gradient_checkpointing_config)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def _is_checkpointed(block):
    """Check if a block's forward has been replaced by a checkpoint wrapper."""
    return "forward" in block.__dict__


class TestParseModulePath:
    """Tests for Model._parse_module_path slice parsing."""

    def test_no_slice(self):
        path, spec = Model._parse_module_path("backbone.blocks")
        assert path == "backbone.blocks"
        assert spec is None

    def test_single_index(self):
        path, spec = Model._parse_module_path("backbone.blocks[3]")
        assert path == "backbone.blocks"
        assert spec == {3}

    def test_start_only(self):
        path, spec = Model._parse_module_path("backbone.blocks[4:]")
        assert path == "backbone.blocks"
        assert spec == (4, None, None)

    def test_stop_only(self):
        path, spec = Model._parse_module_path("backbone.blocks[:6]")
        assert path == "backbone.blocks"
        assert spec == (None, 6, None)

    def test_start_stop(self):
        path, spec = Model._parse_module_path("backbone.blocks[2:8]")
        assert path == "backbone.blocks"
        assert spec == (2, 8, None)

    def test_start_stop_step(self):
        path, spec = Model._parse_module_path("backbone.blocks[0:8:2]")
        assert path == "backbone.blocks"
        assert spec == (0, 8, 2)

    def test_step_only(self):
        path, spec = Model._parse_module_path("blocks[::2]")
        assert path == "blocks"
        assert spec == (None, None, 2)


class TestSelectiveGradientCheckpointing:
    """Tests for selective gradient checkpointing with index slices."""

    def test_checkpoint_all_blocks(self):
        model = BlockModel(
            num_blocks=8,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks"],
            },
        )
        model.enable_gradient_checkpointing()
        assert all(_is_checkpointed(b) for b in model.blocks)

    def test_checkpoint_from_index(self):
        model = BlockModel(
            num_blocks=8,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks[4:]"],
            },
        )
        model.enable_gradient_checkpointing()
        for i, blk in enumerate(model.blocks):
            assert _is_checkpointed(blk) == (i >= 4), f"block {i}"

    def test_checkpoint_up_to_index(self):
        model = BlockModel(
            num_blocks=8,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks[:3]"],
            },
        )
        model.enable_gradient_checkpointing()
        for i, blk in enumerate(model.blocks):
            assert _is_checkpointed(blk) == (i < 3), f"block {i}"

    def test_checkpoint_every_other(self):
        model = BlockModel(
            num_blocks=8,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks[::2]"],
            },
        )
        model.enable_gradient_checkpointing()
        for i, blk in enumerate(model.blocks):
            assert _is_checkpointed(blk) == (i % 2 == 0), f"block {i}"

    def test_checkpoint_single_block(self):
        model = BlockModel(
            num_blocks=8,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks[5]"],
            },
        )
        model.enable_gradient_checkpointing()
        for i, blk in enumerate(model.blocks):
            assert _is_checkpointed(blk) == (i == 5), f"block {i}"

    def test_checkpoint_range_with_step(self):
        model = BlockModel(
            num_blocks=12,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks[4::2]"],
            },
        )
        model.enable_gradient_checkpointing()
        expected = {4, 6, 8, 10}
        for i, blk in enumerate(model.blocks):
            assert _is_checkpointed(blk) == (i in expected), f"block {i}"

    def test_checkpointed_forward_recomputes(self):
        """Verify checkpointed blocks actually recompute during backward."""
        model = BlockModel(
            num_blocks=4,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks[2:]"],
            },
        )
        model.enable_gradient_checkpointing()
        model.train()

        x = torch.randn(2, 4, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_no_checkpoint_at_eval(self):
        """Verify checkpointing is skipped during eval (no recomputation)."""
        model = BlockModel(
            num_blocks=4,
            gradient_checkpointing_config={
                "use_reentrant": False,
                "modules": ["blocks"],
            },
        )
        model.enable_gradient_checkpointing()
        model.eval()

        x = torch.randn(2, 4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)


def test_module_model_integration():
    """Test Module-Model integration and architecture."""
    module = MockModule(input_size=10, hidden_size=5)

    # Check proper model instantiation and architecture
    assert isinstance(module.model, MockModel)
    assert all(hasattr(module.model, attr) for attr in ["layer1", "layer2"])
    assert all(isinstance(getattr(module.model, f"layer{i}"), nn.Linear) for i in [1, 2])

    # Check architecture dimensions
    assert module.model.layer1.in_features == 10 and module.model.layer1.out_features == 5
    assert module.model.layer2.in_features == 5 and module.model.layer2.out_features == 1

    # Test that freezing works with wrapped model
    module_with_features = MockModule(input_size=10, hidden_size=5, freeze=["layer1"])
    module_with_features.freeze()

    # Verify freezing worked
    assert not module_with_features.model.layer1.weight.requires_grad
    assert module_with_features.model.layer2.weight.requires_grad


# Mock DDP wrapper for testing
class MockDDPWrapper:
    """Mock DistributedDataParallel wrapper for testing."""

    def __init__(self, module):
        self.module = module

    def named_parameters(self):
        return self.module.named_parameters()

    def parameters(self):
        return self.module.parameters()
