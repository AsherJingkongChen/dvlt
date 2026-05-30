# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for logging callbacks (replaces old Module._log_train tests)."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from dvlt.callbacks import ParameterLoggingCallback
from tests.utils.fixtures import MockModule, MockTracker


def add_gradients_to_model(model):
    """Add dummy gradients to model parameters for testing."""
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.randn(4, 1)
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()


@pytest.fixture
def mock_trainer():
    """Create a mock trainer with a model."""
    trainer = Mock()
    trainer.accelerator = Mock()
    trainer.accelerator.trackers = []

    # Create actual model
    module = MockModule(input_size=10, hidden_size=5)
    trainer.model = module

    return trainer


@pytest.mark.parametrize(
    "log_params,expected_count",
    [
        (None, 0),  # No logging
        ([], 8),  # Log all (4 params × 2 metrics)
        (["layer1"], 4),  # Log layer1 only (2 params × 2 metrics)
        (["layer1.weight", "layer2.bias"], 4),  # Log specific params
    ],
)
def test_parameter_logging_callback(mock_trainer, log_params, expected_count):
    """Test ParameterLoggingCallback with different configurations."""
    callback = ParameterLoggingCallback(log_params=log_params, log_every_n_steps=1)

    # Add gradients to model
    add_gradients_to_model(mock_trainer.model.model)

    # Create mock tracker
    tracker = MockTracker()
    mock_trainer.accelerator.trackers = [tracker]

    # Call callback
    callback.on_train_batch(batch={}, predictions={}, step=1, trainer=mock_trainer)

    # Check logging
    assert len(tracker.logged_data) == expected_count


def test_parameter_logging_with_frozen_parameters(mock_trainer):
    """Test that frozen parameters are not logged."""
    # Create module with frozen parameter
    module = MockModule(input_size=10, hidden_size=5, freeze=["layer1.weight"])
    module.freeze()
    mock_trainer.model = module

    # Log all parameters
    callback = ParameterLoggingCallback(log_params=[], log_every_n_steps=1)

    add_gradients_to_model(module.model)

    tracker = MockTracker()
    mock_trainer.accelerator.trackers = [tracker]

    callback.on_train_batch({}, {}, 1, mock_trainer)

    # Should log only trainable parameters (3 params × 2 metrics)
    assert len(tracker.logged_data) == 6

    # Check that frozen parameter is not logged
    logged_keys = set()
    for _, data in tracker.logged_data:
        logged_keys.update(data.keys())

    assert "param_hist/layer1.weight" not in logged_keys
    assert "param_hist/layer1.bias" in logged_keys


def test_parameter_logging_frequency(mock_trainer):
    """Test that logging respects log_every_n_steps."""
    callback = ParameterLoggingCallback(log_params=[], log_every_n_steps=10)

    add_gradients_to_model(mock_trainer.model.model)
    tracker = MockTracker()
    mock_trainer.accelerator.trackers = [tracker]

    # Should not log at step 5
    callback.on_train_batch({}, {}, 5, mock_trainer)
    assert len(tracker.logged_data) == 0

    # Should log at step 10
    callback.on_train_batch({}, {}, 10, mock_trainer)
    assert len(tracker.logged_data) > 0
