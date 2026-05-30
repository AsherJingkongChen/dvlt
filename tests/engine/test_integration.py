# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Integration test for the Trainer class.
Tests the complete training pipeline including data loading, model training,
validation, checkpointing, and state management using mock components.
"""

import os

import numpy as np
import pytest
import torch
from accelerate.utils import send_to_device

from dvlt.common.constants import DataField
from dvlt.data.datasets.eval import EvalDataset
from dvlt.data.datasets.train import TrainDataset
from dvlt.data.module import DataModule
from dvlt.engine.trainer import Trainer

# Import MockModule from fixtures
from tests.utils.fixtures import MockModule


# Mock implementations for testing


class MockTrainDataset(TrainDataset):
    """Mock training dataset for testing."""

    def __init__(self, length: int = 100, image_size: tuple = (3, 32, 32)):
        super().__init__()
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def available_data_fields(self):
        return {
            DataField.IMAGES,
            DataField.EXTRINSICS_C2W,
            DataField.INTRINSICS,
            DataField.DEPTHS,
            DataField.WORLD_POINTS,
            DataField.POINT_MASKS,
        }

    def get_data(self, seq_index=None, data_fields=None, img_per_seq=None, aspect_ratio=1.0):
        """Mock data generation following the expected interface."""
        if seq_index is None:
            seq_index = torch.randint(0, self.length, (1,)).item()

        if img_per_seq is None:
            img_per_seq = 2

        # Generate mock data - images should be numpy arrays in HWC format (uint8, 0-255)
        C, H, W = self.image_size
        images = [np.random.randint(0, 256, (H, W, C), dtype=np.uint8) for _ in range(img_per_seq)]

        # Convert to numpy arrays as expected by ComposedTrainDataset
        sample = {
            DataField.SEQ_NAME: f"mock_seq_{seq_index}",
            DataField.IDS: np.array(range(img_per_seq)),
            DataField.IMAGES: images,
            DataField.DEPTHS: [torch.randn(self.image_size[1], self.image_size[2]).numpy() for _ in range(img_per_seq)],
            DataField.EXTRINSICS_C2W: [torch.eye(4).numpy() for _ in range(img_per_seq)],
            DataField.INTRINSICS: [torch.eye(3).numpy() for _ in range(img_per_seq)],
            DataField.WORLD_POINTS: [
                torch.randn(self.image_size[1], self.image_size[2], 3).numpy() for _ in range(img_per_seq)
            ],
            DataField.POINT_MASKS: [
                torch.ones(self.image_size[1], self.image_size[2], dtype=torch.bool).numpy() for _ in range(img_per_seq)
            ],
            DataField.ORIGINAL_SIZES: [self.image_size[1:] for _ in range(img_per_seq)],
        }

        return sample


class MockEvalDataset(EvalDataset):
    """Mock test dataset for testing."""

    def __init__(self, length: int = 20, image_size: tuple = (3, 32, 32), **kwargs):
        super().__init__(**kwargs)  # Pass kwargs to EvalDataset
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def available_data_fields(self):
        return {
            DataField.IMAGES,
            DataField.EXTRINSICS_C2W,
            DataField.INTRINSICS,
            DataField.DEPTHS,
            DataField.WORLD_POINTS,
            DataField.POINT_MASKS,
        }

    def get_data(self, seq_index=None, data_fields=None):
        """Return a single test sample following the data contract."""
        C, H, W = self.image_size

        idx = 0 if seq_index is None else seq_index

        # Generate mock data following the data contract (lists of numpy arrays)
        return {
            DataField.SEQ_NAME: f"test_seq_{idx}",
            DataField.IDS: np.array([idx], dtype=np.int64),
            DataField.IMAGES: [np.random.randint(0, 256, (H, W, C), dtype=np.uint8)],  # List of [H, W, C]
            DataField.DEPTHS: [np.random.randn(H, W).astype(np.float32)],  # List of [H, W]
            DataField.EXTRINSICS_C2W: [np.eye(4, dtype=np.float32)],  # List of [4, 4]
            DataField.INTRINSICS: [np.eye(3, dtype=np.float32)],  # List of [3, 3]
            DataField.WORLD_POINTS: [np.random.randn(H, W, 3).astype(np.float32)],  # List of [H, W, 3]
            DataField.POINT_MASKS: [np.ones((H, W), dtype=bool)],  # List of [H, W]
            DataField.ORIGINAL_SIZES: [(H, W)],
        }


# Import centralized fixtures
pytest_plugins = ["tests.utils.fixtures"]


# Test-specific fixtures that can't be centralized
@pytest.fixture
def mock_data_module():
    """Create a mock data module for testing."""
    train_dataset = MockTrainDataset(length=50)
    test_dataset = MockEvalDataset(length=10)
    # Ensure datasets have a name attribute used by DataModule for keying
    train_dataset.name = "mock_train"
    test_dataset.name = "mock_test"

    # DataModule now expects dicts for datasets
    return DataModule(
        train_datasets={train_dataset.name: train_dataset},
        test_datasets={test_dataset.name: test_dataset},
        images_per_batch=4,
        images_per_element=(1, 3),
        aspect_ratios=(0.5, 2.0),
        train_num_workers=0,  # Avoid multiprocessing in tests
        test_num_workers=0,
    )


# Integration tests


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTrainerIntegration:
    """Integration tests for the complete trainer pipeline."""

    def test_basic_training_loop(self, mock_data_module, basic_trainer_config, temp_output_dir):
        """Test basic training loop with comprehensive state verification."""
        model = MockModule()
        initial_state = model.get_model_state_dict()

        trainer = Trainer(
            model=model,
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_basic",
            **basic_trainer_config,
        )

        # Test initial trainer state
        assert trainer.accelerator is None

        trainer.setup(mode="train")

        # Verify trainer state after setup
        assert trainer.accelerator is not None
        assert trainer.output_dir.exists()

        # Test that configuration values are preserved
        assert trainer.max_train_steps == basic_trainer_config["max_train_steps"]
        assert trainer.validation_steps == basic_trainer_config["validation_steps"]
        assert trainer.learning_rate == basic_trainer_config["learning_rate"]
        assert trainer.gradient_accumulation_steps == basic_trainer_config["gradient_accumulation_steps"]

        # Test that model was properly prepared by accelerator
        assert trainer.model is not None
        assert hasattr(trainer.model, "module") or hasattr(trainer.model, "model")

        trainer.fit()

        # Test optimizer configuration is correct
        optimizer = trainer.accelerator._optimizers[0]
        assert len(optimizer.param_groups) > 0
        param_group = optimizer.param_groups[0]
        assert "lr" in param_group
        assert param_group["lr"] == trainer.learning_rate

        # Verify model parameters changed meaningfully
        final_state = model.get_model_state_dict()
        assert not model.compare_model_states(
            initial_state, final_state
        ), "Model parameters should have changed during training"

        # Verify meaningful parameter changes (not just numerical noise)
        total_param_change = 0.0
        for name in initial_state:
            initial_param = initial_state[name].cpu()
            final_param = final_state[name].cpu()
            param_change = torch.norm(final_param - initial_param).item()
            total_param_change += param_change

        min_expected_change = 1e-4  # Should have meaningful learning
        assert (
            total_param_change > min_expected_change
        ), f"Total parameter change {total_param_change} too small, expected > {min_expected_change}"

        # Verify output artifacts were created
        final_model_path = os.path.join(trainer.output_dir, "final_model")
        assert os.path.exists(final_model_path)

        log_file = os.path.join(trainer.output_dir, "train.log")
        assert os.path.exists(log_file), "Training log should be created"

        # Test that trainer state remains consistent after training
        assert trainer.accelerator is not None
        assert trainer.max_train_steps == basic_trainer_config["max_train_steps"]

    def test_training_with_validation(self, mock_data_module, temp_output_dir):
        """Test training with validation steps."""
        config = {
            "max_train_steps": 9,
            "validation_steps": 3,  # Run validation every 3 steps
            "validation_batches": 2,
            "checkpointing_steps": 20,  # No checkpointing for this test
            "learning_rate": 1e-3,
            "tqdm": False,
            "experiment_logger": (),
        }

        model = MockModule()
        trainer = Trainer(
            model=model, data=mock_data_module, output_dir=temp_output_dir, experiment_name="test_validation", **config
        )

        trainer.setup(mode="train")

        # Track initial validation state
        initial_test_step_count = model.test_step_count

        trainer.fit()

        # Check that validation directories were created (trainer keeps only the most recent one)
        validation_dirs = [d for d in os.listdir(trainer.output_dir) if d.startswith("validation-")]

        # Should have validation at steps 3, 6, and 9 (3 total), but trainer only keeps the latest
        expected_validation_runs = 3
        assert (
            len(validation_dirs) == 1
        ), f"Expected 1 validation dir (latest), got {len(validation_dirs)}. Dirs: {validation_dirs}"

        # Verify the final validation directory exists and is named correctly
        final_validation_dir = validation_dirs[0]
        assert (
            final_validation_dir == "validation-9"
        ), f"Expected final validation dir to be 'validation-9', got '{final_validation_dir}'"

        # For each validation run, we should have processed some test steps
        # The exact number depends on the dataset size and validation_batches setting
        total_test_steps = model.test_step_count - initial_test_step_count
        assert total_test_steps > 0, "Should have processed some validation batches"

        # With validation_batches=2, we should process at most 2 batches per validation run
        max_expected_steps = expected_validation_runs * config["validation_batches"]
        assert (
            total_test_steps <= max_expected_steps
        ), f"Processed {total_test_steps} test steps, but expected at most {max_expected_steps}"

    def test_checkpointing_and_resume(self, mock_data_module, temp_output_dir):
        """Test checkpointing and resuming from checkpoint."""
        torch.manual_seed(42)
        np.random.seed(42)

        config = {
            "max_train_steps": 8,
            "validation_steps": 50,  # No validation
            "checkpointing_steps": 4,
            "checkpoints_total_limit": 3,
            "learning_rate": 1e-2,
            "tqdm": False,
            "experiment_logger": (),
        }

        # First training run
        model1 = MockModule()
        initial_state = model1.get_model_state_dict()

        trainer1 = Trainer(
            model=model1,
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_checkpoint",
            timestamp="2025-01-01_12-00-00",
            **config,
        )

        trainer1.setup(mode="train")
        trainer1.fit()

        # Verify checkpoints were created
        checkpoints = trainer1.get_checkpoints()
        assert len(checkpoints) > 0

        # Test resuming from checkpoint
        target_checkpoint = checkpoints[0] if checkpoints else "checkpoint-4"

        # Create fresh model with same initial state
        model2 = MockModule()
        with torch.no_grad():
            for (name1, _), (_, param2) in zip(
                model1.model.named_parameters(), model2.model.named_parameters(), strict=False
            ):
                initial_param = initial_state[name1].to(param2.device)
                param2.copy_(initial_param)

        trainer2 = Trainer(
            model=model2,
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_checkpoint",
            timestamp="2025-01-01_12-00-00",
            resume_from_checkpoint=target_checkpoint,
            **config,
        )

        trainer2.setup(mode="train")
        assert trainer2.resume_from_checkpoint == target_checkpoint

        # Train a few more steps after resuming
        short_config = config.copy()
        short_config["max_train_steps"] = 10  # Train a bit more

        trainer3 = Trainer(
            model=MockModule(),
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_checkpoint",
            timestamp="2025-01-01_12-00-00",
            resume_from_checkpoint=target_checkpoint,
            **short_config,
        )

        trainer3.setup(mode="train")
        trainer3.fit()
        resumed_final_state = trainer3.model.get_model_state_dict()

        # Verify that resumed training produces different results than fresh training
        fresh_model_state = MockModule().get_model_state_dict()
        assert not trainer3.model.compare_model_states(
            resumed_final_state, fresh_model_state
        ), "Resumed model should be different from fresh initialization"

    def test_different_model_configurations(self, mock_data_module, temp_output_dir):
        """Test trainer with different model configurations."""
        # Test with parameter freezing
        model_with_freeze = MockModule(
            freeze=["layer1.weight"],  # Freeze the first layer's weight
            log_params=["layer2.weight"],  # Log the second layer's weight
        )

        # Get initial parameter states
        initial_frozen_param = model_with_freeze.model.layer1.weight.clone().detach()
        initial_trainable_param = model_with_freeze.model.layer2.weight.clone().detach()

        config = {
            "max_train_steps": 6,
            "validation_steps": 20,  # No validation
            "checkpointing_steps": 20,  # No checkpointing
            "learning_rate": 1e-3,
            "tqdm": False,
            "experiment_logger": (),
        }

        trainer = Trainer(
            model=model_with_freeze,
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_freeze",
            **config,
        )

        trainer.setup(mode="train")
        trainer.fit()

        # Check that frozen parameters didn't receive gradients
        frozen_param = model_with_freeze.model.layer1.weight
        assert not frozen_param.requires_grad

        # Check that non-frozen parameters still require gradients
        trainable_param = model_with_freeze.model.layer2.weight
        assert trainable_param.requires_grad

        # Verify frozen parameter values didn't change during training
        final_frozen_param = model_with_freeze.model.layer1.weight.clone().detach()
        # Move both tensors to CPU for comparison to handle device differences
        initial_frozen_cpu = initial_frozen_param.cpu()
        final_frozen_cpu = final_frozen_param.cpu()
        assert torch.equal(initial_frozen_cpu, final_frozen_cpu), "Frozen parameter should not change during training"

        # Verify trainable parameter values did change during training
        final_trainable_param = model_with_freeze.model.layer2.weight.clone().detach()
        # Move both tensors to CPU for comparison
        initial_trainable_cpu = initial_trainable_param.cpu()
        final_trainable_cpu = final_trainable_param.cpu()
        param_change = torch.norm(final_trainable_cpu - initial_trainable_cpu).item()
        min_expected_change = 1e-5
        assert (
            param_change > min_expected_change
        ), f"Trainable parameter change {param_change} too small, expected > {min_expected_change}"

    def test_testing_mode(self, mock_data_module, temp_output_dir):
        """Test the testing/evaluation mode."""
        # First train a model
        config = {
            "max_train_steps": 5,
            "validation_steps": 20,
            "checkpointing_steps": 5,
            "tqdm": False,
            "experiment_logger": (),
        }

        mock_model = MockModule()
        trainer = Trainer(
            model=mock_model, data=mock_data_module, output_dir=temp_output_dir, experiment_name="test_eval", **config
        )

        trainer.setup(mode="train")
        trainer.fit()

        # Verify model was trained (parameters changed from initialization)
        trained_params = mock_model.get_model_state_dict()
        fresh_model = MockModule()
        fresh_params = fresh_model.get_model_state_dict()

        assert not mock_model.compare_model_states(
            trained_params, fresh_params
        ), "Model should have changed during training"

        # Now test the model in evaluation mode
        test_trainer = Trainer(
            model=mock_model,  # Use same trained model instance
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_eval",
            **config,
        )

        test_trainer.setup(mode="test")
        assert test_trainer.accelerator is not None

        # Run actual test and verify results
        initial_test_step_count = mock_model.test_step_count

        # Simulate test steps - access data module through trainer's internal data attribute
        # or use the mock_data_module directly since they share the same instance
        test_loader = next(iter(mock_data_module.test_dataloaders().values()))
        batch_count = 0
        for batch in test_loader:

            batch = send_to_device(batch, test_trainer.accelerator.device)
            predictions = test_trainer.model.test_step(batch, test_trainer.accelerator)
            # test_step now returns just predictions
            assert isinstance(predictions, dict), "Test step should return predictions dict"
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break

        # Verify test counters were updated
        assert mock_model.test_step_count > initial_test_step_count, "Test step count should increase"

    def test_gradient_accumulation(self, mock_data_module, temp_output_dir):
        """Test gradient accumulation functionality."""
        grad_accum_steps = 3
        max_train_steps = 6  # Should result in 2 optimizer steps (6/3)

        config = {
            "max_train_steps": max_train_steps,
            "validation_steps": 20,  # No validation
            "checkpointing_steps": 20,  # No checkpointing
            "gradient_accumulation_steps": grad_accum_steps,
            "learning_rate": 1e-3,
            "tqdm": False,
            "experiment_logger": (),
        }

        # Test with gradient accumulation
        model_with_accum = MockModule()
        trainer_with_accum = Trainer(
            model=model_with_accum,
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_grad_accum",
            **config,
        )

        trainer_with_accum.setup(mode="train")

        # Capture initial model state
        initial_state_accum = model_with_accum.get_model_state_dict()

        trainer_with_accum.fit()
        final_state_accum = model_with_accum.get_model_state_dict()

        # Test without gradient accumulation (equivalent effective batch size)
        config_no_accum = config.copy()
        config_no_accum["gradient_accumulation_steps"] = 1
        config_no_accum["max_train_steps"] = 2  # 2 optimizer steps to match the accumulated version

        model_no_accum = MockModule()
        trainer_no_accum = Trainer(
            model=model_no_accum,
            data=mock_data_module,
            output_dir=os.path.join(temp_output_dir, "no_accum"),
            experiment_name="test_no_grad_accum",
            **config_no_accum,
        )

        trainer_no_accum.setup(mode="train")

        # Set same initial state for fair comparison
        with torch.no_grad():
            for (name1, _), (_, param2) in zip(
                model_with_accum.model.named_parameters(), model_no_accum.model.named_parameters(), strict=False
            ):
                # Move the initial state to the same device as the parameter
                initial_param = initial_state_accum[name1].to(param2.device)
                param2.copy_(initial_param)

        trainer_no_accum.fit()
        final_state_no_accum = model_no_accum.get_model_state_dict()

        # Both should complete without errors
        assert os.path.exists(os.path.join(trainer_with_accum.output_dir, "final_model"))
        assert os.path.exists(os.path.join(trainer_no_accum.output_dir, "final_model"))

        # Verify both models actually changed from initial state
        assert not model_with_accum.compare_model_states(
            initial_state_accum, final_state_accum
        ), "Model with gradient accumulation should have changed from initial state"

        assert not model_no_accum.compare_model_states(
            initial_state_accum, final_state_no_accum
        ), "Model without gradient accumulation should have changed from initial state"

        # Verify the effective batch size concept: both training regimes should produce
        # similar (but not identical) final model states since they process the same
        # total amount of data but with different batching strategies
        # We don't expect them to be identical due to different gradient computation order,
        # but they should both have learned something meaningful
        param_diffs = []
        for name in initial_state_accum:
            # Move all tensors to CPU for comparison to handle device differences
            initial_param = initial_state_accum[name].cpu()
            final_accum_param = final_state_accum[name].cpu()
            final_no_accum_param = final_state_no_accum[name].cpu()

            diff_accum = torch.norm(final_accum_param - initial_param)
            diff_no_accum = torch.norm(final_no_accum_param - initial_param)
            param_diffs.append((diff_accum.item(), diff_no_accum.item()))

        # Both should have meaningful parameter changes (not too small)
        min_change_threshold = 1e-6
        for accum_change, no_accum_change in param_diffs:
            assert accum_change > min_change_threshold, f"Gradient accumulation model change too small: {accum_change}"
            assert (
                no_accum_change > min_change_threshold
            ), f"No gradient accumulation model change too small: {no_accum_change}"

    @pytest.mark.slow
    def test_large_scale_full_training_pipeline(self, mock_data_module, temp_output_dir):
        """Test a complete training pipeline with all advanced features: mixed precision, checkpoint limits, validation, gradient accumulation, LR scheduling."""
        config = {
            "max_train_steps": 12,  # More steps for better checkpoint limit testing
            "validation_steps": 3,
            "validation_batches": 3,
            "checkpointing_steps": 3,
            "checkpoints_total_limit": 2,  # Test checkpoint limit enforcement
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-3,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 3,
            "max_grad_norm": 1.0,
            "mixed_precision": "bf16",  # Test mixed precision training
            "tqdm": False,
            "experiment_logger": (),
        }

        mock_model = MockModule()
        initial_state = mock_model.get_model_state_dict()

        trainer = Trainer(
            model=mock_model,
            data=mock_data_module,
            output_dir=temp_output_dir,
            experiment_name="test_full_pipeline",
            **config,
        )

        trainer.setup(mode="train")

        # Verify mixed precision configuration
        assert trainer.mixed_precision == "bf16"
        accelerator_precision = trainer.accelerator.mixed_precision
        assert (
            accelerator_precision == "bf16"
        ), f"Expected accelerator mixed_precision='bf16', got {accelerator_precision}"

        # Verify that model parameters are in expected precision after preparation
        for name, param in trainer.model.model.named_parameters():
            if "weight" in name or "bias" in name:
                # Parameters should maintain their original dtype (usually float32)
                # Mixed precision affects forward pass, not parameter storage
                assert param.dtype in [
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ], f"Parameter {name} has unexpected dtype: {param.dtype}"

        trainer.fit()

        # Verify training completed with meaningful parameter changes
        final_state = mock_model.get_model_state_dict()
        assert not mock_model.compare_model_states(
            initial_state, final_state
        ), "Model parameters should have changed during mixed precision training"

        # Verify meaningful parameter changes occurred
        total_param_change = 0.0
        for name in initial_state:
            initial_param = initial_state[name].cpu()
            final_param = final_state[name].cpu()
            param_change = torch.norm(final_param - initial_param).item()
            total_param_change += param_change

        min_expected_change = 1e-4
        assert (
            total_param_change > min_expected_change
        ), f"Mixed precision training param change {total_param_change} too small, expected > {min_expected_change}"

        # Check all expected outputs
        assert os.path.exists(os.path.join(trainer.output_dir, "final_model"))
        assert os.path.exists(os.path.join(trainer.output_dir, "train.log"))

        # Test checkpoint limit enforcement
        checkpoints = trainer.get_checkpoints()
        assert len(checkpoints) > 0
        assert (
            len(checkpoints) <= config["checkpoints_total_limit"]
        ), f"Expected at most {config['checkpoints_total_limit']} checkpoints, got {len(checkpoints)}"

        # Verify the correct checkpoints are kept (should be the latest ones)
        # With max_train_steps=12 and checkpointing_steps=3, we should have checkpoints at steps 3, 6, 9, 12
        # With limit=2, we should keep only the latest 2: checkpoint-9 and checkpoint-12
        actual_checkpoints = sorted(checkpoints)

        # Verify checkpoint names match expected pattern
        for checkpoint in actual_checkpoints:
            assert checkpoint.startswith("checkpoint-"), f"Invalid checkpoint name format: {checkpoint}"

        # Verify only the latest checkpoints are kept
        if len(actual_checkpoints) == 2:
            # Extract step numbers and verify they're the highest ones
            steps = []
            for checkpoint in actual_checkpoints:
                step = int(checkpoint.split("-")[1])
                steps.append(step)

            steps.sort()
            # Should be the two highest step numbers
            assert steps[-1] == 12, f"Expected final checkpoint at step 12, got {steps[-1]}"
            assert steps[-2] >= 6, f"Expected second-to-last checkpoint at step 6 or higher, got {steps[-2]}"

        # Verify checkpoint directories actually exist and contain files
        for checkpoint in actual_checkpoints:
            checkpoint_path = os.path.join(trainer.output_dir, checkpoint)
            assert os.path.exists(checkpoint_path), f"Checkpoint directory {checkpoint} does not exist"
            assert os.path.isdir(checkpoint_path), f"Checkpoint {checkpoint} is not a directory"

            # Check that checkpoint contains some files (accelerator saves multiple files)
            checkpoint_files = os.listdir(checkpoint_path)
            assert len(checkpoint_files) > 0, f"Checkpoint {checkpoint} directory is empty"

        # Check that validation directories were created
        validation_dirs = [d for d in os.listdir(trainer.output_dir) if d.startswith("validation-")]
        # Should have validation at steps 3, 6, 9, 12 since validation_steps=3
        assert len(validation_dirs) >= 1, "At least one validation run should have occurred"

        # Test invalid mixed precision mode should be caught in trainer initialization
        with pytest.raises(AssertionError):
            Trainer(
                model=MockModule(),
                data=mock_data_module,
                output_dir=temp_output_dir,
                mixed_precision="invalid_precision",  # Should trigger assertion
            )

    def test_error_handling(self, mock_data_module, temp_output_dir):
        """Test error handling with invalid configurations."""

        # Test with model that produces NaN loss
        class NaNModel(MockModule):
            def train_step(self, batch: dict, step: int, accelerator=None):
                loss, pbar, tracker, predictions = super().train_step(batch, step, accelerator)
                # Force NaN loss to test error handling
                return torch.tensor(float("nan")), pbar, tracker, predictions

        config = {
            "max_train_steps": 5,
            "validation_steps": 20,
            "checkpointing_steps": 20,
            "learning_rate": 1e-3,
            "tqdm": False,
            "experiment_logger": (),
        }

        trainer = Trainer(
            model=NaNModel(), data=mock_data_module, output_dir=temp_output_dir, experiment_name="test_nan", **config
        )

        trainer.setup(mode="train")

        # Should raise ValueError due to NaN loss
        with pytest.raises(ValueError, match="Train Loss is not finite"):
            trainer.fit()
