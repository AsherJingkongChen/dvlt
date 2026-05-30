# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dvlt.data.sampler import AdaptiveBatchSampler, DistributedAdaptiveBatchSampler
from tests.utils.fixtures import DummyDataset, compare_batches


# Import fixtures from centralized location
pytest_plugins = ["tests.utils.fixtures"]


def collate_fn(batch):
    """Simple collate function for testing.

    Each item in batch is a tuple of (index, image_num, aspect_ratio).
    Returns tensors of indices, image_nums, and aspect_ratios.
    """
    indices = torch.tensor([item[0] for item in batch])
    image_nums = torch.tensor([item[1] for item in batch])
    aspect_ratios = torch.tensor([item[2] for item in batch])
    return indices, image_nums, aspect_ratios


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete training scenarios."""

    def test_integration_training_loop_simulation(self, dummy_dataset, sampler_config):
        """Simulate a training loop with checkpointing and resume."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        loader = DataLoader(dummy_dataset, batch_sampler=sampler, collate_fn=collate_fn)

        # Simulate training for a few epochs
        training_batches = []

        for _ in range(3):
            epoch_batches = []

            for i, batch in enumerate(loader):
                epoch_batches.append(batch)
                if i >= 2:  # Just a few batches per epoch
                    break

            training_batches.append(epoch_batches)

        # Save sampler state after some training
        checkpoint_state = sampler.state_dict()

        # Create new sampler and resume from checkpoint
        sampler2 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        sampler2.load_state_dict(checkpoint_state)
        loader2 = DataLoader(dummy_dataset, batch_sampler=sampler2, collate_fn=collate_fn)

        # Continue training - should produce same results as if we continued with original loader
        original_batch = next(iter(loader))
        resumed_batch = next(iter(loader2))

        # Compare parameters (image_nums and aspect_ratios should match)
        assert torch.equal(original_batch[1], resumed_batch[1]), "Image nums should match"
        assert torch.equal(original_batch[2], resumed_batch[2]), "Aspect ratios should match"

    def test_integration_real_trainer_behavior_simulation(self, dummy_dataset, sampler_config):
        """Test that simulates exactly what the real trainer does."""
        try:
            accelerator = Accelerator()
        except Exception as e:
            pytest.skip(f"Failed to initialize Accelerator: {e}")

        # Create our custom sampler and dataloader (like data_module.train_dataloader does)
        if accelerator.num_processes > 1:
            sampler = DistributedAdaptiveBatchSampler(
                dummy_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, **sampler_config
            )
        else:
            sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Set even_batches=False for custom batch samplers (like module.py does)
        accelerator.even_batches = False
        accelerator.register_for_checkpointing(sampler)

        train_dataloader = DataLoader(dummy_dataset, batch_sampler=sampler, collate_fn=collate_fn)
        train_dataloader = accelerator.prepare(train_dataloader)

        # Simulate the training loop
        global_step = 0
        max_steps = 10  # Small number for testing

        # Simulate the trainer's training loop
        while global_step < max_steps:
            batch_count = 0
            for batch in train_dataloader:
                # Simulate training step
                batch_count += 1
                global_step += 1

                # Test that batch is properly on device
                if torch.is_tensor(batch[0]):
                    batch_device = batch[0].device
                    target_device = accelerator.device

                    # Compare device types
                    if target_device.type == "cuda" and batch_device.type == "cuda":
                        assert True  # Both on CUDA is sufficient
                    else:
                        assert batch_device.type == target_device.type

                # Simulate checkpointing like the trainer (every 5 steps)
                if global_step % 5 == 0:
                    # Save state using accelerator like the trainer does
                    checkpoint_path = f"/tmp/test_checkpoint_{global_step}"
                    accelerator.save_state(checkpoint_path)

                    # Test that we can load the state back
                    accelerator.load_state(checkpoint_path)

                    # Clean up
                    if os.path.exists(checkpoint_path):
                        shutil.rmtree(checkpoint_path)

                if batch_count >= 3:  # Process a few batches per epoch
                    break

        try:
            accelerator.end_training()
        except Exception:
            pass

    def test_integration_distributed_training_simulation(self, dummy_dataset, sampler_config):
        """Simulate distributed training without requiring actual multi-GPU setup."""
        world_size = 2

        # Create samplers and loaders for different ranks
        loaders = []
        for rank in range(world_size):
            sampler = DistributedAdaptiveBatchSampler(
                dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config
            )
            loader = DataLoader(dummy_dataset, batch_sampler=sampler, collate_fn=collate_fn)
            loaders.append((loader, sampler))

        # Simulate training across multiple epochs
        for epoch in range(3):
            # Set epoch on all samplers (simulating distributed training)
            for _, sampler in loaders:
                sampler.set_epoch(epoch)

            # Get batches from each rank
            rank_batches = []
            for loader, _ in loaders:
                batches = []
                for j, batch in enumerate(loader):
                    if j >= 2:  # Just check a few batches
                        break
                    batches.append(batch)
                rank_batches.append(batches)

            # Verify synchronization across ranks
            if len(rank_batches) > 1:
                # All ranks should have same parameters for corresponding batches
                for batch_idx in range(min(len(rank_batches[0]), len(rank_batches[1]))):
                    batch_rank0 = rank_batches[0][batch_idx]
                    batch_rank1 = rank_batches[1][batch_idx]

                    # Parameters should be synchronized
                    assert torch.equal(
                        batch_rank0[1], batch_rank1[1]
                    ), f"Image nums not synchronized at epoch {epoch}, batch {batch_idx}"
                    assert torch.equal(
                        batch_rank0[2], batch_rank1[2]
                    ), f"Aspect ratios not synchronized at epoch {epoch}, batch {batch_idx}"

                    # But indices should be different
                    indices_rank0 = set(batch_rank0[0].tolist())
                    indices_rank1 = set(batch_rank1[0].tolist())
                    assert (
                        len(indices_rank0.intersection(indices_rank1)) == 0
                    ), f"Indices overlap at epoch {epoch}, batch {batch_idx}"

    def test_integration_end_to_end_determinism(self, dummy_dataset, sampler_config):
        """Test end-to-end determinism with checkpointing and resume."""
        # Run 1: Train for some epochs, then checkpoint
        sampler1 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        loader1 = DataLoader(dummy_dataset, batch_sampler=sampler1, collate_fn=collate_fn)

        # Train for 2 epochs
        run1_batches = []
        for _ in range(2):
            epoch_batches = []
            for i, batch in enumerate(loader1):
                epoch_batches.append(batch)
                if i >= 3:
                    break
            run1_batches.append(epoch_batches)

        # Save checkpoint
        checkpoint = sampler1.state_dict()

        # Run 2: Start fresh and train for the same 2 epochs
        sampler2 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        loader2 = DataLoader(dummy_dataset, batch_sampler=sampler2, collate_fn=collate_fn)

        run2_batches = []
        for _ in range(2):
            epoch_batches = []
            for i, batch in enumerate(loader2):
                epoch_batches.append(batch)
                if i >= 3:
                    break
            run2_batches.append(epoch_batches)

        # Should be identical up to the checkpoint point
        for epoch in range(2):
            for batch_idx in range(len(run1_batches[epoch])):
                assert compare_batches(
                    run1_batches[epoch][batch_idx], run2_batches[epoch][batch_idx]
                ), f"Batches differ at epoch {epoch}, batch {batch_idx}"

        # Now load checkpoint and continue
        sampler2.load_state_dict(checkpoint)
        loader2_resumed = DataLoader(dummy_dataset, batch_sampler=sampler2, collate_fn=collate_fn)

        # Continue training from checkpoint - should produce same parameters
        batch_continue_1 = next(iter(loader1))
        batch_continue_2 = next(iter(loader2_resumed))

        assert torch.equal(batch_continue_1[1], batch_continue_2[1]), "Image nums should match after resume"
        assert torch.equal(batch_continue_1[2], batch_continue_2[2]), "Aspect ratios should match after resume"

    def test_integration_mixed_sampler_scenarios(self, dummy_dataset, sampler_config):
        """Test scenarios mixing single and distributed samplers."""
        # Single sampler scenario
        single_sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        single_loader = DataLoader(dummy_dataset, batch_sampler=single_sampler, collate_fn=collate_fn)

        # Distributed sampler scenario (rank 0 of 1 - should behave similarly to single)
        distributed_sampler = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=1, rank=0, **sampler_config)
        distributed_loader = DataLoader(dummy_dataset, batch_sampler=distributed_sampler, collate_fn=collate_fn)

        # Both should work and produce valid batches
        # We explicitly set epoch to 5 here to test specific epoch behavior
        single_sampler.set_epoch(5)
        distributed_sampler.set_epoch(5)

        single_batch = next(iter(single_loader))
        distributed_batch = next(iter(distributed_loader))

        # Should have same structure
        assert len(single_batch) == len(distributed_batch) == 3

        # Should both respect the parameter ranges
        for batch in [single_batch, distributed_batch]:
            image_num = batch[1][0].item() if torch.is_tensor(batch[1][0]) else batch[1][0]
            aspect_ratio = batch[2][0].item() if torch.is_tensor(batch[2][0]) else batch[2][0]

            assert (
                sampler_config["frames_per_sample_range"][0]
                <= image_num
                <= sampler_config["frames_per_sample_range"][1]
            )
            assert sampler_config["aspect_ratio_bounds"][0] <= aspect_ratio <= sampler_config["aspect_ratio_bounds"][1]

    @pytest.mark.distributed
    def test_integration_accelerate(self, dummy_dataset, sampler_config):
        """
        Basic Accelerate integration test that works reliably in test suites.

        For comprehensive distributed testing, see tests/data/test_distributed_integration.py
        which should be run with: accelerate launch --num_processes 2
        """
        try:
            accelerator = Accelerator()
        except Exception as e:
            pytest.skip(f"Failed to initialize Accelerator: {e}")

        rank = accelerator.process_index
        world_size = accelerator.num_processes

        # Test both single and distributed scenarios
        if world_size > 1:
            sampler = DistributedAdaptiveBatchSampler(
                dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config
            )
        else:
            sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Set even_batches=False for custom batch samplers
        accelerator.even_batches = False
        accelerator.register_for_checkpointing(sampler)

        loader = DataLoader(dummy_dataset, batch_sampler=sampler, collate_fn=collate_fn)
        loader = accelerator.prepare(loader)

        # Test basic training loop
        for epoch in range(3):  # Shorter test for reliability
            batch_count = 0
            for batch in loader:
                batch_count += 1

                # Test device placement
                if torch.is_tensor(batch[0]):
                    batch_device = batch[0].device
                    target_device = accelerator.device

                    # Compare device types
                    if target_device.type == "cuda" and batch_device.type == "cuda":
                        assert True  # Both on CUDA is sufficient
                    else:
                        assert (
                            batch_device.type == target_device.type
                        ), f"Batch not on correct device: {batch_device} != {target_device}"

                if batch_count >= 5:  # Just test a few batches per epoch
                    break

            assert batch_count > 0, f"No batches generated at epoch {epoch}"

        # Test state dict functionality
        state_dict = sampler.state_dict()
        assert "epoch" in state_dict
        assert "num_remaining" in state_dict

        # Test that no manual set_epoch calls are needed for continuing
        batch = next(iter(loader))
        assert batch is not None

        try:
            accelerator.end_training()
        except Exception:
            pass

    def test_large_scale_simulation(self, dummy_dataset, sampler_config):
        """Test simulation of large-scale training scenario."""
        # Simulate larger dataset
        large_dataset = DummyDataset(length=10000)

        # Create sampler and loader
        sampler = AdaptiveBatchSampler(large_dataset, **sampler_config)
        loader = DataLoader(large_dataset, batch_sampler=sampler, collate_fn=collate_fn)

        # Simulate training for many epochs with periodic checkpointing
        checkpoints = {}
        completed_epochs = 0

        for epoch in range(5):  # Reduced to fewer epochs for clearer testing
            batch_count = 0
            for _ in loader:
                batch_count += 1
                if batch_count >= 20:  # Process some batches per epoch
                    break

            # Complete a full epoch by exhausting the iterator to auto-increment epoch
            if epoch % 2 == 1:  # Complete every other epoch
                for _ in loader:
                    pass  # Exhaust the remaining batches
                completed_epochs += 1

                # Save checkpoint after completed epochs
                checkpoints[sampler.epoch] = sampler.state_dict()

            assert batch_count > 0, f"No batches at epoch {epoch}"

        # Test resuming from different checkpoints
        for expected_epoch, checkpoint_state in checkpoints.items():
            # Create fresh sampler
            new_sampler = AdaptiveBatchSampler(large_dataset, **sampler_config)

            # Resume from checkpoint - this automatically restores epoch
            new_sampler.load_state_dict(checkpoint_state)
            assert new_sampler.epoch == expected_epoch, f"Expected epoch {expected_epoch}, got {new_sampler.epoch}"

            # Should be able to continue training
            new_loader = DataLoader(large_dataset, batch_sampler=new_sampler, collate_fn=collate_fn)
            batch = next(iter(new_loader))
            assert batch is not None
