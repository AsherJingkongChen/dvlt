# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Comprehensive distributed integration tests for the data loading pipeline.
These tests reproduce all key behaviors from reproduce_epoch_drift.py and should be run with:

    accelerate launch --num_processes 2 -m pytest tests/data/test_distributed_integration.py -v

This file contains the full-strength distributed tests without any compromises.
"""

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dvlt.data.sampler import DistributedAdaptiveBatchSampler


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


@pytest.mark.distributed
def test_distributed_comprehensive_integration(dummy_dataset, sampler_config):
    """
    Comprehensive distributed test - reproduces ALL behaviors from reproduce_epoch_drift.py.

    This test maintains full testing strength and should be run with accelerate launch.
    It tests:
    - Parameter synchronization across ranks
    - Non-overlapping dataset indices
    - Element count consistency
    - Long-term epoch drift prevention
    - Device placement
    - Real accelerator gather operations
    """
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Only run this test in true distributed environments
    if world_size < 2:
        pytest.skip("This test requires multiple processes - run with: accelerate launch --num_processes 2")

    # Create distributed setup exactly like reproduce_epoch_drift.py
    sampler = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config)

    # Register sampler for checkpointing
    accelerator.register_for_checkpointing(sampler)

    # IMPORTANT: Don't call accelerator.prepare(loader) when using a distributed batch sampler!
    # The sampler already handles distribution across ranks. Using prepare() would cause
    # accelerate to try to distribute already-distributed batches, resulting in each rank
    # seeing only half the expected data.
    loader = DataLoader(dummy_dataset, batch_sampler=sampler, collate_fn=collate_fn)

    if rank == 0:
        print(f"\n***** Comprehensive distributed test (world_size={world_size}) *****\n")

    # Test across many epochs like reproduce_epoch_drift.py
    epoch_counts = []

    for epoch in range(20):  # Long-term consistency test
        if rank == 0:
            print(f"🔍 Rank {rank}: Starting epoch {epoch}")

        im_nums = []
        aspect_ratios = []
        dataset_indices = []

        for batch in loader:
            # Move batch to device (since we don't use accelerator.prepare)
            batch = tuple(t.to(accelerator.device) if torch.is_tensor(t) else t for t in batch)

            # Verify device placement
            if torch.is_tensor(batch[0]):
                batch_device = batch[0].device
                target_device = accelerator.device
                assert batch_device.type == target_device.type, f"Device mismatch: {batch_device} != {target_device}"

            # Collect data exactly like reproduce_epoch_drift.py
            if torch.is_tensor(batch[1]):
                im_nums.extend(batch[1].tolist())
            else:
                im_nums.extend(batch[1] if isinstance(batch[1], (list, tuple)) else [batch[1]])

            if torch.is_tensor(batch[2]):
                aspect_ratios.extend(batch[2].tolist())
            else:
                aspect_ratios.extend(batch[2] if isinstance(batch[2], (list, tuple)) else [batch[2]])

            if torch.is_tensor(batch[0]):
                dataset_indices.extend(batch[0].tolist())
            else:
                dataset_indices.extend(batch[0] if isinstance(batch[0], (list, tuple)) else batch[0])

        epoch_counts.append(len(dataset_indices))

        if rank == 0:
            print(
                f"🔍 Rank {rank}: {len(dataset_indices)} batches, {len(torch.cat([torch.tensor(im_nums)]))} total elements"
            )

        # Test element count consistency across ranks (prevents deadlocks)
        total_elements = torch.tensor([len(im_nums)], device=accelerator.device)
        all_counts = accelerator.gather(total_elements)

        if rank == 0:
            assert torch.all(all_counts == all_counts[0]), f"❌ Uneven element counts: {all_counts.tolist()}"

        # Sync parameters across ranks exactly like reproduce_epoch_drift.py
        im_nums_tensor = torch.tensor(im_nums, dtype=torch.float32, device=accelerator.device)
        aspect_ratios_tensor = torch.tensor(aspect_ratios, dtype=torch.float32, device=accelerator.device)
        dataset_indices_tensor = torch.tensor(dataset_indices, dtype=torch.long, device=accelerator.device)

        gathered_im_nums = accelerator.gather(im_nums_tensor).view(world_size, -1)
        gathered_aspect_ratios = accelerator.gather(aspect_ratios_tensor).view(world_size, -1)
        gathered_indices = accelerator.gather(dataset_indices_tensor).view(world_size, -1)

        if rank == 0:
            # Verify parameter synchronization (like reproduce_epoch_drift.py)
            for i in range(1, world_size):
                assert torch.allclose(
                    gathered_im_nums[0], gathered_im_nums[i], rtol=1e-4
                ), f"❌ Rank {rank} has different im_nums with rank {i}"
                assert torch.allclose(
                    gathered_aspect_ratios[0], gathered_aspect_ratios[i], rtol=1e-4
                ), f"❌ Rank {rank} has different aspect_ratios with rank {i}"

            # Verify non-overlapping indices (like reproduce_epoch_drift.py)
            indices_per_rank = [gathered_indices[i].cpu().numpy().tolist() for i in range(world_size)]
            for i in range(1, world_size):
                overlap = set(indices_per_rank[0]) & set(indices_per_rank[i])
                assert len(overlap) == 0, f"❌ Rank overlap between rank 0 and rank {i}: {overlap}"

    # Test epoch drift prevention (like reproduce_epoch_drift.py)
    epoch_counts_tensor = torch.tensor(epoch_counts, device=accelerator.device)
    all_epoch_counts = accelerator.gather(epoch_counts_tensor)
    all_epoch_counts = all_epoch_counts.view(world_size, len(epoch_counts))

    if rank == 0:
        for r in range(world_size):
            print(f"rank {r} batches per epoch: {all_epoch_counts[r].tolist()}")

        # Check for drift across epochs
        diverged = False
        for epoch_idx in range(len(epoch_counts)):
            epoch_counts_this_epoch = all_epoch_counts[:, epoch_idx]
            unique_counts = torch.unique(epoch_counts_this_epoch)
            if len(unique_counts) > 1:
                diverged = True
                print(f"❌ Epoch drift detected at epoch {epoch_idx}: {epoch_counts_this_epoch.tolist()}")
                break

        if not diverged:
            print("✅ No epoch drift detected - batch counts identical across ranks")

        assert not diverged, "Epoch drift detected"

    accelerator.end_training()


@pytest.mark.distributed
def test_distributed_dataloader_state_management(dummy_dataset, sampler_config):
    """Test state management in distributed scenarios."""
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if world_size < 2:
        pytest.skip("This test requires multiple processes")

    sampler = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config)

    # Register sampler for checkpointing
    accelerator.register_for_checkpointing(sampler)

    # Don't call accelerator.prepare(loader) - sampler already handles distribution
    loader = DataLoader(dummy_dataset, batch_sampler=sampler, collate_fn=collate_fn)

    # Run a few epochs
    for epoch in range(3):
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 5:
                break
        assert batch_count > 0, f"No batches at epoch {epoch}"

    # Test state dict functionality
    state_dict = sampler.state_dict()
    assert "epoch" in state_dict
    assert "num_remaining" in state_dict

    # Test loading state
    new_sampler = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config)
    new_sampler.load_state_dict(state_dict)

    assert new_sampler.epoch == sampler.epoch

    accelerator.end_training()
