# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dvlt.data.loader import AccelerateDataLoader
from dvlt.data.sampler import AdaptiveBatchSampler, DistributedAdaptiveBatchSampler


# Import fixtures from centralized location
pytest_plugins = ["tests.utils.fixtures"]


class TestAccelerateDataLoader:
    """Test suite for AccelerateDataLoader functionality.

    AccelerateDataLoader provides automatic device placement without sharding,
    allowing custom distributed samplers to handle distribution themselves.
    State management (epoch, checkpointing) is handled by the sampler via
    accelerator.register_for_checkpointing().
    """

    def test_basic_dataloader_functionality(self, dummy_dataset, sampler_config):
        """Test AccelerateDataLoader with single GPU setup."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=sampler)

        # Test basic iteration
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 4:  # Just test a few batches
                break

        assert len(batches) == 5

        # Test that batches contain the expected structure
        first_batch = batches[0]
        assert len(first_batch) == 4  # (indices, image_nums, aspect_ratios, sample_seeds)

    def test_dataloader_length(self, dummy_dataset, sampler_config):
        """Test that dataloader length is properly reported."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=sampler)

        # Should have a length (even if it's the large number from sampler)
        assert len(loader) > 0

    def test_device_placement_none(self, dummy_dataset, sampler_config):
        """Test device placement with no device specified."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Create loader without device
        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=sampler, device=None)
        batch = next(iter(loader))

        # Batch should be on CPU (or whatever default device the data starts on)
        if isinstance(batch[0], torch.Tensor):
            assert not batch[0].is_cuda

    def test_device_placement_cpu(self, dummy_dataset, sampler_config):
        """Test explicit CPU device placement."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Test with CPU device explicitly
        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=sampler, device=torch.device("cpu"))
        batch = next(iter(loader))

        if isinstance(batch[0], torch.Tensor):
            assert batch[0].device.type == "cpu"

    def test_device_placement_cuda(self, dummy_dataset, sampler_config):
        """Test CUDA device placement if available."""
        if not torch.cuda.is_available():
            return  # Skip if CUDA not available

        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        cuda_device = torch.device("cuda:0")
        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=sampler, device=cuda_device)
        batch = next(iter(loader))

        if isinstance(batch[0], torch.Tensor):
            assert batch[0].device.type == "cuda"

    def test_distributed_loader_basic(self, dummy_dataset, sampler_config):
        """Test AccelerateDataLoader with distributed sampler."""
        # Test with distributed sampler (simulated)
        world_size = 2
        rank = 0

        distributed_sampler = DistributedAdaptiveBatchSampler(
            dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config
        )

        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=distributed_sampler)

        # Test basic functionality
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 2:
                break

        assert len(batches) == 3

    def test_non_blocking_transfer(self, dummy_dataset, sampler_config):
        """Test non-blocking device transfer functionality."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Test with non_blocking=True
        loader = AccelerateDataLoader(
            dummy_dataset,
            batch_sampler=sampler,
            device=torch.device("cpu"),
            non_blocking=True,
        )

        # Should work without errors
        batch = next(iter(loader))
        assert batch is not None

        # Test with non_blocking=False
        loader_blocking = AccelerateDataLoader(
            dummy_dataset, batch_sampler=sampler, device=torch.device("cpu"), non_blocking=False
        )

        batch_blocking = next(iter(loader_blocking))
        assert batch_blocking is not None

    def test_custom_dataloader_kwargs(self, dummy_dataset, sampler_config):
        """Test that custom DataLoader kwargs are properly passed through."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Test with custom collate_fn
        def custom_collate_fn(batch):
            return {"custom": batch}

        loader = AccelerateDataLoader(
            dummy_dataset,
            batch_sampler=sampler,
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))
        assert "custom" in batch, "Custom collate_fn not applied"

    def test_multiple_iterations(self, dummy_dataset, sampler_config):
        """Test that multiple iterations through the loader work correctly."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        loader = AccelerateDataLoader(dummy_dataset, batch_sampler=sampler)

        # First iteration
        first_batches = []
        for i, batch in enumerate(loader):
            first_batches.append(batch)
            if i >= 2:
                break

        # Second iteration (sampler auto-increments epoch)
        second_batches = []
        for i, batch in enumerate(loader):
            second_batches.append(batch)
            if i >= 2:
                break

        # Both should have batches
        assert len(first_batches) == 3
        assert len(second_batches) == 3
