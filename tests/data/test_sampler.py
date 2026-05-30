# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dvlt.data.sampler import AdaptiveBatchSampler, DistributedAdaptiveBatchSampler


# Import fixtures from centralized location
pytest_plugins = ["tests.utils.fixtures"]


class TestAdaptiveBatchSampler:
    """Test suite for AdaptiveBatchSampler functionality."""

    def test_basic_functionality(self, dummy_dataset, sampler_config):
        """Test basic AdaptiveBatchSampler functionality."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Test that sampler generates batches
        batches = list(sampler)
        assert len(batches) > 0

        # Test that each batch contains tuples of (index, image_num, aspect_ratio)
        first_batch = batches[0]
        assert len(first_batch) > 0
        assert len(first_batch[0]) == 4  # (index, image_num, aspect_ratio, sample_seed)

        # Test that all items in a batch have the same image_num and aspect_ratio
        image_nums = [item[1] for item in first_batch]
        aspect_ratios = [item[2] for item in first_batch]
        assert len(set(image_nums)) == 1  # All same
        assert len(set(aspect_ratios)) == 1  # All same

    def test_epoch_setting_changes_parameters(self, dummy_dataset, sampler_config):
        """Test that setting epoch changes the generated parameters."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Get first batch from epoch 0
        sampler.set_epoch(0)
        epoch_0_batch = next(iter(sampler))

        # Get first batch from epoch 1
        sampler.set_epoch(1)
        epoch_1_batch = next(iter(sampler))

        # Parameters should be different between epochs
        epoch_0_params = (epoch_0_batch[0][1], epoch_0_batch[0][2])
        epoch_1_params = (epoch_1_batch[0][1], epoch_1_batch[0][2])
        assert epoch_0_params != epoch_1_params

    def test_epoch_setting_is_deterministic(self, dummy_dataset, sampler_config):
        """Test that same epoch produces same parameters."""
        sampler1 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        sampler2 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Set both to same epoch
        sampler1.set_epoch(5)
        sampler2.set_epoch(5)

        # Should produce identical batches
        batch1 = next(iter(sampler1))
        batch2 = next(iter(sampler2))

        assert batch1 == batch2

    def test_parameter_ranges(self, dummy_dataset, sampler_config):
        """Test that generated parameters respect the specified ranges."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Check multiple batches to ensure parameters are within ranges
        for i, batch in enumerate(sampler):
            if i >= 10:  # Check first 10 batches
                break

            image_num = batch[0][1]
            aspect_ratio = batch[0][2]

            # Check ranges
            assert (
                sampler_config["frames_per_sample_range"][0]
                <= image_num
                <= sampler_config["frames_per_sample_range"][1]
            )
            assert sampler_config["aspect_ratio_bounds"][0] <= aspect_ratio <= sampler_config["aspect_ratio_bounds"][1]

            # Check batch size respects max_frames_per_gpu
            expected_batch_size = max(1, sampler_config["max_frames_per_gpu"] // image_num)
            assert len(batch) <= expected_batch_size

    def test_weighted_dataset_sampling(self, dummy_multi_dataset, sampler_config):
        """Test that weighted dataset sampling produces correct proportional distribution.

        Core behavior verified:
        - Indices are distributed across datasets proportionally to their weights
        - Higher weight = more samples from that dataset
        - All sampled indices are valid (within dataset bounds)
        """
        sampler = AdaptiveBatchSampler(dummy_multi_dataset, **sampler_config)
        repeat_factors = sampler.repeat_factors
        cumulative_sizes = dummy_multi_dataset.cumulative_sizes

        # Collect all indices by iterating through the sampler
        all_indices = []
        for batch in sampler:
            all_indices.extend([item[0] for item in batch])

        assert len(all_indices) > 0, "Sampler should produce at least some indices"
        sampled_indices = torch.tensor(all_indices)

        # Core check 1: All indices should be valid (within dataset bounds)
        assert sampled_indices.min() >= 0, "Indices should be non-negative"
        assert sampled_indices.max() < cumulative_sizes[-1], "Indices should be within dataset bounds"

        # Core check 2: Verify proportional distribution matches repeat factors
        # Calculate expected proportions from repeat factors
        total_repeat = sum(repeat_factors)
        expected_proportions = [r / total_repeat for r in repeat_factors]

        # Calculate actual proportions
        dataset_low = [0] + cumulative_sizes[:-1]
        actual_proportions = []
        for low, high in zip(dataset_low, cumulative_sizes, strict=False):
            count = ((sampled_indices >= low) & (sampled_indices < high)).sum().item()
            actual_proportions.append(count / len(sampled_indices))

        # Verify proportions match (with tolerance for random sampling)
        for i, (expected, actual) in enumerate(zip(expected_proportions, actual_proportions, strict=False)):
            assert abs(actual - expected) < 0.10, (
                f"Dataset {i}: actual proportion {actual:.3f} differs from "
                f"expected {expected:.3f} (repeat_factor={repeat_factors[i]:.2f})"
            )

        # Core check 3: Higher repeat factors should result in more samples
        # Dataset with highest repeat factor should have highest proportion
        max_repeat_idx = repeat_factors.argmax().item()
        assert actual_proportions[max_repeat_idx] == max(actual_proportions), (
            f"Dataset with highest repeat factor ({repeat_factors[max_repeat_idx]:.2f}) "
            f"should have highest proportion"
        )

    def test_state_dict_save_restore(self, dummy_dataset, sampler_config):
        """Test that state_dict saves and restores sampler state correctly.

        Core behavior verified:
        - state_dict contains required fields (epoch, num_remaining, param_rng_state)
        - After restore, sampler produces IDENTICAL batches (same indices, params)
        - Mid-epoch resume continues from exact position
        """
        sampler1 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        # Consume some batches to get mid-epoch state
        batches_before_checkpoint = []
        for i, batch in enumerate(sampler1):
            batches_before_checkpoint.append(batch)
            if i >= 2:
                break

        # Save state mid-epoch
        state = sampler1.state_dict()

        # Core check 1: state_dict contains required fields
        assert "epoch" in state, "state_dict must contain 'epoch'"
        assert "num_remaining" in state, "state_dict must contain 'num_remaining'"
        assert "param_rng_state" in state, "state_dict must contain 'param_rng_state'"
        assert state["num_remaining"] > 0, "Should have remaining indices mid-epoch"

        # Continue consuming from original sampler
        remaining_batches_1 = []
        for i, batch in enumerate(sampler1):
            remaining_batches_1.append(batch)
            if i >= 2:
                break

        # Create new sampler and restore state
        sampler2 = AdaptiveBatchSampler(dummy_dataset, **sampler_config)
        sampler2.load_state_dict(state)

        # Core check 2: epoch is restored correctly
        assert sampler2.epoch == state["epoch"], "Epoch should be restored"

        # Should produce IDENTICAL batches as remaining from sampler1
        remaining_batches_2 = []
        for i, batch in enumerate(sampler2):
            remaining_batches_2.append(batch)
            if i >= 2:
                break

        # Core check 3: Batches must be identical (same indices AND parameters)
        assert len(remaining_batches_1) == len(remaining_batches_2), "Should produce same number of batches"
        for i, (b1, b2) in enumerate(zip(remaining_batches_1, remaining_batches_2, strict=False)):
            assert len(b1) == len(b2), f"Batch {i}: different sizes"
            for j, (item1, item2) in enumerate(zip(b1, b2, strict=False)):
                assert item1[0] == item2[0], f"Batch {i}, item {j}: indices differ ({item1[0]} vs {item2[0]})"
                assert item1[1] == item2[1], f"Batch {i}, item {j}: image_nums differ"
                assert item1[2] == item2[2], f"Batch {i}, item {j}: aspect_ratios differ"

    def test_auto_epoch_increment(self, dummy_dataset, sampler_config):
        """Test that epoch auto-increments after exhausting batches."""
        sampler = AdaptiveBatchSampler(dummy_dataset, **sampler_config)

        initial_epoch = sampler.epoch

        # Exhaust all batches
        for _ in sampler:
            pass

        # Epoch should have incremented
        assert sampler.epoch == initial_epoch + 1, "Epoch should auto-increment after exhausting batches"


class TestDistributedAdaptiveBatchSampler:
    """Test suite for DistributedAdaptiveBatchSampler functionality."""

    def test_distributed_parameter_synchronization_across_ranks(self, dummy_dataset, sampler_config):
        """Test that distributed sampler synchronizes parameters across ranks."""
        # Test with simulated distributed setup
        world_size = 2

        # Create samplers for different ranks
        sampler_rank0 = DistributedAdaptiveBatchSampler(
            dummy_dataset, num_replicas=world_size, rank=0, **sampler_config
        )

        sampler_rank1 = DistributedAdaptiveBatchSampler(
            dummy_dataset, num_replicas=world_size, rank=1, **sampler_config
        )

        # Set same epoch for both
        epoch = 5
        sampler_rank0.set_epoch(epoch)
        sampler_rank1.set_epoch(epoch)

        # Get first batches
        batch_rank0 = next(iter(sampler_rank0))
        batch_rank1 = next(iter(sampler_rank1))

        # Should have same parameters but different indices
        assert batch_rank0[0][1] == batch_rank1[0][1], "Image nums should be synchronized"
        assert batch_rank0[0][2] == batch_rank1[0][2], "Aspect ratios should be synchronized"

        # But different dataset indices (non-overlapping)
        indices_rank0 = set(item[0] for item in batch_rank0)
        indices_rank1 = set(item[0] for item in batch_rank1)
        assert len(indices_rank0.intersection(indices_rank1)) == 0, "Indices should not overlap between ranks"

    def test_deterministic_across_epochs(self, dummy_dataset, sampler_config):
        """Test that distributed sampler is deterministic across epochs."""
        world_size = 2

        sampler = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=0, **sampler_config)

        # Test same epoch produces same results
        epoch = 3
        sampler.set_epoch(epoch)
        batch1 = next(iter(sampler))

        sampler.set_epoch(epoch)  # Reset to same epoch
        batch2 = next(iter(sampler))

        assert batch1 == batch2, "Same epoch should produce deterministic results"

    def test_different_epochs_produce_different_results(self, dummy_dataset, sampler_config):
        """Test that different epochs produce different parameters."""
        world_size = 2

        sampler = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=0, **sampler_config)

        # Get batches from different epochs
        sampler.set_epoch(0)
        batch_epoch_0 = next(iter(sampler))

        sampler.set_epoch(1)
        batch_epoch_1 = next(iter(sampler))

        # Should have different parameters
        params_0 = (batch_epoch_0[0][1], batch_epoch_0[0][2])
        params_1 = (batch_epoch_1[0][1], batch_epoch_1[0][2])
        assert params_0 != params_1, "Different epochs should produce different parameters"

    def test_distributed_rank_distribution(self, dummy_dataset, sampler_config):
        """Test that different ranks get different but complementary data."""
        world_size = 3
        epoch = 7

        samplers = []
        batches = []

        # Create samplers for all ranks
        for rank in range(world_size):
            sampler = DistributedAdaptiveBatchSampler(
                dummy_dataset, num_replicas=world_size, rank=rank, **sampler_config
            )
            sampler.set_epoch(epoch)
            samplers.append(sampler)

            # Get first batch from each rank
            batch = next(iter(sampler))
            batches.append(batch)

        # All ranks should have same parameters
        ref_image_num = batches[0][0][1]
        ref_aspect_ratio = batches[0][0][2]

        for i, batch in enumerate(batches):
            assert batch[0][1] == ref_image_num, f"Rank {i} has different image_num"
            assert batch[0][2] == ref_aspect_ratio, f"Rank {i} has different aspect_ratio"

        # But different, non-overlapping indices
        all_indices = set()
        for batch in batches:
            batch_indices = set(item[0] for item in batch)
            assert len(batch_indices.intersection(all_indices)) == 0, "Overlapping indices detected"
            all_indices.update(batch_indices)

    def test_weighted_dataset_sampling_distributed(self, dummy_multi_dataset, sampler_config):
        """Test that weighted dataset sampling maintains correct proportions across distributed ranks.

        Core behavior verified:
        - Each rank gets a subset of indices
        - Combined indices from all ranks have correct proportional distribution
        - Ranks don't overlap in their indices (verified in other tests)
        """
        world_size = 4

        # Create samplers for all ranks and collect their indices
        rank_indices = []
        for rank in range(world_size):
            sampler = DistributedAdaptiveBatchSampler(
                dummy_multi_dataset, num_replicas=world_size, rank=rank, **sampler_config
            )
            sampler.set_epoch(0)

            # Collect indices by iterating through the sampler
            indices = []
            for batch in sampler:
                indices.extend([item[0] for item in batch])
            rank_indices.append(indices)

        # Core check 1: Each rank should get some indices
        for rank, indices in enumerate(rank_indices):
            assert len(indices) > 0, f"Rank {rank} should receive indices"

        # Combine indices from all ranks
        combined_indices = torch.tensor([idx for indices in rank_indices for idx in indices])

        repeat_factors = dummy_multi_dataset.repeat_factors
        cumulative_sizes = dummy_multi_dataset.cumulative_sizes

        # Core check 2: All indices should be valid
        assert combined_indices.min() >= 0, "Indices should be non-negative"
        assert combined_indices.max() < cumulative_sizes[-1], "Indices should be within dataset bounds"

        # Core check 3: Verify proportional distribution matches repeat factors
        total_repeat = sum(repeat_factors)
        expected_proportions = [r / total_repeat for r in repeat_factors]

        dataset_low = [0] + cumulative_sizes[:-1]
        actual_proportions = []
        for low, high in zip(dataset_low, cumulative_sizes, strict=False):
            count = ((combined_indices >= low) & (combined_indices < high)).sum().item()
            actual_proportions.append(count / len(combined_indices))

        for i, (expected, actual) in enumerate(zip(expected_proportions, actual_proportions, strict=False)):
            assert abs(actual - expected) < 0.10, (
                f"Dataset {i}: combined proportion {actual:.3f} differs from "
                f"expected {expected:.3f} (repeat_factor={repeat_factors[i]:.2f})"
            )

    def test_distributed_state_dict_save_restore(self, dummy_dataset, sampler_config):
        """Test that state_dict saves and restores distributed sampler state correctly.

        Core behavior verified:
        - state_dict contains required fields
        - After restore, sampler produces IDENTICAL batches
        - Both ranks can independently save/restore and produce consistent results
        """
        world_size = 2

        # Test for rank 0
        sampler1 = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=0, **sampler_config)

        # Consume some batches
        for i, _ in enumerate(sampler1):
            if i >= 2:
                break

        # Save state mid-epoch
        state = sampler1.state_dict()

        # Core check 1: state_dict contains required fields
        assert "epoch" in state, "state_dict must contain 'epoch'"
        assert "num_remaining" in state, "state_dict must contain 'num_remaining'"
        assert "param_rng_state" in state, "state_dict must contain 'param_rng_state'"

        # Continue consuming from original sampler
        remaining_batches_1 = []
        for i, batch in enumerate(sampler1):
            remaining_batches_1.append(batch)
            if i >= 2:
                break

        # Create new sampler and restore state
        sampler2 = DistributedAdaptiveBatchSampler(dummy_dataset, num_replicas=world_size, rank=0, **sampler_config)
        sampler2.load_state_dict(state)

        # Core check 2: epoch is restored correctly
        assert sampler2.epoch == state["epoch"], "Epoch should be restored"

        # Should produce IDENTICAL batches
        remaining_batches_2 = []
        for i, batch in enumerate(sampler2):
            remaining_batches_2.append(batch)
            if i >= 2:
                break

        # Core check 3: Batches must be identical
        assert len(remaining_batches_1) == len(remaining_batches_2), "Should produce same number of batches"
        for i, (b1, b2) in enumerate(zip(remaining_batches_1, remaining_batches_2, strict=False)):
            assert len(b1) == len(b2), f"Batch {i}: different sizes"
            for j, (item1, item2) in enumerate(zip(b1, b2, strict=False)):
                assert item1[0] == item2[0], f"Batch {i}, item {j}: indices differ"
                assert item1[1] == item2[1], f"Batch {i}, item {j}: image_nums differ"
                assert item1[2] == item2[2], f"Batch {i}, item {j}: aspect_ratios differ"
