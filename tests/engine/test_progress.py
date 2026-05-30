# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time


# Import centralized fixtures (includes mock_accelerate_logger and capture_stdout)
pytest_plugins = ["tests.utils.fixtures"]

# Import after fixtures to avoid initialization errors
from dvlt.engine.progress import ProgressBar  # noqa: E402


def test_basic_functionality(capture_stdout):
    """Test the basic functionality of the ProgressBar."""
    pb = ProgressBar(total=100)
    assert pb.n == 0  # Initial position is 0

    # Update and check position
    pos = pb.update(5)
    assert pos == 5
    assert pb.n == 5

    # Update more and check
    pos = pb.update(10)
    assert pos == 15
    assert pb.n == 15


def test_tqdm_behavior(mock_accelerate_logger, capture_stdout):
    """Test that the ProgressBar behaves like tqdm by default."""
    pb = ProgressBar(total=100)
    assert not pb.sequential_print_enabled  # Default is tqdm mode

    # Update several times
    for _ in range(5):
        pb.update(1)

    # In tqdm mode, logger should not be called
    mock_accelerate_logger.info.assert_not_called()


def test_sequential_print_mode(mock_accelerate_logger, capture_stdout):
    """Test sequential printing mode behavior."""
    pb = ProgressBar(total=100, use_tqdm=False, print_step_interval=5)
    assert pb.sequential_print_enabled

    # Update a few times (not enough to trigger print)
    pos = 0
    for _ in range(3):
        pos = pb.update(1)

    # Position should update but no log yet
    assert pos == 3
    assert pb.n == 3
    assert mock_accelerate_logger.info.call_count == 0

    # Update more to cross the interval threshold
    pos = pb.update(2)

    # Now should have logged
    assert pos == 5
    assert pb.n == 5
    assert mock_accelerate_logger.info.call_count == 1

    # One more round of updates
    for _ in range(5):
        pos = pb.update(1)

    # Should have logged again
    assert pos == 10
    assert pb.n == 10
    assert mock_accelerate_logger.info.call_count == 2


def test_initial_position(mock_accelerate_logger, capture_stdout):
    """Test starting from an initial position."""
    pb = ProgressBar(total=100, initial=20, use_tqdm=False)

    # Should start at the initial position
    assert pb.n == 20

    # Should have printed the initial state
    assert mock_accelerate_logger.info.call_count == 1
    call_args = mock_accelerate_logger.info.call_args[0][0]
    assert "20/100" in call_args

    # Reset the mock to focus on updates
    mock_accelerate_logger.reset_mock()

    # Update and check
    pos = pb.update(10)

    # Verify the position is correct
    assert pos == 30
    assert pb.n == 30

    # Should have logged again (with default print_step_interval=1)
    assert mock_accelerate_logger.info.call_count == 1
    call_args = mock_accelerate_logger.info.call_args[0][0]
    assert "30/100" in call_args


def test_postfix(mock_accelerate_logger, capture_stdout):
    """Test postfix functionality."""
    pb = ProgressBar(total=10, use_tqdm=False)

    # Set postfix and update
    pb.set_postfix(loss=0.5, acc=0.95)
    pb.update(1)

    # Check postfix in log
    call_args = mock_accelerate_logger.info.call_args[0][0]
    assert "loss=0.5" in call_args
    assert "acc=0.95" in call_args


def test_numeric_postfix_averaging(mock_accelerate_logger, capture_stdout):
    """Test that numeric postfix values are averaged over print_step_interval."""
    pb = ProgressBar(total=10, use_tqdm=False, print_step_interval=4)

    # Set different loss values over several updates
    loss_values = [0.8, 0.6, 0.4, 0.2]
    for loss in loss_values:
        pb.set_postfix(loss=loss)
        pb.update(1)

    # After 4 updates, should have triggered a print with the average loss
    assert mock_accelerate_logger.info.call_count == 1
    call_args = mock_accelerate_logger.info.call_args[0][0]

    # Average of [0.8, 0.6, 0.4, 0.2] is 0.5
    assert "loss=0.5" in call_args

    # Add more updates with different values
    loss_values = [0.3, 0.2, 0.1, 0.0]
    for loss in loss_values:
        pb.set_postfix(loss=loss)
        pb.update(1)

    # After 4 more updates, should have triggered another print
    assert mock_accelerate_logger.info.call_count == 2
    call_args = mock_accelerate_logger.info.call_args[0][0]

    # Average of [0.3, 0.2, 0.1, 0.0] is 0.15
    assert "loss=0.15" in call_args

    # Verify that non-numeric postfix values are not averaged
    pb = ProgressBar(total=10, use_tqdm=False, print_step_interval=2)
    pb.set_postfix(status="training")
    pb.update(1)
    pb.set_postfix(status="validating")
    pb.update(1)

    call_args = mock_accelerate_logger.info.call_args[0][0]
    # Should show the most recent value, not an average
    assert "status=validating" in call_args


def test_close_prints_final_state(mock_accelerate_logger, capture_stdout):
    """Test that close() prints the final state in sequential mode."""
    pb = ProgressBar(total=10, use_tqdm=False, print_step_interval=5)

    # Update a few times (not enough to trigger print)
    for _ in range(3):
        pb.update(1)

    # No logs yet
    assert mock_accelerate_logger.info.call_count == 0

    # Close should trigger a final print
    pb.close()

    # Should have logged the final state
    assert mock_accelerate_logger.info.call_count == 1


# Simple demo script
if __name__ == "__main__":
    print("Demonstrating ProgressBar with sequential printing")

    # Set up logging manually for the demo
    import logging

    print("Using standard logger (accelerate not initialized)")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Patch the logger to use our local logger
    import dvlt.engine.progress

    dvlt.engine.progress.logger = logging.getLogger()

    # Standard tqdm mode
    print("\nStandard tqdm mode:")
    pb1 = ProgressBar(total=20, desc="tqdm mode")
    for i in range(20):
        time.sleep(0.05)
        loss = 1.0 - (i / 20)
        pb1.set_postfix(loss=loss)
        curr_pos = pb1.update(1)
    pb1.close()

    # Sequential print mode
    print("\nSequential print mode (update every 4 steps):")
    pb2 = ProgressBar(total=20, use_tqdm=False, print_step_interval=4, desc="Sequential")
    for i in range(20):
        time.sleep(0.05)
        loss = 1.0 - (i / 20)
        pb2.set_postfix(loss=loss)
        curr_pos = pb2.update(1)
    pb2.close()

    # With initial position
    print("\nResuming from position 10:")
    pb3 = ProgressBar(total=20, initial=10, use_tqdm=False, desc="Resumed")
    for _ in range(10):
        time.sleep(0.05)
        curr_pos = pb3.update(1)
    pb3.close()
