# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Simple test runner for executing tests with proper markers and distributed setup.

Usage:
    python tests/run_tests.py                    # Run all appropriate tests for your hardware
    python tests/run_tests.py -c cpu             # CPU tests only
    python tests/run_tests.py -c gpu             # GPU tests
    python tests/run_tests.py -c integration     # Integration tests
    python tests/run_tests.py -c distributed     # Distributed tests (requires accelerate)
    python tests/run_tests.py -c slow            # Slow tests
    python tests/run_tests.py --list             # List available categories
"""

import argparse
import subprocess
import sys

import torch


def check_environment():
    """Check the test environment and print information."""
    print("🔍 Test Environment")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, check=False)

    print("-" * 50)
    if result.returncode == 0:
        print("✅ Tests passed!")
        return True
    else:
        print("❌ Tests failed!")
        return False


def get_test_commands():
    """Get available test commands."""
    base_cmd = [sys.executable, "-m", "pytest", "tests/"]

    commands = {
        "cpu": {
            "cmd": base_cmd + ["-m", "not gpu and not distributed and not slow and not integration"],
            "description": "CPU-only tests (fast, no GPU required)",
            "requires_gpu": False,
            "requires_distributed": False,
        },
        "gpu": {
            "cmd": base_cmd + ["-m", "gpu and not distributed and not integration"],
            "description": "GPU tests (single GPU, not integration)",
            "requires_gpu": True,
            "requires_distributed": False,
        },
        "integration": {
            "cmd": base_cmd + ["-m", "integration and not distributed"],
            "description": "Integration tests (requires GPU for performance)",
            "requires_gpu": True,
            "requires_distributed": False,
        },
        "slow": {
            "cmd": base_cmd + ["-m", "slow"],
            "description": "Slow/comprehensive tests",
            "requires_gpu": False,  # May or may not need GPU depending on specific test
            "requires_distributed": False,
        },
        "distributed": {
            "cmd": ["accelerate", "launch", "--num_processes", "2", "-m", "pytest", "tests/", "-m", "distributed"],
            "description": "Distributed tests (requires accelerate launch)",
            "requires_gpu": True,
            "requires_distributed": True,
        },
    }

    return commands


def check_requirements(category, commands):
    """Check if requirements are met for a test category."""
    cmd_info = commands[category]

    if cmd_info["requires_gpu"] and not torch.cuda.is_available():
        print(f"⚠️  Warning: {category} tests require GPU but CUDA not available")
        return False

    if cmd_info["requires_distributed"] and torch.cuda.device_count() < 2:
        print(f"⚠️  Warning: {category} tests require multiple GPUs but only {torch.cuda.device_count()} available")
        return False

    return True


def list_categories():
    """List available test categories."""
    commands = get_test_commands()

    print("📋 Available Test Categories")
    print("=" * 50)

    for category, info in commands.items():
        gpu_req = "GPU required" if info["requires_gpu"] else "CPU compatible"
        dist_req = " + Multi-GPU" if info["requires_distributed"] else ""

        print(f"{category:12} - {info['description']}")
        print(f"{'':12}   Requirements: {gpu_req}{dist_req}")
        print(f"{'':12}   Command: {' '.join(info['cmd'])}")
        print()


def run_all_appropriate_tests():
    """Run all tests appropriate for the current hardware."""
    commands = get_test_commands()

    # Determine which categories to run based on hardware
    categories_to_run = ["cpu"]  # Always run CPU tests

    if torch.cuda.is_available():
        categories_to_run.extend(["gpu", "integration"])

        if torch.cuda.device_count() >= 2:
            categories_to_run.append("distributed")

    print("🚀 Running all appropriate tests for your hardware")
    print(f"Categories: {', '.join(categories_to_run)}")
    if torch.cuda.is_available():
        print(f"Hardware: {torch.cuda.device_count()} GPU(s) detected")
    else:
        print("Hardware: CPU only")
    print()

    # Run each category
    all_passed = True
    failed_categories = []

    for category in categories_to_run:
        if category not in commands:
            continue

        if not check_requirements(category, commands):
            print(f"⚠️  Skipping {category} tests - requirements not met")
            continue

        print(f"\n📋 Running {category.upper()} tests")
        cmd_info = commands[category]
        success = run_command(cmd_info["cmd"], cmd_info["description"])

        if not success:
            failed_categories.append(category)
            all_passed = False

    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    if all_passed:
        print("✅ All test categories passed!")
    else:
        print(f"❌ {len(failed_categories)} categories failed: {', '.join(failed_categories)}")

    return all_passed


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Simple test runner with marker-based categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all appropriate tests for your hardware
  %(prog)s -c cpu             # Run CPU tests only
  %(prog)s -c gpu             # GPU tests
  %(prog)s -c integration     # Integration tests (GPU accelerated)
  %(prog)s -c distributed     # Distributed tests (needs accelerate)
  %(prog)s --list             # Show available categories
        """,
    )

    parser.add_argument("--category", "-c", help="Test category to run (see --list for options)")
    parser.add_argument("--list", action="store_true", help="List available test categories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose pytest output")

    args = parser.parse_args()

    # Show environment info
    check_environment()

    # Handle list command
    if args.list:
        list_categories()
        return 0

    # Get available commands
    commands = get_test_commands()

    # Add verbose flag if requested
    if args.verbose:
        for cmd_info in commands.values():
            if "-v" not in cmd_info["cmd"]:
                cmd_info["cmd"].append("-v")

    # Determine what to run
    if args.category:
        if args.category not in commands:
            print(f"❌ Unknown category: {args.category}")
            print(f"Available categories: {', '.join(commands.keys())}")
            print("Use --list to see detailed information")
            return 1

        # Run specific category
        category = args.category

        # Check requirements
        if not check_requirements(category, commands):
            print("❌ Requirements not met, skipping tests")
            return 1

        # Run the tests
        cmd_info = commands[category]
        success = run_command(cmd_info["cmd"], cmd_info["description"])

        return 0 if success else 1
    else:
        # Default: run all appropriate tests for the hardware
        success = run_all_appropriate_tests()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
