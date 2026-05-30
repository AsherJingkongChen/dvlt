# Contribution guidelines

If you contribute to the codebase, please follow these steps


## Development Setup

If you're contributing to the codebase, we recommend installing the development dependencies and setting up pre-commit hooks:

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Style

This project uses:
- Black for code formatting
- Ruff for linting and import sorting
- Google-style docstrings

You can manually run the formatters:
```bash
# Format code with Black
black src tests

# Run Ruff linter and auto-fix Ruff violations where possible
ruff check --fix src tests
```

### Running Tests

The project has a comprehensive test suite with smart hardware detection. For quick development feedback:

```bash
# Fast CPU-only tests (recommended for development)
python tests/run_tests.py --category cpu

# All tests appropriate for your hardware
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --category cpu           # CPU-only tests (fast, no GPU required)
python tests/run_tests.py --category gpu           # GPU tests (single GPU, not integration)
python tests/run_tests.py --category integration   # Integration tests (requires GPU for performance)
python tests/run_tests.py --category distributed   # Distributed tests (requires accelerate launch)
python tests/run_tests.py --category slow          # Slow/comprehensive tests

# List all available test categories
python tests/run_tests.py --list
```

For direct pytest usage:
```bash
# Run all tests (NOTE: Cannot run distributed tests)
pytest

# Run specific test file
pytest tests/path/to/test_file.py
```

See [docs/TESTING.md](TESTING.md) for comprehensive testing documentation.

The pre-commit hooks automatically run fast tests before each commit to ensure code quality.

## Sanity Check Mode

For quick development testing of training functionality, use the built-in sanity check mode (with your desired config):

```bash
# Instead of long training runs for testing
python -m dvlt.scripts.train --config-name dvlt-large.yaml trainer.sanity_check=True
```

This mode:
- **Tests all key functions**: train_step, test_step, log_train, log_test, model saving
- **Uses temporary directories**: No persistent output clutter
- **Mock loggers**: Prints logged keys/types instead of creating wandb runs
- **Fast execution**: Runs 11 training steps + 2 validation batches (~1-2 minutes)
- **Auto cleanup**: Removes temporary files when complete

Automatically overrides settings to:
- `max_train_steps=11`, `validation_steps=11`, `validation_batches=2`
- `log_every_n_steps=1`


## Design

Check [DATA.md](docs/data/DATA.md) for an overview of the data pipeline flow. We explain there:

- How to add new parsers for arbitrary new datasets
- How to maintain and extend the data preprocessing code
- What batch format the model will be fed during training and evaluation
