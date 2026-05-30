# Testing Guide

## 🚀 Quick Start

```bash
# Run all appropriate tests for your hardware (default)
python tests/run_tests.py

# Run specific categories
python tests/run_tests.py -c cpu              # CPU tests only
python tests/run_tests.py -c gpu              # GPU tests (requires CUDA)
python tests/run_tests.py -c integration      # Integration tests (requires GPU)
python tests/run_tests.py -c distributed      # Distributed tests (requires multiple GPUs)

# See all available categories
python tests/run_tests.py --list
```

## 📋 Test Categories

The system uses **explicit pytest markers** to categorize tests:

### Available Categories

| Category | Command | Marker Required | Hardware | Description |
|----------|---------|----------------|----------|-------------|
| **cpu** | `pytest tests/ -m "not gpu and not distributed and not slow and not integration"` | None | Any CPU | Fast CPU-only tests |
| **gpu** | `pytest tests/ -m "gpu and not distributed and not integration"` | `@pytest.mark.gpu` | NVIDIA GPU | GPU tests (excluding integration) |
| **integration** | `pytest tests/ -m "integration and not distributed"` | `@pytest.mark.integration` | NVIDIA GPU | End-to-end tests |
| **slow** | `pytest tests/ -m "slow"` | `@pytest.mark.slow` | Varies | Time-intensive tests |
| **distributed** | `accelerate launch --num_processes 2 -m pytest tests/ -m "distributed"` | `@pytest.mark.distributed` | 2+ GPUs | Multi-GPU tests |

### Default Behavior (Intelligent Hardware Detection)

When no category is specified, the system automatically runs appropriate tests:
- **CPU only**: Runs CPU tests
- **Single GPU**: Runs CPU + GPU + Integration tests
- **Multiple GPUs**: Runs All tests including Distributed

## 🎯 Test Markers

### Required Markers for Non-CPU Tests

```python
# CPU test (no marker needed - default)
def test_utility_function():
    """Fast CPU-only test."""
    pass

# GPU test
@pytest.mark.gpu
def test_device_computation():
    """Test requiring GPU acceleration."""
    pass

# Integration test (requires GPU for performance)
@pytest.mark.integration
def test_training_pipeline():
    """End-to-end integration test."""
    pass

# Distributed test
@pytest.mark.distributed
def test_multi_process_sync():
    """Test requiring multiple GPUs with accelerate launch."""
    pass

# Slow test
@pytest.mark.slow
def test_comprehensive_scenario():
    """Time-intensive comprehensive test."""
    pass
```

## 🚀 Development Workflow

**Recommended testing order:**
1. **Development**: `python tests/run_tests.py -c cpu` (fast feedback)
2. **Feature Testing**: `python tests/run_tests.py -c gpu` (when GPU available)
3. **Integration**: `python tests/run_tests.py -c integration` (end-to-end validation)
4. **Full Testing**: `python tests/run_tests.py` (all appropriate tests)

## 📝 Adding New Tests

**Choose appropriate marker:**
- No marker = CPU test (default)
- `@pytest.mark.gpu` = GPU required
- `@pytest.mark.integration` = End-to-end test (GPU needed for performance)
- `@pytest.mark.distributed` = Multi-GPU test
- `@pytest.mark.slow` = Time-intensive test

**Verify categorization:**
```bash
# Test your specific function
pytest tests/path/to/test_file.py::test_function_name -v

# Run the expected category
python tests/run_tests.py -c [expected_category]
```

## 📊 Example Test Structure

```python
import pytest
import torch
from src.dvlt.model.base import BaseModel

# CPU test (no marker)
def test_model_initialization():
    model = BaseModel()
    assert model is not None

# GPU test
@pytest.mark.gpu
def test_model_cuda_operations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = BaseModel().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    output = model(x)
    assert output.device.type == 'cuda'

# Integration test
@pytest.mark.integration
def test_training_integration():
    # ... comprehensive training test
    pass

# Distributed test
@pytest.mark.distributed
def test_multi_gpu_training():
    # ... distributed training test
    pass
```
