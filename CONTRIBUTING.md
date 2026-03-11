# Contributing to Robotic Simulator Benchmarks

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing to the simulator benchmarking suite.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Adding a New Simulator](#adding-a-new-simulator)
- [Creating Benchmark Scenarios](#creating-benchmark-scenarios)
- [Submitting Results](#submitting-results)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/sim-benchmarks.git
   cd sim-benchmarks
   ```
3. **Create a branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

- **Add new simulators** to the benchmarking suite
- **Create new benchmark scenarios** (manipulation, locomotion, etc.)
- **Improve existing benchmarks** with better metrics or implementations
- **Submit benchmark results** from your hardware
- **Fix bugs** or improve code quality
- **Enhance documentation** (tutorials, examples, API docs)
- **Add analysis tools** for visualizing and comparing results

## Adding a New Simulator

### Prerequisites

- The simulator should be actively maintained
- Must have Python bindings or a Python API
- Should support programmatic scene creation and control
- Must be reproducible (deterministic given same seed)

### Implementation Steps

1. **Create a simulator wrapper** in `simulators/<simulator_name>/`:
   ```
   simulators/
   └── <simulator_name>/
       ├── __init__.py
       ├── wrapper.py          # Main simulator interface
       ├── environment.py      # Environment implementations
       ├── utils.py            # Helper functions
       └── README.md           # Simulator-specific setup
   ```

2. **Implement the base interface**:
   ```python
   from simulators.base import BaseSimulator

   class YourSimulator(BaseSimulator):
       def __init__(self, config):
           """Initialize the simulator with config"""
           pass

       def reset(self, seed=None):
           """Reset environment to initial state"""
           pass

       def step(self, action):
           """Step simulation forward"""
           pass

       def get_state(self):
           """Return current state"""
           pass

       def close(self):
           """Cleanup resources"""
           pass
   ```

3. **Add installation instructions** in `simulators/<simulator_name>/README.md`:
   - Installation commands
   - Dependencies and versions
   - Platform-specific notes
   - License information

4. **Create benchmark implementations** for standard scenarios:
   - Pendulum swing-up
   - Cart-pole
   - Reach task
   - Multi-agent scenario

5. **Add tests** in `tests/test_<simulator_name>.py`:
   ```python
   def test_initialization():
       """Test simulator can initialize"""
       pass

   def test_determinism():
       """Test simulation is deterministic"""
       pass

   def test_performance():
       """Test basic performance metrics"""
       pass
   ```

6. **Update documentation**:
   - Add simulator to [README.md](README.md)
   - Update benchmark compatibility matrix
   - Add example usage

## Creating Benchmark Scenarios

### Benchmark Requirements

Each benchmark should:
- Be **reproducible** across simulators
- Have **clear success metrics**
- Test **specific capabilities**
- Include **multiple difficulty levels**
- Be **well-documented**

### Scenario Structure

```
benchmarks/
└── <scenario_name>/
    ├── __init__.py
    ├── scenario.py         # Scenario definition
    ├── assets/             # Models, meshes, URDFs
    ├── config.yaml         # Configuration parameters
    └── README.md           # Scenario description
```

### Example Scenario Implementation

```python
from benchmarks.base import BaseBenchmark

class YourBenchmark(BaseBenchmark):
    """
    Description: Brief description of what this benchmark tests

    Metrics:
    - Steps per second
    - Memory usage
    - Success rate

    Difficulty levels: easy, medium, hard
    """

    def setup(self, simulator, difficulty='medium'):
        """Setup the benchmark scenario"""
        pass

    def run(self, num_steps=1000):
        """Run the benchmark"""
        pass

    def evaluate(self):
        """Compute and return metrics"""
        pass
```

### Benchmark Documentation

Include in the scenario README:
- **Objective**: What the benchmark tests
- **Setup**: Initial conditions and parameters
- **Metrics**: What is measured and how
- **Expected ranges**: Typical performance ranges
- **Known issues**: Simulator-specific limitations

## Submitting Results

### Result Format

Results should be submitted as JSON:

```json
{
  "simulator": "mujoco",
  "version": "2.3.0",
  "benchmark": "cart_pole",
  "hardware": {
    "cpu": "Intel i9-12900K",
    "gpu": "NVIDIA RTX 4090",
    "ram": "32GB",
    "os": "Ubuntu 22.04"
  },
  "metrics": {
    "steps_per_second": 15000,
    "real_time_factor": 150,
    "memory_mb": 256,
    "success_rate": 0.95
  },
  "timestamp": "2026-03-11T10:30:00Z",
  "seed": 42
}
```

### Submitting Results

1. Run benchmarks using the standard scripts
2. Place results in `results/<simulator>/<benchmark>/`
3. Include system information and configuration
4. Submit via pull request with description of test conditions

## Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Required Development Tools

- **Python 3.8+**
- **pytest** for testing
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

### Running Tests

```bash
# Run all tests
pytest

# Run specific simulator tests
pytest tests/test_mujoco.py

# Run with coverage
pytest --cov=simulators --cov-report=html
```

## Style Guidelines

### Python Code Style

- Follow **PEP 8** conventions
- Use **type hints** for function signatures
- Write **docstrings** for all public functions (Google style)
- Keep functions **focused and concise**
- Maximum line length: **100 characters**

### Example

```python
def run_benchmark(
    simulator: BaseSimulator,
    num_steps: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """Run a benchmark scenario and return metrics.

    Args:
        simulator: The simulator instance to benchmark
        num_steps: Number of simulation steps to run
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing benchmark metrics

    Raises:
        SimulatorError: If simulation fails
    """
    pass
```

### Commit Messages

- Use **present tense** ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues and PRs where applicable

Example:
```
Add MJX benchmarking support

- Implement MJX wrapper class
- Add cart-pole benchmark
- Include installation instructions

Closes #123
```

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Run all tests** and ensure they pass
4. **Update README.md** if adding new simulators
5. **Format code** with black: `black .`
6. **Check linting**: `flake8 .`
7. **Create pull request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Test results or benchmark data

### PR Review Process

- Maintainers will review your PR within 7 days
- Address any requested changes
- Once approved, a maintainer will merge your PR

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] CI/CD checks pass

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue with reproduction steps
- **Feature requests**: Open an Issue with detailed description
- **Security issues**: Email maintainers directly (do not open public issue)

## Recognition

Contributors will be:
- Listed in the project README
- Acknowledged in release notes
- Credited in academic papers using this work

Thank you for contributing to robotics simulation benchmarking!