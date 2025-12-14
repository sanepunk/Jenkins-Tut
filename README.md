# Jenkins Tutorial - Resource Heavy Testing Suite

## Overview
This project contains a comprehensive suite of 50 resource-heavy pytest tests designed for performance testing, load testing, and stress testing various system resources including CPU, memory, I/O, and concurrency.

## Project Structure
```
Jenkins-Tut/
├── README.md
├── requirements.txt
└── test_resource_heavy.py
```

## Test Categories

### 1. CPU Intensive Tests (5 tests)
- Prime number calculations
- Factorial computations
- Recursive Fibonacci sequences
- Large matrix multiplications
- Cryptographic hashing operations

### 2. Memory Intensive Tests (5 tests)
- Large array allocations
- Massive list creation and manipulation
- Dictionary operations with 100K+ entries
- String concatenation stress tests
- Deeply nested data structures

### 3. I/O Intensive Tests (5 tests)
- Large file write/read operations (100MB+)
- Multiple concurrent file operations
- JSON serialization/deserialization
- Binary file operations
- Buffered I/O operations

### 4. Concurrency Intensive Tests (5 tests)
- Multithreading with thread pools
- Multiprocessing with process pools
- Thread safety and race conditions
- Parallel I/O operations
- Producer-consumer patterns

### 5. Computational Intensive Tests (5 tests)
- Monte Carlo Pi estimation
- Numerical integration
- Linear algebra operations
- Fast Fourier Transform (FFT)
- Statistical computations

### 6. Data Processing Intensive Tests (5 tests)
- Sorting large datasets (1M+ items)
- Filtering operations
- Data aggregation
- Transformation pipelines
- Join operations

### 7. Network Simulation Tests (5 tests)
- Data serialization
- Compression simulation
- Packet processing
- Bandwidth simulation
- Protocol handling

### 8. Combined Stress Tests (5 tests)
- CPU + Memory stress
- I/O + CPU stress
- All resources combined
- Parallel stress testing
- Endurance testing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Jenkins-Tut
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

### Run all tests:
```bash
pytest test_resource_heavy.py -v
```

### Run specific test class:
```bash
pytest test_resource_heavy.py::TestCPUIntensive -v
```

### Run with parallel execution (using pytest-xdist):
```bash
pytest test_resource_heavy.py -n auto
```

### Run with detailed output:
```bash
pytest test_resource_heavy.py -v -s
```

### Run with coverage:
```bash
pytest test_resource_heavy.py --cov=. --cov-report=html
```

## Performance Considerations

⚠️ **Warning**: These tests are intentionally resource-heavy and may:
- Consume significant CPU resources
- Use several GB of RAM
- Generate large temporary files
- Take several minutes to complete
- Potentially slow down your system during execution

### Recommended System Requirements:
- **CPU**: 4+ cores
- **RAM**: 8GB+ available
- **Disk Space**: 5GB+ free space
- **OS**: Linux, macOS, or Windows

## CI/CD Integration

### Jenkins Pipeline Example:
```groovy
pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest test_resource_heavy.py --junitxml=test-results.xml'
            }
        }
    }
    post {
        always {
            junit 'test-results.xml'
        }
    }
}
```

### GitHub Actions Example:
```yaml
name: Resource Heavy Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest test_resource_heavy.py -v
```

## Test Execution Times

Expected execution times (approximate):
- CPU Intensive: ~30-60 seconds
- Memory Intensive: ~20-40 seconds
- I/O Intensive: ~60-120 seconds
- Concurrency: ~30-60 seconds
- Computational: ~40-80 seconds
- Data Processing: ~30-60 seconds
- Network Simulation: ~20-40 seconds
- Combined Stress: ~40-80 seconds

**Total Suite**: 4-10 minutes (depending on system resources)

## Troubleshooting

### Out of Memory Errors
If you encounter memory errors, consider:
- Running tests individually or by class
- Increasing system swap space
- Running on a machine with more RAM

### Slow Execution
- Use `pytest -n auto` for parallel execution
- Run tests on dedicated hardware
- Skip certain test categories using markers

### File Permission Errors
- Ensure proper write permissions in temp directories
- Run with appropriate user permissions

## Contributing
Feel free to add more resource-heavy tests or optimize existing ones. Please ensure:
- Tests are well-documented
- Resource usage is clearly indicated
- Tests are reproducible across different systems

## License
MIT License