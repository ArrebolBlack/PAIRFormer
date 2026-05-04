# PAIR-Former Tests

This directory contains tests for PAIR-Former.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_models.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

- `test_models.py` - Model registry and building tests
- `test_data.py` - Data loading and processing tests
- `test_training.py` - Training logic tests (TODO)
- `test_ddp.py` - DDP functionality tests (TODO)

## Requirements

```bash
pip install pytest pytest-cov
```

## CI/CD

Tests are automatically run on:
- Pull requests
- Commits to main branch

See `.github/workflows/tests.yml` for CI configuration.
