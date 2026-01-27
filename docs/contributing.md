# Contributing

We welcome contributions to WEIRDO! This document provides guidelines for
contributing to the project.

## Development Setup

1. Fork the repository on GitHub

2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/weirdo.git
   cd weirdo
   ```

3. Install in development mode:
   ```bash
   ./develop.sh
   ```

4. Run tests to verify your setup:
   ```bash
   ./test.sh
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use NumPy-style docstrings for all public functions and classes
- Add type hints where practical
- Keep line length under 100 characters

## Running Lint

Before submitting a PR, ensure your code passes linting:

```bash
./lint.sh
```

## Running Tests

Run the full test suite:

```bash
./test.sh
```

Run specific tests:

```bash
pytest test/test_scorers.py -v
```

## Writing Tests

- Add tests for all new functionality
- Place tests in the `test/` directory
- Use pytest fixtures for common setup
- Aim for high coverage of edge cases

Example test:

```python
import pytest
from weirdo.scorers import MLPScorer

def test_mlp_scorer_basic():
    """Test basic MLP scoring."""
    peptides = ['AAAAAAAA', 'XXXXXXXX'] * 10
    labels = [0.0, 1.0] * 10
    scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=1)
    scorer.train(peptides, labels, epochs=20, verbose=False)
    scores = scorer.score(['AAAAAAAA', 'XXXXXXXX'])
    assert scores[0] < scores[1]
```

## Documentation

- Update documentation for new features
- Build docs locally to check:
  ```bash
  mkdocs serve  # Preview at http://localhost:8000
  mkdocs build  # Build static site
  ```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure tests pass and lint is clean
4. Update documentation if needed
5. Submit a pull request with a clear description

## Adding New Scorers

To add a new scorer:

1. Create a new file in `weirdo/scorers/`
2. Subclass `BaseScorer` or `BatchScorer`
3. Implement `fit()` and `score()` methods
4. Register with `@register_scorer` decorator
5. Add tests in `test/test_scorers.py`
6. Document in `docs/scorers.md`

Example:

```python
from weirdo.scorers import register_scorer, BaseScorer

@register_scorer('my_scorer', description='My custom scorer')
class MyScorer(BaseScorer):
    def fit(self, reference):
        # Setup logic
        self._is_fitted = True
        return self

    def score(self, peptides):
        self._check_is_fitted()
        # Scoring logic
        pass
```

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0.
