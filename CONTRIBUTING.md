# Contributing to WarpFactory Python Port

Thank you for your interest in contributing to the Python port of WarpFactory! This document outlines the process for contributing to this implementation.

## Getting Started

1. Make sure you have Python 3.9 or higher installed
2. Install poetry for dependency management
3. Fork the repository
4. Clone your fork locally
5. Set up the development environment:
   ```bash
   poetry install
   ```

## Development Process

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Write tests first (Test-Driven Development):
   ```bash
   poetry run pytest warpfactory/tests/
   ```

3. Implement your changes following the project structure
4. Ensure all tests pass and add new ones as needed
5. Update documentation if necessary

## Pull Request Process

1. Create an issue first to discuss major changes
2. Update the README.md if needed
3. Add tests for new functionality
4. Ensure your code follows Python best practices:
   - Type hints
   - Docstrings
   - PEP 8 style guide
5. Submit a pull request with a clear description of changes

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and returns
- Include docstrings in NumPy format
- Keep functions focused and modular
- Add comments only for non-obvious behavior

## Spacetime Analysis Features

When contributing to spacetime analysis features:

1. **Geodesic Solver**:
   - Ensure proper normalization of four-velocities
   - Handle timelike/null geodesics correctly
   - Validate energy conservation
   - Test with known analytic solutions

2. **Event Horizons**:
   - Use robust numerical methods for horizon finding
   - Handle both apparent and event horizons
   - Test with standard black hole metrics
   - Validate horizon properties

3. **Singularity Detection**:
   - Calculate curvature invariants correctly
   - Classify singularity types properly
   - Handle coordinate vs. physical singularities
   - Test with known singular spacetimes

4. **Gravitational Lensing**:
   - Ensure proper ray tracing implementation
   - Calculate optical properties accurately
   - Handle caustics and multiple images
   - Test with standard lensing scenarios

## Testing

- Write tests using pytest
- Aim for high test coverage
- Include edge cases and error conditions
- Test GPU functionality if applicable

## Documentation

- Update docstrings for any modified functions
- Keep the README.md up to date
- Add examples for new features
- Document any deviations from the original MATLAB implementation

## Relationship with Original Project

This is a Python port of the [original WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory) MATLAB implementation. While we maintain functional compatibility, we follow Python-specific best practices and conventions. If you're interested in contributing to the original MATLAB version, please visit their repository.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License. Make sure you have the right to submit any code you contribute.

## Questions or Problems?

- Check existing issues
- Create a new issue for bugs or feature requests
- Ask for clarification on implementation details
- Discuss major changes before starting work

Thank you for helping make WarpFactory more accessible to the Python community!