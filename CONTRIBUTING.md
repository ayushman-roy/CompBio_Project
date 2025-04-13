# Contributing to Age-Structured SIRQVD Model

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by the following code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Be open to different perspectives
- Maintain professional communication

## How to Contribute

### 1. Reporting Issues

When reporting issues, please include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant error messages or logs

### 2. Feature Requests

For feature requests, please:
- Describe the feature clearly
- Explain its benefits
- Provide use cases
- Suggest implementation approaches if possible

### 3. Pull Requests

When submitting pull requests:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure tests pass
5. Update documentation
6. Submit the pull request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/age-structured-sirqvd-model.git
cd age-structured-sirqvd-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Document all functions with docstrings
- Keep lines under 80 characters
- Use meaningful variable names

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting
- Include edge cases in tests
- Document test coverage

## Documentation

- Update README.md for significant changes
- Keep model_assumptions.txt current
- Document new parameters
- Explain mathematical formulations
- Provide usage examples

## Project Structure

```
age-structured-sirqvd-model/
├── script.py              # Main model implementation
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── model_assumptions.txt # Model parameters and assumptions
├── CONTRIBUTING.md       # Contribution guidelines
├── LICENSE               # MIT License
└── .gitignore           # Git ignore file
```

## Review Process

1. Pull requests will be reviewed by maintainers
2. Feedback will be provided within 7 days
3. Changes may be requested before merging
4. All tests must pass before merging

## Questions?

Feel free to:
- Open an issue for questions
- Contact the maintainers
- Join the discussion 