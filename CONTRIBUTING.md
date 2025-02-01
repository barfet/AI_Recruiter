# Contributing to Aridmi

We love your input! We want to make contributing to Aridmi as easy and transparent as possible, whether it's:
- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the docs/ with any new information.
3. The PR will be merged once you have the sign-off of two other developers.

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/aridmi/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/aridmi/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* Use Black for Python code formatting
* 4 spaces for indentation rather than tabs
* Run `pylint` over your code
* Keep line length to 88 characters
* Write docstrings for all public functions

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## Testing

### Running Tests
```bash
PYTHONPATH=$PYTHONPATH:. python src/agent/test_chains.py
```

### Writing Tests
1. Create test files in the appropriate test directory
2. Use pytest for writing tests
3. Follow the existing test structure
4. Include both unit and integration tests
5. Mock external services appropriately

## Documentation

### Code Documentation
- Use docstrings for all public functions and classes
- Follow Google style docstrings
- Keep comments clear and relevant
- Update documentation when changing code

### Project Documentation
- Update README.md for major changes
- Keep technical docs up to date
- Document all configuration options
- Include examples where appropriate

## Development Setup

1. Clone the repo
```bash
git clone https://github.com/yourusername/aridmi.git
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. Set up pre-commit hooks
```bash
pre-commit install
```

## License
By contributing, you agree that your contributions will be licensed under its MIT License. 