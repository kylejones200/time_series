Contributing
============

We welcome contributions! This document outlines how to contribute to the Time Series Forecasting library.

Getting Started
---------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Code Quality
------------

Please follow the code quality guidelines in ``.cursor/code_quality.md``:

- Use string forward references for optional dependency types
- Add defensive checks for None when optional dependencies might not be available
- Use explicit UTF-8 encoding for file operations
- Properly close file handles (especially on Windows)

Documentation
-------------

When adding new features:

1. Update relevant documentation in ``docs/sphinx/``
2. Add docstrings following NumPy/Google style
3. Include examples in the appropriate template or guide
4. Update the changelog

Testing
-------

Before submitting:

1. Run the reference forecast to ensure core functionality works
2. Test your changes with at least one template
3. Verify no linter errors

Style Guide
-----------

- Follow PEP 8
- Use `black` for formatting
- Maximum line length: 100 characters
- Use type hints where appropriate

