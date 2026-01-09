# Sphinx Documentation

This directory contains the Sphinx documentation source files for ReadTheDocs.

## Building Locally

Install dependencies:

```bash
pip install -r docs/requirements.txt
```

Build documentation:

```bash
cd docs/sphinx
sphinx-build -b html . _build/html
```

View documentation:

```bash
open _build/html/index.html
```

## Structure

- `index.rst` - Main entry point
- `getting_started/` - Installation and quick start guides
- `guides/` - User guides
- `api/` - API reference documentation
- `examples/` - Template examples and usage
- `architecture/` - Design and structure documentation

## ReadTheDocs

The documentation is configured for ReadTheDocs via `.readthedocs.yaml`.

To set up on ReadTheDocs:

1. Connect your GitHub repository
2. ReadTheDocs will automatically detect `.readthedocs.yaml`
3. Documentation will build automatically on each commit

## Adding Documentation

When adding new features:

1. Update relevant `.rst` files
2. Add API documentation using autodoc
3. Include examples
4. Update the changelog

See `contributing.rst` for more details.

