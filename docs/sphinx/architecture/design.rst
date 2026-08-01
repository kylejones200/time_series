Design Principles
==================

The library is designed around several key principles:

Consistency
-----------

All templates follow the same structure and interface:

- ``config.yaml`` for configuration
- ``main.py`` as the entry point
- ``outputs/`` for results
- Consistent evaluation metrics

DRY (Don't Repeat Yourself)
----------------------------

Common functionality is consolidated in ``src/``:

- Data loading: ``src.loader``
- Configuration: ``src.config``
- Plotting: ``src.plotting``
- Evaluation: ``src.evaluator``

This reduces duplication and ensures consistency.

Config-Driven
--------------

All templates use YAML configuration files, making it easy to:

- Reproduce experiments
- Share configurations
- Version control settings
- Parameterize without code changes

Extensibility
------------

The library is designed to be extended:

- Add new models via the model registry
- Create custom templates following the base structure
- Extend the pipeline with new evaluation metrics
- Integrate with external tools

Platform Compatibility
----------------------

The library is designed to work across platforms:

- Cross-platform file handling
- UTF-8 encoding for all file operations
- Proper resource cleanup (especially on Windows)
- Defensive checks for optional dependencies

