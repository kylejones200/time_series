# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Time Series Forecasting'
copyright = '2025, Time Series Forecasting Team'
author = 'Time Series Forecasting Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'planning']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None

# -- Extension configuration -------------------------------------------------

# Napoleon settings for NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__'
}
autodoc_mock_imports = [
    'torch', 'tensorflow', 'keras', 'prophet', 'pmdarima', 'statsmodels',
    'sklearn', 'xgboost', 'lightgbm', 'darts', 'statsforecast', 'greykite',
    'merlion', 'orbit', 'chronos', 'timesfm', 'tsai', 'tsfresh', 'tslearn',
    'aeon', 'stumpy', 'pyod', 'pymc', 'arviz', 'copulas', 'pyinform',
    'linearmodels', 'autogluon', 'pycaret', 'pybsts', 'signalplot'
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

