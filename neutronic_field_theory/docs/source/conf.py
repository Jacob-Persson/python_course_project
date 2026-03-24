
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neutronic_field_theory'
copyright = '2026, Vattenfall Nuclear Fuel AB'
author = 'Jacob Persson'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc',     # Pulls docstrings from code
    'sphinx.ext.napoleon',    # Supports Google/NumPy styles
    'sphinx.ext.viewcode',    # Links to source code in HTML
    'sphinx.ext.mathjax',     # Renders LaTeX formulas
]
html_theme = 'sphinx_rtd_theme' # The "Read the Docs" look

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
