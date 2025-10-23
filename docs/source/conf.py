# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('/home/amirsamandar/Desktop/Cosmology/COMPACT/Computation/TopologyPy'))

# Project information
project = 'TopologyPy'
copyright = '2025, Amirhossein Samandar'
author = 'Amirhossein Samandar'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',      # Extract docstrings
    'sphinx.ext.napoleon',     # Support Google-style docstrings
    'sphinx.ext.viewcode',     # Link to source code
    'sphinx.ext.mathjax',      # Render math (e.g., C_\ell)
]

# Paths
templates_path = ['_templates']
html_static_path = ['_static']

# HTML theme
html_theme = 'sphinx_rtd_theme'  # Explicitly set to ReadTheDocs theme

# html_theme_options = {
#     'display_version': True,    # Show version number (e.g., 0.1.0)
#     'logo_only': False,         # Show both logo and project name
#     'collapse_navigation': False,  # Keep navigation expanded
#     'navigation_depth': 4,      # Show deeper TOC levels
# }

# Autodoc settings
autodoc_member_order = 'bysource'
autoclass_content = 'both'