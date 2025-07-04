# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath('./../'))


# -- Project information -----------------------------------------------------

project = 'TRIDENT'
copyright = '2025, Guillaume Jaume'
author = 'Guillaume Jaume'

# The full version, including alpha/beta/rc tags
release = 'v0.1.1'

# HTML style
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/lab_logo.svg'
html_theme_options = {
    "sidebar_hide_name": True,
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # For NumPy or Google-style docstrings
    "sphinx_design",  
]
autosummary_generate = True

autoclass_content = 'both'  # Shows class-level and __init__ docstring
napoleon_include_init_with_doc = True  # for Google/NumPy-style docstrings

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_context = {
    "display_github": True,
    "github_user": "guillaumejaume",
    "github_repo": "TRIDENT",
    "github_version": "docs",
    "conf_py_path": "/docs/",
}

# === Auto-generate CLI help files before building docs ===

def run_cli_generate():
    print("Auto-generating CLI help text...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_generate_script = os.path.join(root_dir, 'docs', 'cli_helpers', 'cli_generate.py')
    output_help_txt = os.path.join(root_dir, 'docs', 'generated', 'run_batch_of_slides_help.txt')

    os.makedirs(os.path.dirname(output_help_txt), exist_ok=True)

    with open(output_help_txt, 'w') as f:
        subprocess.run(["python", cli_generate_script], stdout=f, check=True)

run_cli_generate()
