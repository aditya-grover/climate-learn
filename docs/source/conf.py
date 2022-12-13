# Configuration file for the Sphinx documentation builder.
import sphinx_rtd_theme
import os
import sys
import climate_learn

# Point ReadTheDocs to the directory
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information

project = 'ClimateLearn'
copyright = '2022; Bansal, Goel, Jewik, Nandy, Nguyen, Park, Tang, Grover'
author = """
    Hritik Bansal,
    Shashank Goel,
    Jason Jewik,
    Siddharth Nandy,
    Tung Nguyen,
    Seongbin Park,
    Jingchen Tang,
    Aditya Grover
"""

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'nbsphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# How to represents typehints
autodoc_typehints = 'signature'

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
