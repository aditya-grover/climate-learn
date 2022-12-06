# Configuration file for the Sphinx documentation builder.
import sphinx_rtd_theme

extensions = [
    'sphinx_rtd_theme'
]

# -- Project information

project = 'ClimateLearn'
copyright = '2022; Bansal, Goel, Nguyen, Grover'
author = 'Hritik Bansal, Shashank Goel, Tung Nguyen, Aditya Grover'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
