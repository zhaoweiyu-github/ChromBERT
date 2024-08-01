# Configuration file for the Sphinx documentation builder.

# -- Project information
import sphinx_rtd_theme

project = 'ChromBERT'
copyright = '2024, Zhang Lab'
author = 'Zhaowei Yu, Dongxu Yang, Qianqian Cheng, Yuxuan Zhang'

release = '1.0.0'
version = '1.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "nbsphinx",
    "nbsphinx_link",
    "recommonmark",
    "sphinx.ext.viewcode"
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- Options for EPUB output
epub_show_urls = 'footnote'
nbsphinx_execute = "never"