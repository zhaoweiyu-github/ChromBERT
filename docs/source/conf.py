# Configuration file for the Sphinx documentation builder.

# -- Project information
import sphinx_rtd_theme

project = 'ChromBERT'
copyright = '2024, Zhang Lab'
author = 'Zhaowei Yu, Dongxu Yang, Qianqian Chen, Yuxuan Zhang'

release = '1.1.0'
version = '1.1.0'

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

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "zhaoweiyu-github", # Username
    "github_repo": "ChromBERT", # Repo name
    "github_version": "main/", # Branch
    "conf_py_path": "/docs/source/", # Path in the repo to conf.py
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
nbsphinx_execute = "never"