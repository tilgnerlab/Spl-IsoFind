import os
import sys
sys.path.insert(0, os.path.abspath("../src"))  # adjust if using top-level layout

project = "SplIsoFind"
author = "Lieke Michielsen"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "myst_nb",
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

nbsphinx_execute = 'never'