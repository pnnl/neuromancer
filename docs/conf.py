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
import sys

sys.path.insert(0, os.path.abspath("../neuromancer"))


# -- Project information -----------------------------------------------------

project = "NeuroMANCER"
copyright = "2023, Aaron Tuor, Jan Drgona, Mia Skomski, Stefan Dernbach, James Koch, Zhao Chen, Christian Møldrup Legaard, Draguna Vrabie, Madelyn Shapiro"
author = "Aaron Tuor, Jan Drgona, Mia Skomski, Stefan Dernbach, James Koch, Zhao Chen, Christian Møldrup Legaard, Draguna Vrabie, Madelyn Shapiro"

# The full version, including alpha/beta/rc tags
release = "1.5.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Automatically generated rst files
    "sphinx.ext.autosummary",  # Automatically generated rst files
    "sphinx.ext.napoleon",  # For Google-style or NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",  # generates .nojekyll file
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinxcontrib.apidoc",  # runs sphinx-apidoc before build
]

# sphinx-apidoc generates rst files from python source
apidoc_module_dir = "../src/neuromancer/"
apidoc_output_dir = "generated"
apidoc_separate_modules = True
apidoc_template_dir = "_templates"
apidoc_toc_file = False

# Add any paths that contain templates here, relative to this directory.
# responsible for darker blue color background in sidebar
templates_path = ["_templates"]

master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {"navigation_depth": 4}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_extra_path = ["copied"]  # copied to output directory

html_logo = "figs/Neuromancer.png"

html_favicon = "_static/Neuromancer.ico"

html_sidebars = {
    "**": [
        "localtoc.html",
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

html_js_files = ["js/custom.js"]

html_theme_options = {
    "collapse_navigation": False,
}
