# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import setuptools_scm

# Used when building API docs, put the dependencies
# of any class you are documenting here
autodoc_mock_imports = []

# Add the module path to sys.path here.
# If the directory is relative to the documentation root,
# use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../.."))

project = "RatInABox"
copyright = "2022, Tom George"
author = "Tom George"
try:
    release = setuptools_scm.get_version(root="../..", relative_to=__file__)
except LookupError:
    # if git is not initialised, still allow local build
    # with a dummy version
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
]

# Configure the myst parser to enable cool markdown features
# See https://sphinx-design.readthedocs.io
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
# Automatically add anchors to markdown headings
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Automatically generate stub pages for API
autosummary_generate = True
autodoc_default_flags = ["members", "inherited-members"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "**.ipynb_checkpoints",
    # to ensure that include files (partial pages) aren't built, exclude them
    # https://github.com/sphinx-doc/sphinx/issues/1965#issuecomment-124732907
    "**/includes/**",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_title = "ratinabox"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_favicon = "_static/logo_small.png"

# Customize the theme
html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/RatInABox-Lab/ratinabox",  # required
            # Icon class (if "type": "fontawesome"),
            # or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "logo": {
        "text": f"{project}", #v{release} for release version
        "image_light": "logo_small.png",
        "image_dark": "logo_small.png",
    },
}

# Redirect the webpage to another URL
# Sphinx will create the appropriate CNAME file in the build directory
# The default is the URL of the GitHub pages
# https://www.sphinx-doc.org/en/master/usage/extensions/githubpages.html
github_user = "RatInABox-Lab"
html_baseurl = f"https://{github_user}.github.io/{project}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

