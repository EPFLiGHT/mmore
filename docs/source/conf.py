project = "mmore"
author = "mmore contributors"
copyright = "2026, mmore contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "mmore Documentation"
# Uncomment if you want to use a project logo placed in docs/source/_static/
# html_logo = "_static/mmore_logo.jpg"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]

autosummary_generate = True

# Replace with your real repository information when ready.
html_theme_options = {
    "source_repository": "https://github.com/EPFLiGHT/mmore/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
