import os
import sys
# Point to your source code
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',    # To pull docstrings from code
    'sphinx.ext.napoleon',   # To support Google/NumPy style docstrings
    'sphinx.ext.viewcode',   # To add links to highlighted source code
]

html_theme = 'sphinx_rtd_theme'
