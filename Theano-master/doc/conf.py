

from __future__ import absolute_import, print_function, division

import os
import sys
theano_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(theano_path))


extensions = ['sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.doctest', 'sphinx.ext.napoleon']

todo_include_todos = True
napoleon_google_docstring = False
napoleon_include_special_with_doc = False

try:
    from sphinx.ext import pngmath
    extensions.append('sphinx.ext.pngmath')
except ImportError:
    pass


templates_path = ['.templates']

source_suffix = '.txt'

master_doc = 'index'

project = 'Theano'
copyright = '2008--2016, LISA lab'

version = '0.8'
release = '0.8.0'

today_fmt = '%B %d, %Y'


exclude_dirs = ['images', 'scripts', 'sandbox']





pygments_style = 'sphinx'



html_theme = 'sphinxdoc'



html_logo = 'images/theano_logo_allblue_200x46.png'


html_static_path = ['.static', 'images', 'library/d3viz/examples']

html_last_updated_fmt = '%b %d, %Y'

html_use_smartypants = True









htmlhelp_basename = 'theanodoc'




latex_font_size = '11pt'

latex_documents = [
  ('index', 'theano.tex', 'theano Documentation',
   'LISA lab, University of Montreal', 'manual'),
]

latex_logo = 'images/theano_logo_allblue_200x46.png'




