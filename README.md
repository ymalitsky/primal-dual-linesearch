About
======

This is a supplementary code (in Python 3) for the paper Malitsky Y., Pock T. "A first-order primal-dual algorithm with linesearch"

Usage
======
There are 3 problems: Min-max game, l1-regularized least square, and nonnegative least square.
Each problem is presented in the individual Jupyter (formerly Ipython) Notebook file with extension `ipynb`.
To play with these problems you need Jypyter Notebook to be installed on your computer. 
Alternatively, you can read these notebooks using http://nbviewer.jupyter.org/

File `algorithms.py` collects codes of all optimization algorithms except primal-dual ones.

File `pd_algorithms.py` includes codes of all primal-dual algorithms (including accelerated versions and ones with linesearch)

File `opt_operators.py` includes codes for several proximal operators.

File `misc_for_output.py` includes codes of some functions for nicer outputs.

Folder `figures` collects all plots that were used in the paper.

