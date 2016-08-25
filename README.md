About
======

This is a supplementary code (in Python 2.7) for the paper Malitsky Y., Pock T. "Primal-dual method with a linesearch"

Usage
======
There are 3 problems: Min-max game, l1-regularized least square, and nonnegative least square.
Each problem is presented in the individual Jupyter (formerly Ipython) Notebook file with extension `ipynb`.
To play with these problems you need Jypyter Notebook to be installed on your computer. 
Alternatively, you can read these notebooks using http://nbviewer.jupyter.org/

Folder `real_data` collects 4 files with real-data matrices. These files are taken from Matrix Market Library http://math.nist.gov/MatrixMarket/

Folder `figures` collects all plots that were usen in the paper.

File `algorithms.py` collects codes of all optimization algorithms except primal-dual ones.

File `pd_algorithms` includes codes of all primal-dual algorithms (including acceleration and linesearch)

File `opt_operators` includes codes for several proximal operators.