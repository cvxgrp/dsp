.. dsp documentation master file, created by
   sphinx-quickstart on Sat Jan 28 22:51:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Disciplined Saddle Programming
===============================

.. meta::
   :description: A CVXPY extension for saddle problems
   :keywords: convex optimization, python, cvxpy, saddle programming, saddle-point, modeling-language

Disciplined Saddle Programming (DSP) is a CVXPY extension for saddle problems.
DSP allows solving convex-concave saddle point problems, and more generally convex optimization problems we refer to
as saddle problems, which include the partial supremum or infimum of convex-concave saddle functions. Saddle functions
are functions that are (jointly) convex in a subset of their arguments, and (jointly) concave in the remaining
arguments. A detailed description of the underlying method is given in our `accompanying paper <https://arxiv.org/abs/2301.13427>`_.

This page contains the API documentation for DSP. We limit the public API to the functions and classes listed contained
in this reference. The remainder of the code is considered private and may change without notice.

.. toctree::
   :hidden:

   API Documentation <api_reference/dsp>
