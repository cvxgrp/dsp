# DSP -  Disciplined Saddle Programming
[![test](https://github.com/cvxgrp/dsp/actions/workflows/test.yml/badge.svg)](https://github.com/cvxgrp/dsp/actions/workflows/test.yml)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=dsp&metric=coverage)](https://sonarcloud.io/summary/new_code?id=dsp)

A CVXPY extension for Disciplined Saddle Programming.
DSP allows solving convex-concave saddle point problems, and more generally
convex optimization problems we refer to as _saddle problems_, which include the partial
supremum or infimum of convex-concave saddle functions.
Saddle functions are functions that are (jointly) convex in a subset of their
arguments, and (jointly) concave in the remaining arguments.
A forthcoming paper will describe the method in detail.

## Installation

The DSP package can be installed using pip as follows
```bash
pip install git+https://github.com/cvxgrp/dsp.git
```
A release to PyPI will follow shortly.
The DSP package requires CVXPY 1.3 or later.

## Minimal example

The following example creates and solves a simple saddle point problem known as a matrix game.
A saddle point problem is created by specifying a `MinimizeMaximize` objective and a list of constraints.
Here, the objective is $f(x, y) = x^TCy$, which is simultaneously minimized over $x$ and maximized over $y$.
The resulting saddle point is an optimal mixed strategy for both players in the matrix game.

```python
import dsp
import cvxpy as cp
import numpy as np

x = cp.Variable(2)
y = cp.Variable(2)
C = np.array([[1, 2], [3, 1]])

f = dsp.inner(x, C @ y)
obj = dsp.MinimizeMaximize(f)

constraints = [x >= 0, cp.sum(x) == 1, y >= 0, cp.sum(y) == 1]
prob = dsp.SaddlePointProblem(obj, constraints)
prob.solve()  # solves the problem

prob.value  # 1.6666666666666667
x.value  # array([0.66666667, 0.33333333])
y.value  # array([0.33333333, 0.66666667])
```

## New atoms
In DSP, saddle functions are created from atoms. Each atom represents a saddle function, with the convention being
that the first argument is the convex argument and the second argument is the concave argument.

- `inner(x, y)`
The inner product $x^Ty$, with both arguments affine.
- `saddle_innder(Fx, Gy)`
The inner product $F(x)^TG(y)$, with $F$ convex and nonnegative, and $G$ concave. If $G$ is not nonnegative, a constraint
$G \geq 0$ is added.
- `weighted_norm2(x, y)`
The weighted $\ell_2$ norm $\left(\sum_i y_i x_i^2\right)^{1/2}$. Here too, a constraint $y \geq 0$ is added if $y$ is not
nonnegative.
- `weighted_log_sum_exp(x, y)`
The weighted log-sum-exp function $\log\left(\sum_i y_i \exp(x_i)\right)$. Again a constraint $y \geq 0$ is added if $y$ is not
nonnegative.
- `quasidef_quad_form(x, y, P, Q, S)`
This atom represent the function $f(x, y) = \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} P & S \\ S^T & Q \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$,
where $P$ is positive semidefinite and $S$ is negative semidefinite.
- `saddle_quad_form(x, Y)`
The quadratic form $x^TYx$, where $Y$ a positive semindefinite matrix.

## Calculus rules


## Citation
TBD