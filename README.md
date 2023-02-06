# DSP -  Disciplined Saddle Programming
[![test](https://github.com/cvxgrp/dsp/actions/workflows/test.yml/badge.svg)](https://github.com/cvxgrp/dsp/actions/workflows/test.yml)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=dsp&metric=coverage)](https://sonarcloud.io/summary/new_code?id=dsp)

A CVXPY extension for Disciplined Saddle Programming.
DSP allows solving convex-concave saddle point problems, and more generally
convex optimization problems we refer to as _saddle problems_, which include the partial
supremum or infimum of convex-concave saddle functions.
Saddle functions are functions that are (jointly) convex in a subset of their
arguments, and (jointly) concave in the remaining arguments.
A detailed description of the underlying method is given in our [accompanying paper](https://arxiv.org/abs/2301.13427).

## Installation

The DSP package can be installed using pip as follows
```bash
pip install git+https://github.com/cvxgrp/dsp.git
```
A release to PyPI will follow shortly.
The DSP package requires CVXPY 1.3 or later.

## Minimal example

The following example creates and solves a simple saddle point problem known as a matrix game.
A saddle point problem is created by specifying an objective and a list of constraints.
Here, the objective is $f(x, y) = x^TCy$, which is simultaneously minimized over $x$ and maximized over $y$.
The resulting saddle point is an optimal mixed strategy for both players in the matrix game.
Each component is explained below in more detail.

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
- `saddle_inner(Fx, Gy)`  
The inner product $F(x)^TG(y)$, with $F$ convex and nonnegative, and $G$ concave. If $G$ is not nonnegative, a constraint
$G \geq 0$ is added.
- `weighted_norm2(x, y)`  
The weighted $\ell_2$ norm $\left(\sum_i y_i x_i^2\right)^{1/2}$. Here too, a constraint $y \geq 0$ is added if $y$ is not
nonnegative.
- `weighted_log_sum_exp(x, y)`
The weighted log-sum-exp function $\log\left(\sum_i y_i \exp(x_i)\right)$. Again a constraint $y \geq 0$ is added if $y$ is not
nonnegative.
- `quasidef_quad_form(x, y, P, Q, S)`  
For a positive semidefinite matrix $P$ and a negative semidefinite matrix $S$, this atom represents the function

$$
f(x,y) = \left[\begin{array}{c} x \\\\ y \end{array}\right]^T
\left[\begin{array}{cc} P & S \\\\ S^T & Q \end{array}\right]
\left[\begin{array}{c} x \\\\ y \end{array}\right].
$$
- `saddle_quad_form(x, Y)`  
The quadratic form $x^TYx$, where $Y$ a positive semindefinite matrix.

## Calculus rules
Saddle functions can be scaled and composed by addition. DCP convex expressions are treated as saddle functions with
no concave arguments, and DCP concave expressions are treated as saddle functions with no convex arguments.
When adding two saddle functions, a variable may not appear as a convex variable in one expression and as a concave
variable in the other expression.

## Saddle point problems
To create a saddle point problem, a `MinimizeMaximize` object is created first, which represents the objective function,
using
```python
obj = dsp.MinimizeMaximize(f)
```
where `f` is a DSP-compliant expression.

The syntax for specifying saddle point problems is
```python
problem = dsp.SaddlePointProblem(obj, constraints, cvx_vars, ccv_vars)
```
where `obj` is the `MinimizeMaximize` object, `constraints` is a list of constraints, and `cvx_vars` and `ccv_vars` are
lists of variables to be minimized and maximized over, respectively.

Each constraint must be DCP, and can only involve variables that are either convex or concave.
When the role of a variable can be inferred, it can be omitted from the list of convex or concave variables.
The role can be inferred either from a saddle atom, a DCP atom that is convex or concave, but not affine, or from a
constraint, when a variable appears in a constraint that involves variables with known roles.

Nevertheless, specifying the role of each variable can add clarity to the problem formulation, and is especially
useful for debugging.

To solve the problem, call `problem.solve()`. This returns the optimal saddle value, which is also stored in the
problem's `value` attribute. Further all `value` attribute of the variables are populated with their optimal values.

## Saddle extremum functions
A saddle extremum function has one of the forms

$$
G(x) = \sup_{y \in \mathcal{Y}} f(x,y) \quad \text{or} \quad
H(y) = \inf_{x \in \mathcal{X}} f(x,y),
$$

where $f$ is a saddle function, and $\mathcal{X}$ and $\mathcal{Y}$ are convex.
Since the functions only depend on $x$ or $y$, respectively, the other variables have to be declared as
`LocalVariable`s. Any `LocalVariable` can only be used in one saddle extremum function. The syntax for
creating a saddle extremum function is
```python
dsp.saddle_max(f, constraints)
dsp.saddle_min(f, constraints)
```
where `f` is a DSP-compliant scalar saddle function, and `constraints` is a list of constraints, which can
only involve `LocalVariable`s. DSP-compliant saddle extremum functions are DCP-convex or DCP-concave, respectively,
and as such can be used in DCP optimization problems.

An example of a saddle extremum function is
```python
# Creating variables
x = cp.Variable(2)

# Creating local variables
y_loc = dsp.LocalVariable(2)

# Convex in x, concave in y_loc
f = dsp.saddle_inner(C @ x, y_loc)

# maximizes over y_loc
G = dsp.saddle_max(f, [y_loc >= 0, cp.sum(y_loc) == 1])
```

## Saddle problems
A saddle problem is a convex optimization problem that involves saddle extremum functions. Any DCP convex optimization
can include saddle extremum functions when they are DSP-compliant. Using the saddle extremum function `G` from above,
we can solve the following problem:
```python
prob = cp.Problem(cp.Minimize(G), [x >= 0, cp.sum(x) == 1])
prob.solve() # solving the problem

prob.value # 1.6666666666666667
x.value # array([0.66666667, 0.33333333])
```

## Citation
If you want to reference DSP in your research, please consider citing us by using the following BibTeX:

```BibTeX
@misc{schiele2023dsp,
  title={Disciplined Saddle Programming},
  author= {Schiele*, Philipp and Luxenberg*, Eric and Boyd, Stephen},
  year={2023},
  selected={true},
  journal={arXiv preprint arXiv:2301.13427},
}
```
