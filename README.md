# GDDP

Generalized dual dynamic programming (GDDP) tool written in Python. This tool creates an approximate lower bounds on a given control problem's optimal value function using the Benders decomposition approach developed in [1].

## Getting Started

### Requirements

The package uses standard libraries, plus the solver Gurobi (free for academic users, commercial otherwise). To use Gurobi you need to install the package `gurobipy` and set the environment variable `GRB_LICENSE_FILE` to point to a valid `gurobi.lic` file.

### Sample usage

Make a system, either by specifying a system's dynamics via a `model_dict`, or by calling on one of the preset systems hard-coded in the definition of class `System`.

```python
from gddp import System, VFApproximator
sys1 = System(name='Simple 1D system', model_preset='1D')
```

The `System` object carried not only the dynamics and constraints of the system itself, but also the stage cost function and discount factor.

You can now pass this system to a `VFApproximator` object, and create an approximation of the problem's value function:

```python
vfa = VFApproximator(system=sys1)
vfa.create_vfa_model()
vfa.approximate()
```

By default, this performs the Benders decomposition algorithm at 100 randomly-chosen points in the state space, normally distributed about the origin.

The routine creates a number of output files that contain information on the value function created.

### Q functions

Some functionality also exists for working with Q functions, which are defined on state-action space rather than the state alone. The syntax is the same:

```python
from gddp import QFApproximator
qfa = QFApproximator(system=sys1)
qfa.create_qfa_model()
qfa.approximate()
```

## Parameters

Aside from the choice of system, numerous parameters can be chosen to govern how value function approximation is carried out by the function `vfa.approximate(strategy_in=None, audit_in=None, outputs_in=None)`. Dictionaries overriding any of the default settings can be passed via the three arguments.

The `strategy_in` dictionary chooses how the approximation routine selects and acts on points in the state space:

Key | Data type | Default value | Description
---|---|---|---
max_iter | int | 100 | Number of GDDP iterations to perform
n_x_points | int | 100 | Number of _x_ points to sample
rand_seed | int | 1 | Seed for randomly generating _x_ points
sol_strategy | str | 'random' | Strategy for picking next _x_ point
conv_tol |float|1e-4| Bellman error below which to stop iterations
stop_on_convergence|Boolean|False| Stop iterations when convergence criterion satisfied
remove_redundant|Boolean|False| Remove redundant lower bounding functions
removal_freq|int|10| If remove_redundant==True, frequency of LBF removal
removal_resolution|int|1e5|Number of random points used to assess LBF redundancy
focus_on_origin|Boolean|False| Concentrate half of points closer to the origin (lower variance)
value_function_limit|float|1e4| Threshold value above which new LBF won't be created at this iteration

The `audit_in` dictionary determines how performance is measured during the iterations:

Key | Data type | Default value | Description
---|---|---|---
eval_ub|Boolean |False| Run closed-loop simulations at intermediate iterations to assess suboptimality of value function approximation
eval_ub_freq|int|5| If eval_ub==True, requency of closed-loop suboptimality tests
eval_ub_final|Boolean|False| Run closed-loop simulations of policy induced by final VF approximation, to assess its suboptimality
eval_convergence|Boolean|False| Evaluate mean/max Bellman error and value function integral at intermediate iterations
eval_convergence_freq|int|5| If eval_convergence==True, frequency of evaluations
n_independent_x|int|100| Number of independent samples to use to measure performance under settings below
eval_ind_bellman|Boolean|False| Evaluate Bellman error and closed-loop upper bound at independent _x_ samples at intermediate iterations
eval_ind_bellman_freq|int|5| If eval_ind_bellman==True, frequency of evaluation
eval_ind_integral|Boolean|False| Evaluate VF integral over independent _x_ samples at intermediate iterations
eval_ind_integral_freq|int|5| If eval_ind_integral==True, frequency of evaluation

The `outputs_in` dictionary chooses what outputs are generated during the iterations:

Key | Data type | Default value | Description
---|---|---|---
cl_plot_j|Boolean|False|Run a closed-loop simulation at intermediate iterations, from x_0 = ones(m)
cl_plot_freq|int|20| Frequency of closed-loop simulations
cl_plot_final|Boolean|False| Run closed-loop simulations at end to evaluate final VF suboptimality
vfa_plot_j|Boolean|False| Plot value function approximation at intermediate iterations (if dimension low enough)
vfa_plot_freq|int|1| Frequency of plots of intermediate value function approximation
vfa_plot_final|Boolean|True| Plot final value function approximation
policy_plot_j|Boolean|False|  Plot control policy arising from VF approximation at intermediate iterations
policy_plot_freq|int|5| Frequency of plotting control policy arising from intermediate VF approximation iterations
policy_plot_final|Boolean|True| Plot control policy arising from final VF approximation
suppress_all|Boolean|False| Suppress all but essential outputs

## Output format

The function `print_function_approximation()` prints, and optionally saves to a text file, an explicit representation in string form of the final value function.

The function `save_function_approximation()` creates a .mat file containing the constant, linear, and quadratic components of the lower-bounding functions that define the V- or Q-function approximation.

Example usage:

```python
vfa.print_function_approximation(command_line=True, save=False)
vfa.save_function_approximation()
```

The .mat file generated by the latter function contains the following entries, where _I_ is the number of GDDP iterations that added a lower bounding function, _n_ is the state dimension, and _m_ is the input dimension:

Function saved | Variable | Dimension | Description
---------|----------|-----------|------------
V | g_const | ( _I_+1, 1) | Constant component of VF lower bounding function
V | g_lin | ( _I_+1, _n_) | Linear component of VF lower bounding function
V | g_quad | ( _I_+1, _n_, _n_) | Quadratic (Hessian) component of VF lower bounding function
Q | q_const | ( _I_+1, 1) | Constant component of VF lower bounding function
Q | q_x_lin | ( _I_+1, _n_) | Linear x component of VF lower bounding function
Q | q_x_quad | ( _I_+1, _n_, _n_) | Quadratic (Hessian) x component of VF lower bounding function
Q | q_u_lin | ( _I_+1, _m_) | Linear u component of VF lower bounding function
Q | q_u_quad | ( _I_+1, _m_, _m_) | Quadratic (Hessian) u component of VF lower bounding function

## References

[1] J. Warrington, P. Beuchat, J. Lygeros, "Generalized Dual Dynamic Programming for Infinite Horizon Problems in Continuous State and Action Spaces", _IEEE Transactions on Automatic Control, to appear December 2019._

[2] J. Warrington, "Learning Continuous Q-Functions via generalized Benders cuts", _European Control Conference 2019, Naples, Italy, June 2019._
