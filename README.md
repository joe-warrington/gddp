# GDDP

Generalized dual dynamic programming (GDDP) tool written in Python. This tool creates lower bounds on a given control problem's optimal value function using the Benders decomposition approach developed in [1].

## Getting Started

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

## Q functions

Some functionality also exists for working with Q functions, which are defined on state-action space rather than the state alone. The syntax is the same:

```python
from gddp import QFApproximator
qfa = QFApproximator(system=sys1)
qfa.create_qfa_model()
qfa.approximate()
```

## References

[1] J. Warrington, P. Beuchat, J. Lygeros, "Generalized Dual Dynamic Programming for Infinite Horizon Problems in Continuous State and Action Spaces", _IEEE Transactions on Automatic Control, to appear December 2019._

[2] J. Warrington, "Learning Q-Functions via generalized Benders cuts", _European Control Conference 2019, Naples, Italy, June 2019._
