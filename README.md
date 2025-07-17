# Project aim
The aim of this project is to explore the Euclidean Steiner Tree problem, seeking a faster approach compared to the mathematical programming formulation proposed by Fampa and Maculan in 2004, which is a MINLP problem and known to be NP-hard.
The main idea is to apply Benders decomposition to the problem, in order to separate the resolution of the integer and continuous variables.

# Main work
The workflow begins with implementing the Fampa-Maculan formulation for the Euclidean Steiner Tree problem, analyzing its computational drawbacks.
Subsequently, we attempt to develop a Benders decomposition using the cutting plane method. The approach involves iteratively solving a master problem and, at each iteration, adding a cut generated from the corresponding subproblem.

We explore two variants of Benders decomposition:

1. First variant: The integer variables are treated as the complicating variables and are included in the master problem, while the continuous variables are handled in the subproblem.

2. Second variant: The continuous variables are treated as the complicating variables and are included in the master problem, while the subproblem involves only the integer variables.

The second variant does not yield any significant improvement in the final result. This is because the subproblem, composed only of integer variables, is frequently infeasible. Consequently, we can only add feasibility cuts to the master problem. However, feasibility cuts are derived from infeasibility directions, which are well-defined only in continuous problems. To address this, we must consider the linear relaxation of the subproblem, which results in weaker cuts that do not necessarily lead to a faster approach compared to the original full-variable formulation.

In the first variant, to further improve performance beyond the standard cutting plane method, we integrate Benders decomposition into a Branch-and-Cut framework using callbacks. With this approach, we avoid solving the master problem iteratively. Instead, the subproblem is solved at each node of the Branch-and-Cut tree, allowing us to dynamically generate Benders cuts during the search process.



# Code
To verify whether the use of Benders decomposition leads to faster performance compared to solving the full mathematical formulation, we implemented both models using FICO Xpress 9.6 with its Python interface.

This repository contains four main Python files:

1. Cutting_plane_Benders.py – Executes the first Benders decomposition approach using the cutting plane method (with integer variables in the master problem).

2. Benders_decomposition2.py – Executes the second Benders decomposition approach using the cutting plane method (with continuous variables in the master problem).

3. Steiner_Tree.py – Implements and solves the full mathematical formulation of the Euclidean Steiner Tree problem as proposed by Fampa and Maculan.

4. Steiner_Tree_Benders.py – Contains both the full formulation and the Branch-and-Cut implementation of the first Benders decomposition.

# How to run Steiner_Tree_Benders.py
To execute the full mathematical formulation of the Steiner Tree problem:
```
python Steiner_Tree_Benders.py Name_instance 0
```
To execute the Benders decomposition with the Branch-and-Cut approach:
```
python Steiner_Tree_Benders.py Name_instance 1
```
Replace Name_instance with the actual name of the problem instance you want to solve.

# References
- Marcia Fampa and Nelson Maculan. Using a conic formulation for finding steiner min-
imal trees: Theory and practice in optimization. guest editors: José mario martínez
and jin yun yuan. Numerical Algorithms, 35, 04 2004.
- Benders, J. F. (Sept. 1962), "Partitioning procedures for solving mixed-variables programming problems", Numerische Mathematik 4(3): 238–252.
