# Project aim
The aim of the project is to explore the problem of Euclidean Steiner tree, looking for a faster approach respect to Mathematical programming formulation proposed by Fampa Maculan in 2004, which is a MINLP problem and it is NP-hard. 
The main idea is to use Benders decomposition for the problem, such has to separate in the resolution integeer and continuous variable.

# Main work
The work flow starts with building the Fampa Maculan formulation for the Euclidean Steiner tree problem looking to its computational disadvantages.
Afterwards, we  try to build the Benders decomposition through the cutting plane method. So the flow consinst in solving iteratively the Master problem adding in each iteration a cut generated from the subproblem.
We try in particular we two types of Benders decomposition. The first one we consider as complicated variable the integer one and we consither that one in the MAster while in the subproblem we add the continuous one
While with the second one we do the opposite, we consider as complicated variable the continuous one adding it in the Master.
The second Benders decomposition could not bring any improvement to the final result since, the subproblem, which has only integer variable, is always infeasible, so we can add to the Master problem only feasibility cut



# Code
To verify if the idea to use Benders decomposition code is faster then solving the problem with the entire formulation we build both model using as solver FICO Xpress 9.6 with Python interface.
In this repository there are more precisely 3 .py files where
1. **Cutting_plane_Benders.py**
2. **Benders_decomposition2**
3. **Steiner_Tree_Benders decomposition**
