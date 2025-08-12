import xpress as xp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import time
import sys
import pdb

# In this Python file, we solve the Euclidean Steiner Tree problem using the Fampa-Maculan mathematical
# programming model and using a Benders Branch and cut approach for the problem.
# The mathematical programming model given by Fampa-Maculan is defined as follows:
# Given a set P of terminal points, and setting the number of Steiner points equal
# to the number of terminal points minus 2, the model is formulated as follows:

# Minimize:  sum_{p<q ∈ X} v_pq + sum_{p ∈ P, q ∈ X} w_pq
#s.t
# sum_{q ∈ X} y_pq == 1   ∀ p ∈ P
# sum_{p ∈ P} y_pq + sum_{p < q ∈ X} z_pq + sum_{p > q ∈ X} z_qp == 3   ∀ q ∈ X
# sum_{p < q ∈ X} z_pq == 1   ∀ q ∈ X, q > 1
# ||x_q - x_p||^2 ≤ s_pq²   ∀ p < q ∈ X
# ||x_q[k] - zeta_p[k]||^2 ≤ t_pq²   ∀ p ∈ P, q ∈ X
# w_pq ≥ t_pq - M_p(1 - y_pq)   ∀ p ∈ P, q ∈ X
# v_pq ≥ s_pq - M(1 - z_pq)   ∀ p < q ∈ X
# y_pq ∈ {0,1}  ∀ p ∈ P, q ∈ X
# z_pq ∈ {0,1}  ∀ p,q ∈ X, p<q
# x_p ∈  R^n

#While the Master problem and the Subproblem of Benders decomposition are defined as follow:

# Master problem

# Minimize:  theta
# s.t
# sum_{q ∈ X} y_pq == 1   ∀ p ∈ P
# sum_{p ∈ P} y_pq + sum_{p < q ∈ X} z_pq + sum_{p > q ∈ X} z_qp == 3   ∀ q ∈ X
# sum_{p < q ∈ X} z_pq == 1   ∀ q ∈ X, q > 1
# y_pq ∈ {0,1}  ∀ p ∈ P, q ∈ X
# z_pq ∈ {0,1}  ∀ p,q ∈ X, p<q
# Feasibility or Optimality cut

# Subproblem

# Minimize:  sum_{p<q ∈ X} v_pq + sum_{p ∈ P, q ∈ X} w_pq
#s.t
# ||x_q - x_p||^2 ≤ s_pq²   ∀ p < q ∈ X
# ||x_q[k] - zeta_p[k]||^2 ≤ t_pq²   ∀ p ∈ P, q ∈ X
# w_pq ≥ t_pq - M_p(1 - y_pq)   ∀ p ∈ P, q ∈ X
# v_pq ≥ s_pq - M(1 - z_pq)   ∀ p < q ∈ X
# x_p ∈  R^n

#To implement the Benders Branch and cut approach we use 2 type of callbacks that are preintsol and optnode.

class Callbacks:
    def __init__(self):
        self.Integer_solution = []
        self.Integer_index = []
        self.best_solution = 0
        self.solution_dictionary = {}
        self.optimum_dictionary = {}
        self.iterations=0

    def cb_preintsol(self, problem, data, soltype, cutoff):
        # This callback allow us to declares a user integer solution callback function,
        # called when an integer solution is found by heuristics or during the branch and bound search,
        # but before it is accepted by the Optimizer.
        # In particular in this function we create the cuts and we will store them, since in this callback
        # is not possible to add cuts, and we update the cutoff with the value of the dsubproblem

        try:
            mindex = data[0]
            P = data[1]
            X = data[2]
            Mp = data[3]
            M = data[4]
            coordinates_p = data[5]
            ypq = data[6]
            zpq = data[7]
            theta = data[8]
            coordinates_column = data[9]
            ObjAbsAccuracy = 0.00001
            self.iterations+=1
            print('===================== PREINTSOL ========================')
            ypq_vals = problem.getCallbackSolution(list(ypq.values()))
            ypq_solution = dict(zip(ypq.keys(), ypq_vals))
            zpq_vals = problem.getCallbackSolution(list(zpq.values()))
            zpq_solution = dict(zip(zpq.keys(), zpq_vals))
            new_solution = list(ypq_vals) + list(zpq_vals)
            key = tuple(new_solution)

            if key not in self.solution_dictionary:
                optimum, valori, duali = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution, X, P,
                                                    coordinates_column)
                self.solution_dictionary[key] = duali
                self.optimum_dictionary[key] = optimum
            else:
                duali = self.solution_dictionary[key]
                optimum = (self.optimum_dictionary[key])
            (colind, cutcoef, rhs_final, index_y, index_z) = cut_generation(duali, coordinates_p, Mp, M, ypq, zpq,
                                                                            theta, P, X, coordinates_column)
            cuttype = [1]
            rowtype = ['L']
            rhs = [rhs_final]
            start = [0, len(colind)]
            thetasol = problem.getCallbackSolution(theta)
            print(
                f"Check cut condition: theta = {thetasol}, subproblem value = {optimum}, ObjAbsAccuracy = {ObjAbsAccuracy}")
            if thetasol >= optimum - ObjAbsAccuracy:
                print('===================== PREINTSOL : FALSE ========================')
                return (False, optimum)
            else:
                """
                value=0
                for j, (p, q) in enumerate(index_y):
                    value += ypq_val[p, q] * cutcoef[j]
                for j, (p, q) in enumerate(index_z):
                    value = value + zpq_val[p, q] * cutcoef[j + len(index_y)]
                if value - rhs_final - optimum_general > 0:
                    print("TAGLIO VIOLATO IN PREINTSOL")
                    #pdb.set_trace()
                """
                helplist = []
                problem.storecuts(2, cuttype, rowtype, rhs, start, helplist, colind, cutcoef)
                mindex.append(helplist[0])
                print(optimum)
                cutcoef.pop()
                incumbent = problem.attributes.mipobjval
                print(incumbent)
                k,ordered = Cover_cut(incumbent, cutcoef, rhs_final)
                if k != None:
                    indexes = [ordered[i][0] for i in range(k)]
                    colinds = [colind[idx] for idx in indexes]
                    print(colinds)
                    cutcoef= [1 for i in range(k)]
                    print(cutcoef)
                    cuttype = [1]
                    rowtype = ['L']
                    rhs = [k-1]
                    start = [0, len(colinds)]
                    helplist = []
                    problem.storecuts(2, cuttype, rowtype, rhs, start, helplist, colinds, cutcoef)
                    mindex.append(helplist[0])
                if thetasol >= self.best_solution + ObjAbsAccuracy:
                    self.best_solution = thetasol
                    soluzione = list(ypq_solution.values()) + list(zpq_solution.values()) + [optimum]
                    ypq_indices = [problem.getIndex(ypq[p, q]) for p in P for q in X]
                    zpq_indices = [problem.getIndex(zpq[p, q]) for p in X for q in X if p < q]
                    theta_index = problem.getIndex(theta)
                    indici = list(ypq_indices) + list(zpq_indices) + [theta_index]
                    self.Integer_index.append(indici)
                    self.Integer_solution.append(soluzione)
                print('===================== PREINTSOL:TRUE ========================')
                if(self.iterations==14):
                    print("ypq", ypq_solution)
                    print("zpq",zpq_solution)
                    print("theta",thetasol)

                return (True, optimum)
        except Exception as e:
            print("ERRORE:", e)

    def cb_optnode(self, problem, data):
        # This callback allow us to declares an optimal node callback function, called during the branch and bound search, after
        # the LP relaxation has been solved for the current node, and after any internal cuts and
        # heuristics have been applied, but before the Optimizer checks if the current node should be branched.
        # In particular in this function we add the cuts stored in preintsol, generate new cuts from the resolution of subproblem
        # and add integeer solution to the solver
        try:
            print('===================== OPTNODE ========================')
            mindex = data[0]
            P = data[1]
            X = data[2]
            Mp = data[3]
            M = data[4]
            ypq = data[6]
            zpq = data[7]
            theta = data[8]
            coordinates_column = data[9]
            ObjAbsAccuracy = 0.00001
            coordinates_p = data[5]
            ypq_vals = problem.getCallbackSolution(list(ypq.values()))
            ypq_solution = dict(zip(ypq.keys(), ypq_vals))
            zpq_vals = problem.getCallbackSolution(list(zpq.values()))
            zpq_solution = dict(zip(zpq.keys(), zpq_vals))
            thetasol = problem.getCallbackSolution(theta)
            for i, val in zip(self.Integer_index, self.Integer_solution):
                problem.addmipsol(val, i, 'From_preintsol')
                del self.Integer_index[:]
                del self.Integer_solution[:]
            if len(mindex) > 0:
                problem.loadcuts(0, -1, mindex)
                print(len(mindex), " cuts added")
                del mindex[:]

            new_solution = list(ypq_vals) + list(zpq_vals)
            key = tuple(new_solution)

            if key not in self.solution_dictionary:
                optimum, valori, duali = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution, X, P,
                                                    coordinates_column)
                self.solution_dictionary[key] = duali
                self.optimum_dictionary[key] = optimum
            else:
                duali = self.solution_dictionary[key]
                optimum = self.optimum_dictionary[key]
            soluzione = list(ypq_solution.values()) + list(zpq_solution.values()) + [optimum]
            ypq_indices = [problem.getIndex(ypq[p, q]) for p in P for q in X]
            zpq_indices = [problem.getIndex(zpq[p, q]) for p in X for q in X if p < q]
            theta_index = problem.getIndex(theta)
            indici = list(ypq_indices) + list(zpq_indices) + [theta_index]
            integer = self.is_solution_integer(soluzione)

            if integer == True:
                if thetasol >= self.best_solution + ObjAbsAccuracy:
                    problem.addmipsol(soluzione, indici, 'from_optnode')
                    self.best_solution = thetasol
            print(f"Check cut condition: theta = {thetasol}, subproblem value = {optimum}, ObjAbsAccuracy = {ObjAbsAccuracy}")
            if thetasol >= optimum - ObjAbsAccuracy:
                print('===================== OPTNODE: EXIT ========================')
            else:
                (colind, cutcoef, rhs_final, index_y, index_z) = cut_generation(duali, coordinates_p, Mp, M, ypq, zpq,
                                                                                theta, P, X, coordinates_column)
                cuttype = [1]
                rowtype = ['L']
                rhs = [rhs_final]
                start = [0, len(colind)]
                print(f"Prima dei tagli - LB: {problem.attributes.lpobjval}, Best: {problem.attributes.bestbound}")
                problem.addcuts(cuttype, rowtype, rhs, start, colind, cutcoef)
                if integer == True:
                    incumbent = problem.attributes.mipobjval
                    print(incumbent)
                    (k, ordered) = Cover_cut(incumbent, cutcoef, rhs_final)
                    if k != None:
                        indexes = [ordered[i][0] for i in range(k)]
                        colinds = [colind[idx] for idx in indexes]
                        print(colinds)
                        cutcoef = [ordered[i][1] for i in range(k)]
                        cuttype = [1]
                        rowtype = ['L']
                        rhs = [k - 1]
                        start = [0, len(colinds)]
                        problem.addcuts(cuttype, rowtype, rhs, start, colinds, cutcoef)
                """
                    value=0
                    for j, (p,q) in enumerate(index_y):
                        value += ypq_val[p, q] * cutcoef[j]
                    for j, (p, q) in enumerate(index_z):
                        value= value+zpq_val[p,q]*cutcoef[j+len(index_y)]
                    if value-rhs_final-optimum_general>0:
                        print("TAGLIO VIOLATO IN OPTNODE")
                        pdb.set_trace()
                    """
                # else:
                # pdb.set_trace()
            incumbent = problem.attributes.mipobjval
            current_bound = problem.attributes.lpobjval
            global_bound = problem.attributes.bestbound
            print(f"UB: {incumbent}")
            print(f"Current LB: {current_bound}")
            print(f"Global LB: {global_bound}")
            print(
                f"Gap corrente calcolato: {(problem.attributes.mipobjval - problem.attributes.bestbound) / abs(problem.attributes.mipobjval)}")
            node = problem.attributes.currentnode
            print(f"Node: {node}")
            print('===================== OPTNODE: RIENTRO ========================')
            return 0
        except Exception as e:
            print("ERRORE:", e)

    def is_solution_integer(self, valori, tolerance=1e-6):
        for value in valori:
            if abs(value - round(value)) > tolerance:
                return False
        return True
def Cover_cut(UB,coef,rhs):
    if UB>1000:
        return (None, [])
    else:
        print(UB)
        print("rhs", rhs)
        eps=1e-2
        coef_enumerate= list(enumerate(coef))
        ordered= sorted(coef_enumerate, key=lambda x: x[1])
        sum_coef=0
        for k in range(len(coef)):
            sum_coef+=ordered[k][1]
            print(sum_coef)
            if sum_coef > eps+ UB+ rhs:
                print(k+1)
                print(ordered)
                return (k+1, ordered)
        return(None,[])


def MINLP_formulation(Mp, M, coordinates_p, X, P, d, coordinates_column):
    # problem creation
    total = xp.problem(name="Steiner Tree")
    xp_var = {}
    ypq = {(p, q): total.addVariable(name=f"y_{p}_{q}", vartype=xp.binary) for p in P for q in X}
    zpq = {(p, q): total.addVariable(name=f"z_{p}_{q}", vartype=xp.binary) for p in X for q in X if p < q}
    for k in X:
        xp_var[k] = {}
        for dim in coordinates_column:
            xp_var[k][dim] = total.addVariable(name=f"xp_{k}_{dim}", vartype=xp.continuous)
    spq = {(p, q): total.addVariable(name=f"s_{p}_{q}",
                                     vartype=xp.continuous) for p in X for q in X if p < q}
    tpq = {(p, q): total.addVariable(name=f"t_{p}_{q}",
                                     vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): total.addVariable(name=f"w_{p}_{q}",
                                     vartype=xp.continuous) for p in P for q in X}
    vpq = {(p, q): total.addVariable(name=f"v_{p}_{q}",
                                     vartype=xp.continuous) for p in X for q in X if p < q}

    # Objective function
    obj = (xp.Sum(vpq[p, q] for p in X for q in X if p < q) + xp.Sum(wpq[p, q] for p in P for q in X))
    total.setObjective(obj, sense=xp.minimize)

    # Constraints
    for p in P:
        total.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

    for q in X:
        total.addConstraint(
            xp.Sum(ypq[p, q] for p in P)
            + xp.Sum(zpq[p, q] for p in X if p < q)
            + xp.Sum(zpq[q, p] for p in X if p > q) == 3
        )

    for q in X:
        if q > 1:
            total.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

    # for q in X:
    #   total.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)

    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (xp_var[q][coordinates_column[k]] - xp_var[p][coordinates_column[k]]) ** 2
                    for k in range(d)
                )
                total.addConstraint(lhs <= spq[p, q] ** 2)

    # Vincoli tra punti dati (P) e Steiner
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (xp_var[q][coordinates_column[k]] - coordinates_p[p][coordinates_column[k]]) ** 2
                for k in range(d)
            )
            total.addConstraint(lhs2 <= tpq[p, q] ** 2)
    for p in P:
        for q in X:
            total.addConstraint(wpq[p, q] >= tpq[p, q] - Mp[p] * (1 - ypq[p, q]))
    for p in X:
        for q in X:
            if p < q:
                total.addConstraint(vpq[p, q] >= spq[p, q] - M * (1 - zpq[p, q]))

    # Problem solution
    total.controls.threads = 1
    total.controls.maxtime = -7200
    total.solve()
    status = total.attributes.solstatus
    LB = total.attributes.objval
    print("LB", LB)
    if status == 1:
        xp_solution = {key: total.getSolution(xp_var[key]) for key in xp_var}
        ypq_solution = {key: total.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: total.getSolution(zpq[key]) for key in zpq}
        tpq_solution = {key: total.getSolution(tpq[key]) for key in tpq}
        print("Optimal solution found")
        print("xp:", xp_solution)
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
        print("tpq", tpq_solution)
    else:
        print(f"Problem status: {status}")
    return (ypq_solution, zpq_solution, xp_solution, LB)


def subproblem(Mp, M, coordinates_p, ypq_values, zpq_values, X, P, coordinate_columns):
    subproblem = xp.problem(name="Subproblem")
    xp_var = {}
    for k in X:
        xp_var[k] = {}
        for dim in coordinate_columns:
            xp_var[k][dim] = subproblem.addVariable(name=f"xp_{k}_{dim}", vartype=xp.continuous)
    spq = {(p, q): subproblem.addVariable(name=f"s_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    tpq = {(p, q): subproblem.addVariable(name=f"t_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): subproblem.addVariable(name=f"w_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    vpq = {(p, q): subproblem.addVariable(name=f"v_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    deltapq = {(p, q, coord): subproblem.addVariable(vartype=xp.continuous, lb=-1e4, name=f"delta_{p}_{q}_{coord}")
               for p in P for q in X for coord in coordinate_columns}

    gammapq = {(p, q, coord): subproblem.addVariable(vartype=xp.continuous, lb=-1e4, name=f"gamma_{p}_{q}_{coord}")
               for p in X for q in X if p < q for coord in coordinate_columns}
    # Objective function
    obj = (xp.Sum(vpq[p, q] for p in X for q in X if p < q) +
           xp.Sum(wpq[p, q] for p in P for q in X))
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints = []
    constraints1 = []
    constraints2 = []
    constraints3 = []
    constraints4 = []
    constraints5 = []
    constraints6 = []
    constraints7 = []
    delta_constraints = {coord: [] for coord in coordinate_columns}
    gamma_constraints = {coord: [] for coord in coordinate_columns}
    for p in P:
        for q in X:
            for coord in coordinate_columns:
                constraint = deltapq[(p, q, coord)] == (-coordinates_p[p][coord] + xp_var[q][coord])
                subproblem.addConstraint(constraint)
                delta_constraints[coord].append(constraint)
    for p in X:
        for q in X:
            if p < q:
                for coord in coordinate_columns:
                    constraint = gammapq[(p, q, coord)] == (xp_var[q][coord] - xp_var[p][coord])
                    subproblem.addConstraint(constraint)
                    gamma_constraints[coord].append(constraint)

    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum((gammapq[(p, q, coord)]) ** 2 for coord in coordinate_columns)
                constraint = -lhs + spq[p, q] ** 2 >= 0
                subproblem.addConstraint(constraint)
                constraints.append(constraint)

    for p in P:
        for q in X:
            lhs2 = xp.Sum((deltapq[(p, q, coord)]) ** 2 for coord in coordinate_columns)
            constraint1 = -lhs2 + tpq[p, q] ** 2 >= 0
            subproblem.addConstraint(constraint1)
            constraints1.append(constraint1)

    for p in P:
        for q in X:
            constraint2 = wpq[p, q] - tpq[p, q] >= -Mp[p] * (1 - ypq_values[p, q])
            subproblem.addConstraint(constraint2)
            constraints2.append(constraint2)
    for p in X:
        for q in X:
            if p < q:
                constraint3 = vpq[p, q] - spq[p, q] >= -M * (1 - zpq_values[p, q])
                subproblem.addConstraint(constraint3)
                constraints3.append(constraint3)
    subproblem.setControl('outputlog', 0)
    subproblem.controls.threads = 1
    subproblem.controls.maxtime = -120
    subproblem.solve()
    optimum = subproblem.attributes.objval
    status = subproblem.attributes.solvestatus
    print(status)
    if status == 3:
        xp_solution = subproblem.getSolution(xp_var)
        tpq_solution = subproblem.getSolution(tpq)
        spq_solution = subproblem.getSolution(spq)
        wpq_solution = subproblem.getSolution(wpq)
        vpq_solution = subproblem.getSolution(vpq)
        delta_solution = subproblem.getSolution(deltapq)
        gamma_solution = subproblem.getSolution(gammapq)
        multipliers = subproblem.getDuals(constraints)
        multipliers1 = subproblem.getDuals(constraints1)
        multipliers2 = subproblem.getDuals(constraints2)
        multipliers3 = subproblem.getDuals(constraints3)
        multipliers_delta = {
            coord: subproblem.getDuals(delta_constraints[coord])
            for coord in coordinate_columns
        }
    valori = (xp_solution, tpq_solution, spq_solution, wpq_solution, vpq_solution, delta_solution, gamma_solution)
    duali = (multipliers, multipliers1, multipliers2, multipliers3, multipliers_delta)
    return (optimum, valori, duali)


def cut_generation(duali, coordinates_p, Mp, M, ypq, zpq, theta, P, X, coordinate_columns):
    colind = []
    cutcoef = []
    rhs_constant = 0.0
    epsilon = 1e-7
    multipliers4 = duali[4]
    multipliers3 = duali[3]
    multipliers2 = duali[2]
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        for coord in coordinate_columns:
            rhs_constant -= multipliers4[coord][j] * coordinates_p[p][coord]
    index_y = []
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        if abs(multipliers2[j]) >= epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
            colind.append(ypq[p, q])
            cutcoef.append(multipliers2[j] * Mp[p])
            index_y.append([p, q])
        elif multipliers2[j] <= -epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
    index_z = []
    for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q):
        if abs(multipliers3[j]) >= epsilon:
            rhs_constant -= multipliers3[j] * M
            colind.append(zpq[p, q])
            cutcoef.append(multipliers3[j] * M)
            index_z.append([p, q])
        elif multipliers3[j] <= -epsilon:
            rhs_constant -= multipliers3[j] * M

    colind.append(theta)
    cutcoef.append(-1.0)
    rhs_final = -rhs_constant
    return (colind, cutcoef, rhs_final, index_y, index_z)

def cut_inter(duali,coordinates_p, Mp, M, ypq, zpq, theta, P, X, coordinate_columns,optimum):
    colind = []
    cutcoef = []
    rhs_constant = 0.0
    epsilon = 1e-7
    multipliers4 = duali[4]
    multipliers3 = duali[3]
    multipliers2 = duali[2]
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        for coord in coordinate_columns:
            rhs_constant -= multipliers4[coord][j] * coordinates_p[p][coord]
    index_y = []
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        if abs(multipliers2[j]) >= epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
            colind.append(ypq[p, q])
            cutcoef.append(int(multipliers2[j] * Mp[p]))
            index_y.append([p, q])
        elif multipliers2[j] <= -epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
    index_z = []
    for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q):
        if abs(multipliers3[j]) >= epsilon:
            rhs_constant -= multipliers3[j] * M
            colind.append(zpq[p, q])
            cutcoef.append(int(multipliers3[j] * M))
            index_z.append([p, q])
        elif multipliers3[j] <= -epsilon:
            rhs_constant -= multipliers3[j] * M

    rhs_constant += optimum
    rhs_final = -int(rhs_constant)
    return (colind, cutcoef, rhs_final, index_y, index_z)

def node_cutoff_callback(problem, mindex, node):
    print("SONO IN NODE CUTOFF")
    print(f"LP obj: {problem.attributes.lpobjval}")
    print(f"MIP obj: {problem.attributes.mipobjval}")
    print(f"Best bound: {problem.attributes.bestbound}")
    print(f"Active nodes: {problem.attributes.activenodes}")
    print(f"LP status: {problem.attributes.lpstatus}")


def log_callback(problem, mindex, msg, msgtype):
    print(f"LOG: {msg}")
    print(f"Final incumbent: {problem.attributes.mipobjval}")
    print(f"Final bound: {problem.attributes.bestbound}")


def cb_intsol(problem, mindex):
    incumbent = problem.attributes.mipobjval
    current_bound = problem.attributes.lpobjval
    global_bound = problem.attributes.bestbound
    print("Sono in INTSOL")
    print(f"UB: {incumbent}")
    print(f"Current LB: {current_bound}")
    print(f"Global LB: {global_bound}")
    print(f"Gap: {incumbent - global_bound}")
    print(f"Gap relativo target: {problem.controls.miprelstop}")
    print(f"Gap assoluto target: {problem.controls.mipabsstop}")
    print(
        f"Gap corrente calcolato: {(problem.attributes.mipobjval - problem.attributes.bestbound) / abs(problem.attributes.mipobjval)}")


def nodeInfeasible(problem, object):
    print("SONO IN NODE INFEASIBLE")
    node = problem.attributes.currentnode
    print("Node {0} infeasible".format(node))


def plot_the_graph(file_name, type_resolution, coordinates_p, xp_solution, ypq_solution, zpq_solution, P, X, d,
                   coordinate_columns):
    # Building the plots
    if d == 2:
        dim1, dim2 = coordinate_columns[:2]
        plt.figure(figsize=(8, 8))

        for p in P:
            plt.scatter(coordinates_p[p][dim1], coordinates_p[p][dim2], color='blue', s=50)

        for p in X:
            plt.scatter(xp_solution[p][dim1], xp_solution[p][dim2], color='red', s=150,  facecolors='none')

        for (p, q), val in ypq_solution.items():
            if val == 1.0:
                x1, y1 = coordinates_p[p][dim1], coordinates_p[p][dim2]
                x2, y2 = xp_solution[q][dim1], xp_solution[q][dim2]
                plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1)

        for (p, q), val in zpq_solution.items():
            if val == 1.0:
                x1, y1 = xp_solution[p][dim1], xp_solution[p][dim2]
                x2, y2 = xp_solution[q][dim1], xp_solution[q][dim2]
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

        plt.title(f"Steiner Tree Solution")
        plt.xlabel(dim1)
        plt.ylabel(dim2)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='datalim')
        #plt.show()
        plt.savefig(f'solutions-6Aug/Soluzione_{file_name}-type{type_resolution}.png')

    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D

        dim1, dim2, dim3 = coordinate_columns[:3]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for p in P:
            ax.scatter(coordinates_p[p][dim1], coordinates_p[p][dim2], coordinates_p[p][dim3],
                       color='blue', s=50)

        for p in X:
            ax.scatter(xp_solution[p][dim1], xp_solution[p][dim2], xp_solution[p][dim3],
                       color='red', s=100,  facecolors='none')

        for (p, q), val in ypq_solution.items():
            if val == 1.0:
                x1, y1, z1 = coordinates_p[p][dim1], coordinates_p[p][dim2], coordinates_p[p][dim3]
                x2, y2, z2 = xp_solution[q][dim1], xp_solution[q][dim2], xp_solution[q][dim3]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=1)

        for (p, q), val in zpq_solution.items():
            if val == 1.0:
                x1, y1, z1 = xp_solution[p][dim1], xp_solution[p][dim2], xp_solution[p][dim3]
                x2, y2, z2 = xp_solution[q][dim1], xp_solution[q][dim2], xp_solution[q][dim3]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=2)

        ax.set_title(f"Steiner Tree Solution")
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        ax.set_zlabel(dim3)
        #plt.show()
        plt.savefig(f'solutions-6Aug/Soluzione_{file_name}-type{type_resolution}.png')

    else:
        print(f" Impossible to do the plot for d = {d}.")


def solve():
    import pandas as pd
    import numpy as np
    import sys
    import time

    start_time = time.time()

    file_name = sys.argv[1]
    type_resolution = sys.argv[2]

    assert type_resolution in ['0', '1']

    # Leggi il file
    data = pd.read_csv(file_name, sep='\s+')
    data = data.drop(data.columns[0], axis=1)
    data = data.reset_index()
    data['id'] = data.index

    exclude_keywords = ['level', 'index', 'id']
    coordinate_columns = [col for col in data.select_dtypes(include=[np.number]).columns
                          if not any(key in col for key in exclude_keywords)]
    d = len(coordinate_columns)
    print(d)
    coordinates_p = data.set_index('id')[coordinate_columns].T.to_dict()
    print(coordinates_p)
    P = range(len(data))
    num_steiner_nodes = len(data) - 2
    X = range(num_steiner_nodes)

    Mp = []
    ObjAbsAccuracy = 0.00001

    # Calcolo Mp[p]
    for p in P:
        max_dist = 0
        for z in P:
            if z != p:
                distanza1 = np.linalg.norm(
                    np.array([coordinates_p[p][dim] - coordinates_p[z][dim] for dim in coordinate_columns])
                )
                if distanza1 > max_dist:
                    max_dist = distanza1
        Mp.append(max_dist)

    # Calcolo M massimo globale
    M = 0
    for p in P:
        for z in P:
            if z != p:
                distanza1 = np.linalg.norm(
                    np.array([coordinates_p[p][dim] - coordinates_p[z][dim] for dim in coordinate_columns])
                )
                if distanza1 > M:
                    M = distanza1
    print(M)
    print(Mp)
    if type_resolution == '0':
        (ypq_solution, zpq_solution, xp_solution, optimum_general) = MINLP_formulation(Mp, M, coordinates_p, X, P, d,
                                                                                       coordinate_columns)
        print("Optimal solution found")
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
        print(optimum_general)
    # Start defining the MASTER PROBLEM
    elif type_resolution == '1':
        # Decision variable
        problem = xp.problem(name="Master problem")
        ypq = {(p, q): problem.addVariable(name=f"y_{p}_{q}", vartype=xp.binary) for p in P for q in X}
        zpq = {(p, q): problem.addVariable(name=f"z_{p}_{q}", vartype=xp.binary) for p in X for q in X if p < q}
        theta = problem.addVariable(vartype=xp.continuous, name="theta")
        # Objective function
        obj = (theta)
        problem.setObjective(obj, sense=xp.minimize)
        # constraints
        for p in P:
            problem.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

        for q in X:
            problem.addConstraint(
                xp.Sum(ypq[p, q] for p in P)
                + xp.Sum(zpq[p, q] for p in X if p < q)
                + xp.Sum(zpq[q, p] for p in X if p > q) == 3
            )

        for q in X:
            if q > 0:
                problem.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

        # for q in X:
        #   problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)

        g = Callbacks()
        mindex = []
        # problem.controls.cutfreq = 0
        problem.controls.miprelstop = 0.0001
        data = (mindex, P, X, Mp, M, coordinates_p, ypq, zpq, theta, coordinate_columns)
        problem.setControl("presolve", 0)
        problem.setControl('CUTSTRATEGY', 0)
        problem.setControl("mippresolve", 0)
        problem.setControl("symmetry", 0)
        problem.setControl("mipdualreductions", 0)
        # problem.setControl("miprefineiterlimit", 0)
        problem.addcbpreintsol(g.cb_preintsol, data, 0)
        # problem.addcbnodecutoff(node_cutoff_callback,mindex,0)
        # problem.addcbmessage(log_callback, mindex, 0)
        # problem.addcbintsol(cb_intsol, None,0)
        # problem.addcbnodecutoff(cb_nodecutoff,None,0)
        # problem.addcbinfnode(nodeInfeasible, None, 0)
        problem.addcboptnode(g.cb_optnode, data, 0)
        problem.controls.threads = 1
        problem.controls.maxtime = -7200
        problem.solve()
        LB = problem.attributes.objval
        print("LB", LB)
        status = problem.attributes.solvestatus
        if status == 3:
            ypq_solution = problem.getSolution(ypq)
            zpq_solution = problem.getSolution(zpq)

            print("Optimal solution found")
            print("ypq:", ypq_solution)
            print("zpq:", zpq_solution)
        else:
            print(f"Error status: {status}")
        (optimum, valori, duali) = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution, X, P,
                                              coordinate_columns)
        xp_solution = valori[0]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.6f} seconds")
    fname = file_name.split('.')[0].split('/')[-1]
    plot_the_graph(fname, type_resolution, coordinates_p, xp_solution, ypq_solution, zpq_solution, P, X, d,
                   coordinate_columns)


if __name__ == '__main__':
    file_name = sys.argv[1].split('.')[0].split('/')[-1]
    type_sol = sys.argv[2]
    #with open(f'results-6Aug/output-{file_name}-type{type_sol}.txt', 'w') as sys.stdout:
    solve()