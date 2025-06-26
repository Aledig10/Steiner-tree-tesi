import xpress as xp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

class Callbacks:
    def __init__(self):
        self.Integer_solution=[]
        self.Integer_index=[]

    def cb_preintsol(self, problem, mindex, soltype, cutoff):
        ypq_vals = problem.getCallbackSolution(list(ypq.values())) #Extract the solution in a list
        ypq_solution = dict(zip(ypq.keys(), ypq_vals)) #build the dictionary from the partial solution
        zpq_vals = problem.getCallbackSolution(list(zpq.values())) #Extract the solution in a list
        zpq_solution = dict(zip(zpq.keys(), zpq_vals)) #build the dictionary from the partial solution
        (optimum, valori, duali) = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution) #solution of the subproblem
        (colind, cutcoef, rhs_final) = cut_generation(duali, coordinates_p, Mp, M) #returns terms for building the cut
        cuttype = [1]
        rowtype = ['L']
        rhs = [rhs_final]
        start = [0, len(colind)]
        thetasol = problem.getCallbackSolution(theta) #partial solution of theta
        print(f"Check cut condition: theta = {thetasol}, Value of subproblem = {optimum}, ObjAbsAccuracy = {ObjAbsAccuracy}")
        if thetasol >= optimum - ObjAbsAccuracy:
            return (False, optimum) #if the distance between optimum and the partial solution is less than object accuracy stop
        else:
            helplist = []
            problem.storecuts(2, cuttype, rowtype, rhs, start, helplist, colind, cutcoef) #store the cut
            mindex.append(helplist[0])
            #Save the solution such to add this Integer solution to the problem through addmipsol
            soluzione = list(ypq_solution.values()) + list(zpq_solution.values()) + [optimum]
            ypq_indices = [problem.getIndex(ypq[p, q]) for p in P for q in X]
            zpq_indices = [problem.getIndex(zpq[p, q]) for p in X for q in X if p < q]
            theta_index = problem.getIndex(theta)
            indici = list(ypq_indices) + list(zpq_indices) + [theta_index]
            self.Integer_index.append(indici)
            self.Integer_solution.append(soluzione)
            return (True, optimum)

    def cb_optnode(self, problem, mindex):
        #Save the values of the partial solution
        ypq_vals = problem.getCallbackSolution(list(ypq.values()))
        ypq_solution = dict(zip(ypq.keys(), ypq_vals))
        zpq_vals = problem.getCallbackSolution(list(zpq.values()))
        zpq_solution = dict(zip(zpq.keys(), zpq_vals))
        thetasol = problem.getCallbackSolution(theta)
        if len(mindex) > 0:
            problem.loadcuts(0, -1, mindex) #load the cuts we have stored in preintsol
            print(len(mindex), " cuts added")
            del mindex[:]
        #solve the subproblem
        (optimum, valori, duali) = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution)
        soluzione = list(ypq_solution.values()) + list(zpq_solution.values()) + [optimum]
        integer = self.is_solution_integer(soluzione)
        #if the solution is integer save the solution with addmipsol
        if integer == True:
            ypq_indices = [problem.getIndex(ypq[p, q]) for p in P for q in X]
            zpq_indices = [problem.getIndex(zpq[p, q]) for p in X for q in X if p < q]
            theta_index = problem.getIndex(theta)
            indici = list(ypq_indices) + list(zpq_indices) + [theta_index]
            problem.addmipsol(soluzione, indici, 'name')
        if thetasol >= optimum - ObjAbsAccuracy:
            return 0
        else:
            #Build the cuts to add to the node
            (colind, cutcoef, rhs_final) = cut_generation(duali, coordinates_p, Mp, M)
            cuttype = [1]
            rowtype = ['L']
            rhs = [rhs_final]
            start = [0, len(colind)]
            problem.addcuts(cuttype, rowtype, rhs, start, colind, cutcoef)
            return 0

    def is_solution_integer(self, valori, tolerance=1e-6):
        #Method of the class Callbacks that checks if the solution is integer
        for value in valori:
            if abs(value - round(value)) > tolerance:
                return False
        return True

def subproblem(Mp,M,coordinates_p,ypq_values,zpq_values):
    #definition of the subproblem
    subproblem = xp.problem(name="Subproblem")
    #variables
    xp_var = {
        k: {
            'X': subproblem.addVariable(vartype=xp.continuous, name=f"xp_{k}"),
            'Y': subproblem.addVariable(vartype=xp.continuous, name=f"yp_{k}")
        }
        for k in X
    }  # x^p in R^d
    spq = {(p, q): subproblem.addVariable(name=f"s_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    tpq = {(p, q): subproblem.addVariable(name=f"t_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): subproblem.addVariable(name=f"w_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    vpq = {(p, q): subproblem.addVariable(name=f"v_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    deltapq = {(p, q, coord): subproblem.addVariable(vartype=xp.continuous, lb=-1e4, name=f"delta_{p}_{q}_{coord}")
               for p in P for q in X for coord in ['X', 'Y']}

    gammapq = {(p, q, coord): subproblem.addVariable(vartype=xp.continuous, lb=-1e4, name=f"gamma_{p}_{q}_{coord}")
               for p in X for q in X if p < q for coord in ['X', 'Y']}

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
    #Constraints
    for p in P:
        for q in X:
            constraint = deltapq[(p, q, 'X')] == (-coordinates_p[p]['X'] + xp_var[q]['X'])
            subproblem.addConstraint(constraint)
            constraints4.append(constraint)
            constraint = deltapq[(p, q, 'Y')] == (-coordinates_p[p]['Y'] + xp_var[q]['Y'])
            subproblem.addConstraint(constraint)
            constraints5.append(constraint)
    for p in X:
        for q in X:
            if p < q:
                constraint = gammapq[(p, q, 'X')] == (xp_var[q]['X'] - xp_var[p]['X'])
                subproblem.addConstraint(constraint)
                constraints6.append(constraint)
                constraint = gammapq[(p, q, 'Y')] == (xp_var[q]['Y'] - xp_var[p]['Y'])
                subproblem.addConstraint(constraint)
                constraints7.append(constraint)
    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (gammapq[(p, q, coord)]) ** 2 for coord in ['X', 'Y'])
                constraint = -lhs + spq[p, q] ** 2 >= 0
                subproblem.addConstraint(constraint)
                constraints.append(constraint)
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (deltapq[(p, q, coord)]) ** 2 for coord in ['X', 'Y'])
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
    subproblem.solve()
    optimum = subproblem.getObjVal()
    status = subproblem.getProbStatus()
    if status == xp.lp_optimal:
        #if the solution is optimal, save the variables
        xp_solution = subproblem.getSolution(xp_var)
        tpq_solution = subproblem.getSolution(tpq)
        spq_solution = subproblem.getSolution(spq)
        wpq_solution = subproblem.getSolution(wpq)
        vpq_solution = subproblem.getSolution(vpq)
        delta_solution = subproblem.getSolution(deltapq)
        gamma_solution = subproblem.getSolution(gammapq)
        # we are going to save the duals coefficients
        multipliers = subproblem.getDuals(constraints)
        multipliers1 = subproblem.getDuals(constraints1)
        multipliers2 = subproblem.getDuals(constraints2)
        multipliers3 = subproblem.getDuals(constraints3)
        multipliers4 = subproblem.getDuals(constraints4)
        multipliers5 = subproblem.getDuals(constraints5)

    valori=(xp_solution, tpq_solution,spq_solution,wpq_solution,vpq_solution,delta_solution,gamma_solution)
    duali=(multipliers,multipliers1,multipliers2,multipliers3,multipliers4,multipliers5)
    return (optimum,valori,duali)

def cut_generation(duali,coordinates_p,Mp,M):
    colind = []
    cutcoef = []
    rhs_constant = 0.0
    epsilon=1e-7
    multipliers4=duali[4]
    multipliers5=duali[5]
    multipliers3=duali[3]
    multipliers2=duali[2]
    #Build the coefficients of the cuts
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
         rhs_constant -= multipliers4[j] * coordinates_p[p]['X']

    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        rhs_constant -= multipliers5[j] * coordinates_p[p]['Y']

    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        if abs(multipliers2[j]) >= epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
            colind.append(ypq[p, q])
            cutcoef.append(multipliers2[j] * Mp[p])

        elif multipliers2[j] <= -epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]

    for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q):
        if abs(multipliers3[j]) >= epsilon:
            rhs_constant -= multipliers3[j] * M
            colind.append(zpq[p, q])
            cutcoef.append(multipliers3[j] * M)
        elif multipliers3[j] <= -epsilon:
            rhs_constant -= multipliers3[j] * M

    colind.append(theta)
    cutcoef.append(-1.0)
    rhs_final = -rhs_constant
    return(colind,cutcoef,rhs_final)

def plot_the_graph(coordinates_p,xp_solution,ypq_solution,zpq_solution):
    #Building the plots
    plt.figure(figsize=(8, 8))
    for p in P:
        plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50,
                    label='Points P' if p == list(P)[0] else "")

    for p in X:
        plt.scatter(xp_solution[p]['X'], xp_solution[p]['Y'], color='red', s=50,
                    label='Points X' if p == list(X)[0] else "")

    for (p, q), val in ypq_solution.items():
        if val == 1.0:
            x1, y1 = coordinates_p[p]["X"], coordinates_p[p]["Y"]
            x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
            if x1 is not None and x2 is not None:
                plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")

    for (p, q), val in zpq_solution.items():
        if val == 1:
            x1, y1 = xp_solution[p]["X"], xp_solution[p]["Y"]
            x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
            if x1 is not None and x2 is not None:
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

    plt.title("Points in the 2D space")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()


start_time = time.time()
file_name= sys.argv[1]
data = pd.read_csv(file_name, sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Set of given nodes
num_steiner_nodes = len(data) - 2  # Number of Steiner Nodes
X = range(num_steiner_nodes)  # Set of Steiner nodes
Mp=[]
ObjAbsAccuracy=0.00001
distanza =0
d = 2  # Space dimension
for p in P:
    for z in P:
            if z != p:
                distanza1 = np.sqrt((coordinates_p[p]['X'] - coordinates_p[z]['X']) ** 2 + (
                            coordinates_p[p]['Y'] - coordinates_p[z]['Y']) ** 2)
                if distanza < distanza1:
                    distanza = distanza1
    Mp.append(distanza)
    distanza = 0
M=0
for p in P:
    for z in P:
            if z != p:
                distanza1 = np.sqrt((coordinates_p[p]['X'] - coordinates_p[z]['X']) ** 2 + (
                            coordinates_p[p]['Y'] - coordinates_p[z]['Y']) ** 2)
                if M < distanza1:
                    M= distanza1
#Start defining the MASTER PROBLEM
# Decision variable
ypq = {(p, q): xp.var(vartype=xp.binary, ub=1, lb=0, name=f"y_{p}_{q}") for p in P for q in X}
zpq = {(p, q): xp.var(vartype=xp.binary, ub=1, lb=0, name=f"z_{p}_{q}") for p in X for q in X if p < q}
theta = xp.var(vartype=xp.continuous, name="theta")

problem = xp.problem(name="Master problem")
problem.addVariable([ypq[key] for key in ypq])
problem.addVariable([zpq[key] for key in zpq])
problem.addVariable(theta)

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

for q in X:
    problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)

mindex=[]
problem.setControl("presolve", 0)
problem.setControl('CUTSTRATEGY', 1)
problem.setControl("FEASTOL", 1e-6)
problem.setControl("OPTIMALITYTOL", 1e-6)
problem.setControl("SCALING", 1)
g = Callbacks()
problem.addcbpreintsol(g.cb_preintsol,mindex,0)
problem.addcboptnode(g.cb_optnode,mindex, 0)
problem.solve()
LB = problem.getObjVal()
print("LB", LB)
status = problem.getProbStatus()
if status == xp.mip_optimal:
    ypq_solution=problem.getSolution(ypq)
    zpq_solution = problem.getSolution(zpq)
    print("Optimal solution found")
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Error status: {status}")
#Now we need to extract the values of the subproblem
(optimum,valori,duali)=subproblem(Mp,M,coordinates_p,ypq_solution,zpq_solution)
xp_solution = valori[0]
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
plot_the_graph(coordinates_p,xp_solution,ypq_solution,zpq_solution)