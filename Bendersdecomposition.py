import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time


def cb_preintsol(problem,subproblem,soltype,cutoff):
    rowind = list(constraints2)
    rhs_values = []
    ypq_vals = problem.getCallbackSolution(list(ypq.values()))
    ypq_solution = dict(zip(ypq.keys(), ypq_vals))
    zpq_vals = problem.getCallbackSolution(list(zpq.values()))
    zpq_solution = dict(zip(zpq.keys(), zpq_vals))
    for p in P:
        for q in X:
            rhs_value = -Mp[p] * (1 - ypq_solution[p, q])
            rhs_values.append(rhs_value)

    # Aggiorna solo i vincoli specifici
    subproblem.chgrhs(rowind[:len(rhs_values)], rhs_values)
    rowind2 = list(constraints3)
    rhs_values = []

    for p in X:
        for q in X:
            if p < q:
                rhs_value = -M * (1 - zpq_solution[p, q])
                rhs_values.append(rhs_value)
    subproblem.chgrhs(rowind2[:len(rhs_values)], rhs_values)
    subproblem.solve()
    UB = subproblem.getObjVal()
    print("UB", UB)
    status = subproblem.getProbStatus()
    if status == xp.lp_optimal:
        xp_solution = subproblem.getSolution(xp_var)
        wpq_solution = subproblem.getSolution(wpq)
        vpq_solution = subproblem.getSolution(vpq)
        value= sum(wpq_solution[p,q] for p in P for q in X)+ sum(vpq_solution[p,q] for p in X for q in X if p<q)
    return (True,value)

def cb_optnode(problem,subproblem):
    rowind = list(constraints2)
    rhs_values = []
    ypq_vals = problem.getCallbackSolution(list(ypq.values()))
    ypq_solution = dict(zip(ypq.keys(), ypq_vals))
    zpq_vals = problem.getCallbackSolution(list(zpq.values()))
    zpq_solution = dict(zip(zpq.keys(), zpq_vals))
    for p in P:
        for q in X:
            rhs_value = -Mp[p] * (1 - ypq_solution[p, q])
            rhs_values.append(rhs_value)

    subproblem.chgrhs(rowind[:len(rhs_values)], rhs_values)
    rowind2 = list(constraints3)
    rhs_values = []

    for p in X:
        for q in X:
            if p < q:
                rhs_value = -M * (1 - zpq_solution[p, q])
                rhs_values.append(rhs_value)

    subproblem.chgrhs(rowind2[:len(rhs_values)], rhs_values)
    subproblem.solve()
    UB = subproblem.getObjVal()
    print("UB", UB)
    status = subproblem.getProbStatus()
    if status == xp.lp_optimal:
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
        multipliers4 = subproblem.getDuals(constraints4)
        multipliers5 = subproblem.getDuals(constraints5)
        colind = []
        cutcoef = []
        rhs_val = 0.0

        epsilon = 1e-7
        print(multipliers4)
        print(multipliers5)
        print(multipliers3)
        print(multipliers2)

        for j, (p, q) in enumerate((p, q) for p in P for q in X):
            rhs_val += -multipliers4[j] * coordinates_p[p]['X']
            rhs_val += -multipliers5[j] * coordinates_p[p]['Y']


        for j, (p, q) in enumerate((p, q) for p in P for q in X):
            if abs(multipliers2[j]) >= epsilon:
                coef = multipliers2[j] * Mp[p]
                colind.append(problem.getIndex(ypq[p, q]))
                cutcoef.append(coef)
                rhs_val += coef
            elif multipliers2[j] <= 0:
                rhs_val += -multipliers2[j] * Mp[p]

        for j, (p, q) in enumerate((p, q) for p in P for q in X if p < q):
            if abs(multipliers3[j]) >= epsilon:
                coef = multipliers3[j] * M
                colind.append(problem.getIndex(zpq[p, q]))
                cutcoef.append(coef)
                rhs_val += -coef
            elif multipliers3[j] <= 0:
                rhs_val += multipliers3[j] * M

        colind.append((problem.getIndex(theta)))
        cutcoef.append(-1.0)
        cuttype = [1]
        rowtype = ['L']
        rhs = [rhs_val]
        start = [0,len(colind)]
        thetasol=problem.getCallbackSolution(theta)
        print("thetasol", thetasol)
        problem.write('problema.lp')
        problem.addcuts(cuttype, rowtype, rhs, start, colind, cutcoef)

    return 0
start_time = time.time()
# Leggi i dati dal file CSV
data = pd.read_csv('istanza5.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Set of given nodes
print(P)
num_steiner_nodes = len(data) - 2  # Number of Steiner Nodes
X = range(num_steiner_nodes)  # Set of Steiner nodes
print(X)
xp.init('C:/xpressmp/bin/xpauth.xpr')
Mp=[]
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
print(Mp)
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
problem = xp.problem(name="Master problem")
ypq= {(p, q): problem.addVariable(name=f"y_{p}_{q}",
                           vartype=xp.binary) for p in P for q in X}
zpq= {(p, q): problem.addVariable(name=f"z_{p}_{q}",
                           vartype=xp.binary) for p in X for q in X if p<q}

theta =  problem.addVariable(vartype=xp.continuous,name="theta")

# Objective function
problem.setObjective(theta, sense=xp.minimize)
#constraints
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

xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous, name=f"xp_{k}"),
        'Y': xp.var(vartype=xp.continuous, name=f"yp_{k}")
    }
    for k in X
}  # x^p in R^d
spq = {(p, q): xp.var(vartype=xp.continuous, name=f"s_{p}_{q}") for p in X for q in X if p < q}
tpq = {(p, q): xp.var(vartype=xp.continuous, name=f"t_{p}_{q}") for p in P for q in X}
wpq = {(p, q): xp.var(vartype=xp.continuous, name=f"w_{p}_{q}") for p in P for q in X}
vpq = {(p, q): xp.var(vartype=xp.continuous, name=f"v_{p}_{q}") for p in X for q in X if p < q}
deltapq = {(p, q, coord): xp.var(vartype=xp.continuous, lb=-1e4, name=f"delta_{p}_{q}_{coord}")
           for p in P for q in X for coord in ['X', 'Y']}

gammapq = {(p, q, coord): xp.var(vartype=xp.continuous, lb=-1e4, name=f"gamma_{p}_{q}_{coord}")
           for p in X for q in X if p < q for coord in ['X', 'Y']}
subproblem = xp.problem(name="Subproblem")
subproblem.addVariable([xp_var[key] for key in xp_var])
subproblem.addVariable([spq[key] for key in spq])
subproblem.addVariable([tpq[key] for key in tpq])
subproblem.addVariable([wpq[key] for key in wpq])
subproblem.addVariable([vpq[key] for key in vpq])
subproblem.addVariable([deltapq[key] for key in deltapq])
subproblem.addVariable([gammapq[key] for key in gammapq])
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
        constraint2 = wpq[p, q] - tpq[p, q] >= 0
        subproblem.addConstraint(constraint2)
        constraints2.append(constraint2)
for p in X:
    for q in X:
        if p < q:
            constraint3 = vpq[p, q] - spq[p, q] >= 0
            subproblem.addConstraint(constraint3)
            constraints3.append(constraint3)
problem.setControl("presolve", 0)  # Disattiva completamente il presolve
problem.addcbpreintsol(cb_preintsol,subproblem,1)
problem.addcboptnode(cb_optnode, subproblem, 0)
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
rowind = list(constraints2)
rhs_values = []

for p in P:
      for q in X:
          rhs_value = -Mp[p] * (1 - ypq_solution[p, q])
          rhs_values.append(rhs_value)

subproblem.chgrhs(rowind[:len(rhs_values)], rhs_values)
rowind2 = list(constraints3)
rhs_values = []

for p in X:
    for q in X:
        if p < q:
            rhs_value = -M * (1 - zpq_solution[p, q])
            rhs_values.append(rhs_value)

subproblem.chgrhs(rowind2[:len(rhs_values)], rhs_values)
subproblem.solve()
status = subproblem.getProbStatus()
if status == xp.lp_optimal:
    xp_solution = subproblem.getSolution(xp_var)
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
plt.figure(figsize=(8, 8))
for p in P:
    plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50, label='Points P' if p == list(P)[0] else "")

for p in X:
    plt.scatter(xp_solution[p]['X'], xp_solution[p]['Y'], color='red', s=50, label='Points X' if p == list(X)[0] else "")

for (p, q), val in ypq_solution.items():
    if val==1.0:
        x1, y1 = coordinates_p[p]["X"], coordinates_p[p]["Y"]
        x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")


for (p, q), val in zpq_solution.items():
    if val==1:
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
