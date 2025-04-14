import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Read the data from CSV
data = pd.read_csv('istanza5.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Number of nodes
num_steiner_nodes = len(data) - 2  # Number of Steiner Nodes
X = range(num_steiner_nodes)  # Steiner nodes
d = 2
xp.init('C:/xpressmp/bin/xpauth.xpr')
Mp=[]
distanza =0
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
print(M)
d = 2  # Space dimension
#Start defining the MASTER PROBLEM
#Decision Variable
xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous,name=f"X1_{k}"),
        'Y': xp.var(vartype=xp.continuous,name=f"X2_{k}")
    }
    for k in X
} # x^p in R^d
spq = {(p, q): xp.var(vartype=xp.continuous, name=f"s_{p}_{q}") for p in X for q in X if p<q} #dichiarazione ulteriore variabili per far si che diventi MISOCP
tpq = {(p, q): xp.var(vartype=xp.continuous,name=f"t_{p}_{q}") for p in P for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
wpq = {(p, q): xp.var(vartype=xp.continuous, name=f"w_{p}_{q}")for p in P for q in X }#additive variable
vpq = {(p, q): xp.var(vartype=xp.continuous, name=f"v_{p}_{q}") for p in X for q in X if p<q}#additive variable

#Create the problem
problem = xp.problem(name="problem")
# Aggiunta delle variabili al modello
problem.addVariable([xp_var[key] for key in xp_var])
problem.addVariable([spq[key] for key in spq])
problem.addVariable([tpq[key] for key in tpq])
problem.addVariable([vpq[key] for key in vpq])
problem.addVariable([wpq[key] for key in wpq])


# Funzione obiettivo
obj = (xp.Sum(vpq[p,q] for p in X for q in X if p<q) + xp.Sum(wpq[p,q]for p in P for q in X))
problem.setObjective(obj, sense=xp.minimize)

for p in X:
   for q in X:
     if p<q:
        lhs = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y' ]) ** 2 for k in range(d))
        problem.addConstraint(lhs <= spq[p, q]**2)
for p in P:
    for q in X:
         lhs2 = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
         problem.addConstraint(lhs2 <= tpq[p, q] ** 2)
max_iters = 30
UB=100
LB=0
iteration =0
while iteration <= max_iters:
    print(f"\nIterazione {iteration + 1}")
    problem.write("model.lp")
    problem.solve()
    LB = problem.getObjVal()
    print(LB)
    status = problem.getProbStatus()
    if status == xp.lp_optimal:
        xp_solution = {key: problem.getSolution(xp_var[key]) for key in xp_var}
        spq_solution = {key: problem.getSolution(spq[key]) for key in spq}
        tpq_solution = {key: problem.getSolution(tpq[key]) for key in tpq}
        wpq_solution= {key: problem.getSolution(wpq[key]) for key in wpq}
        vpq_solution= {key: problem.getSolution(vpq[key]) for key in vpq}
        print("Soluzione ottimale Master trovata!")
        print("xpq:", xp_solution)
        print("wpq:", wpq_solution)
        print("tpq:", tpq_solution)
        print("vpq:", vpq_solution)
        print("spq:", spq_solution)
    else:
        print(f"Stato del problema: {status}")

    ypq = {(p, q): xp.var(vartype=xp.continuous, lb=0, ub=1) for p in P for q in X}
    zpq = {(p, q): xp.var(vartype=xp.continuous,lb=0, ub=1) for p in X for q in X if p<q}

    subproblem = xp.problem(name="Subproblem")
    subproblem.addVariable([ypq[key] for key in ypq])
    subproblem.addVariable([zpq[key] for key in zpq])

    # Funzione obiettivo
    obj = 0
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints=[]
    constraints1=[]
    constraints2=[]
    constraints3=[]
    constraints4=[]
    constraints5=[]
    # constraints
    for p in P:
        for q in X:
            constraint4= wpq_solution[p, q] - tpq_solution[p, q] + Mp[p] * (1 - ypq[p, q]) >=0
            subproblem.addConstraint(constraint4)
            constraints.append(constraint4)
    for p in X:
        for q in X:
            if p<q:
                constraint5= vpq_solution[p, q] - spq_solution[p, q] + M * (1 - zpq[p, q]) >=0
                subproblem.addConstraint(constraint5)
                constraints1.append(constraint5)
    for p in P:
        constraint= xp.Sum(ypq[p, q] for q in X) == 1
        subproblem.addConstraint(constraint)
        constraints2.append(constraint)

    for q in X:
        constraint1= xp.Sum(ypq[p, q] for p in P)+ xp.Sum(zpq[p, q] for p in X if p < q)  + xp.Sum(zpq[q, p] for p in X if p > q) == 3
        subproblem.addConstraint(constraint1)
        constraints3.append(constraint1)

    for q in X:
        if q > 0:
            constraint2=xp.Sum(zpq[p, q] for p in X if p < q) == 1
            subproblem.addConstraint(constraint2)
            constraints4.append(constraint2)
    for q in X:
        constraint3=xp.Sum(ypq[p, q] for p in P) <= 2
        subproblem.addConstraint(constraint3)
        constraints5.append(constraint3)

    print(constraints)
    print(constraints1)
    print(constraints2)
    print(constraints3)
    print(constraints4)
    print(constraints5)

    # Risoluzione del problema
    subproblem.controls.presolve = 0
    subproblem.controls.scaling = 0
    subproblem.write("modello.lp")
    subproblem.solve()

    UB = subproblem.getObjVal()
    print("UB", UB)
    status = subproblem.getProbStatus()
    print(subproblem.getProbStatusString())
    if status == xp.lp_optimal:
        ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

        print("Soluzione ottimale trovata!")
        print("tpq:", tpq_solution)
        print("spq:", spq_solution)
        break
    if subproblem.getProbStatus() == xp.lp_infeas:
        print(subproblem.getProbStatus())
        print("Subproblem infeasible! Generazione di un feasibility cut.")

        constraint = subproblem.getConstraint()
        num_constraints = subproblem.attributes.rows
        print(num_constraints)

        farkas_multipliers = []
        v=subproblem.hasdualray()
        print(v)
        subproblem.getdualray(farkas_multipliers)

        print(f"Farkas Multipliers: {farkas_multipliers} " )

        k = sum(1 for _ in P for _ in X)
        u = sum(1 for p in X for q in X if p<q)
        pairs = [(p, q) for p in P for q in X]
        # Costruisci il feasibility cut per tutti i vincoli
        feasibility_cut = xp.Sum(
            farkas_multipliers[i] * (-wpq[p, q] + tpq[p, q] - Mp[p]
                                     ) for i, (p, q) in enumerate((p, q) for p in P for q in X)
        ) + xp.Sum(
            farkas_multipliers[j + k] * (-vpq[p, q] + spq[p, q] - M
                                         ) for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q)
        ) + xp.Sum(
            farkas_multipliers[i + u + k] for i in P
        ) + xp.Sum(3 * farkas_multipliers[i + u + k + len(P)] for i in X) + xp.Sum(
            farkas_multipliers[i + u + k + len(P) + len(X)] for i in X if i > 1) - xp.Sum(
            2 * farkas_multipliers[i + u + k + len(P) + 2 * len(X) - 1] for i in X) <= -1e-3
        problem.addConstraint(feasibility_cut)

        print(sum(
            farkas_multipliers[i] * (-wpq_solution[p, q] + tpq_solution[p, q] - Mp[p]
                                     ) for i, (p, q) in enumerate((p, q) for p in P for q in X)
        ) + sum(
            farkas_multipliers[j + k] * (-vpq_solution[p, q] + spq_solution[p, q] - M
                                         ) for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q)
        ) + sum(
            farkas_multipliers[i + u + k] for i in P
        ) + sum(3 * farkas_multipliers[i + u + k + len(P)] for i in X) + sum(
            farkas_multipliers[i + u + k + len(P) + len(X)] for i in X if i > 1) - sum(
            2 * farkas_multipliers[i + u + k + len(P) + 2 * len(X) - 1] for i in X) <= -1e-3)
    iteration=iteration+1
    print(iteration)
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
    if val==1:  # Active connection
        x1, y1 = xp_solution[p]["X"], xp_solution[p]["Y"]
        x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

plt.title("Points in the 2D space")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()