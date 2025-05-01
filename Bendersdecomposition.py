import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time


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

ypq = {(p, q): xp.var(vartype=xp.binary,name=f"y_{p}_{q}") for p in P for q in X}
zpq = {(p, q): xp.var(vartype=xp.binary,name=f"z_{p}_{q}") for p in X for q in X }
theta =  xp.var(vartype=xp.continuous,name="theta")


problem = xp.problem(name="Master problem")
problem.addVariable([ypq[key] for key in ypq])
problem.addVariable([zpq[key] for key in zpq])
problem.addVariable(theta)

# Objective function
obj = (theta)
problem.setObjective(obj, sense=xp.minimize)
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

problem.write("Master2.lp")
max_iters = 100
UB=100
LB=0
iteration = 0
while iteration <= max_iters and np.abs(UB-LB)/abs(UB)>=0.01:
    print(f"\nIterazione {iteration + 1}")
    problem.write("Master1.lp")
    problem.solve()
    LB = problem.getObjVal()
    print("LB", LB)
    status = problem.getProbStatus()
    if status == xp.mip_optimal:
        ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

        print("Optimal solution found")
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
    else:
        print(f"Error status: {status}")
    #There we pass to the subproblem
    xp_var = {
        k: {
            'X': xp.var(vartype=xp.continuous),
            'Y': xp.var(vartype=xp.continuous)
        }
        for k in X
    }  # x^p in R^d
    spq = {(p, q): xp.var(vartype=xp.continuous,name=f"s_{p}_{q}") for p in X for q in X if p<q}
    tpq = {(p, q): xp.var(vartype=xp.continuous,name=f"t_{p}_{q}") for p in P for q in X}
    wpq = {(p, q): xp.var(vartype=xp.continuous,name=f"w_{p}_{q}") for p in P for q in X}
    vpq = {(p, q): xp.var(vartype=xp.continuous,name=f"v_{p}_{q}") for p in X for q in X if p<q}
    eps=xp.var(lb=1e-3, ub=1e-3)
    subproblem = xp.problem(name="Subproblem")
    subproblem.addVariable([xp_var[key] for key in xp_var])
    subproblem.addVariable([spq[key] for key in spq])
    subproblem.addVariable([tpq[key] for key in tpq])
    subproblem.addVariable([wpq[key] for key in wpq])
    subproblem.addVariable([vpq[key] for key in vpq])
    subproblem.addVariable(eps)

    # Objective function
    obj = (xp.Sum(vpq[p, q]  for p in X for q in X if p < q) +
           xp.Sum(wpq[p, q] for p in P for q in X))
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints=[]
    constraints1=[]
    constraints2=[]
    constraints3=[]
    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
                constraint=-lhs+ spq[p, q] ** 2>=eps**2
                subproblem.addConstraint(constraint)
                constraints.append(constraint)
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
            constraint1=-lhs2+ tpq[p, q] ** 2>=eps**2
            subproblem.addConstraint(constraint1)
            constraints1.append(constraint1)
    for p in P:
        for q in X:
            constraint2= wpq[p,q]-tpq[p, q] >=-Mp[p] * (1 - ypq_solution[p, q])
            subproblem.addConstraint(constraint2)
            constraints2.append(constraint2)
    for p in X:
        for q in X:
            if p < q:
                constraint3= vpq[p,q]- spq[p, q] >= -M * (1 - zpq_solution[p, q])
                subproblem.addConstraint(constraint3)
                constraints3.append(constraint3)

    subproblem.solve()
    UB = subproblem.getObjVal()
    print("UB", UB)
    status = subproblem.getProbStatus()
    print(subproblem.getProbStatusString())
    if status == xp.lp_optimal:
        xp_solution = {key: subproblem.getSolution(xp_var[key]) for key in xp_var}
        tpq_solution = {key: subproblem.getSolution(tpq[key]) for key in tpq}
        spq_solution = {key: subproblem.getSolution(spq[key]) for key in spq}
        wpq_solution = {key: subproblem.getSolution(wpq[key]) for key in wpq}
        vpq_solution = {key: subproblem.getSolution(wpq[key]) for key in vpq}

        print("Optimal solution found")
        print(tpq_solution)
        print(spq_solution)
        print("xp:", xp_solution)
        multipliers = subproblem.getDual(constraints)
        multipliers1 = subproblem.getDual(constraints1)
        multipliers2 = subproblem.getDual(constraints2)
        multipliers3 = subproblem.getDual(constraints3)

        for i in range(len(multipliers)):
            if abs(multipliers[i]) < 10 ** -6:
                multipliers[i] = 0
        for i in range(len(multipliers1)):
            if abs(multipliers1[i]) < 10 ** -6:
                multipliers1[i] = 0
        for i in range(len(multipliers2)):
            if abs(multipliers2[i]) < 10 ** -6:
                multipliers2[i] = 0
        for i in range(len(multipliers3)):
            if abs(multipliers3[i]) < 10 ** -6:
                multipliers3[i] = 0
        print(f" Multipliers: {multipliers}")
        print(f" Multipliers1: {multipliers1}")
        print(f" Multipliers2: {multipliers2}")
        print(f" Multipliers3: {multipliers3}")
        pairs = [(p, q) for p in P for q in X]  # Generiamo tutte le coppie (p, q)
        coord_squares = {
            p: sum(coordinates_p[p]['X' if k == 0 else 'Y'] ** 2 for k in range(d))
            for p in P
        }
        for p in P:
            for q in X:
                print('valore1')
                print((coordinates_p[p]['X']-xp_solution[q]['X'])*coordinates_p[p]['X']+(coordinates_p[p]['Y']-xp_solution[q]['Y'])*coordinates_p[p]['Y'])
                print('valore2')
                print(np.sqrt(coordinates_p[p]['X']**2+coordinates_p[p]['Y']**2+xp_solution[q]['X']**2+xp_solution[q]['Y']**2-2*coordinates_p[p]['X']*xp_solution[q]['X']-2*coordinates_p[p]['Y']*xp_solution[q]['Y']))
        optimality_cut = (
                + sum(multipliers1[j] * (np.sqrt((1e-3) ** 2+(xp_solution[q]['X']-coordinates_p[p]['X'])**2+(xp_solution[q]['Y']-coordinates_p[p]['Y'])**2)-(coordinates_p[p]['X']-xp_solution[q]['X'])*coordinates_p[p]['X']+(coordinates_p[p]['Y']-xp_solution[q]['Y'])*coordinates_p[p]['Y'])/np.sqrt((1e-3) ** 2+(xp_solution[q]['X']-coordinates_p[p]['X'])**2+(xp_solution[q]['Y']-coordinates_p[p]['Y'])**2)
                    for j, (p, q) in enumerate((p, q) for p in P for q in X))
                + xp.Sum(multipliers[j] * (np.sqrt((1e-3) ** 2+(xp_solution[q]['X']-xp_solution[p]['X'])**2+ (xp_solution[q]['Y']-xp_solution[p]['Y'])**2 )-((xp_solution[q]['Y']-xp_solution[p]['Y'])**2+(xp_solution[q]['Y']-xp_solution[p]['Y'])**2)/(np.sqrt((1e-3) ** 2+(xp_solution[q]['X']-xp_solution[p]['X'])**2+ (xp_solution[q]['Y']-xp_solution[p]['Y'])**2 )))
                         for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q))
                - xp.Sum(multipliers2[j] * (Mp[p] * (1 - ypq[p, q]))
                         for j, (p, q) in enumerate((p, q) for p in P for q in X))
                - xp.Sum(multipliers3[j] * (M * (1 - zpq[p, q]))
                         for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q))
                <= theta
        )

        problem.addConstraint(optimality_cut)

        problem.write("modello.lp")
        print(f"optimality cut: {optimality_cut}")

    if subproblem.getProbStatus() == xp.lp_infeas:
        print(subproblem.getProbStatus())
        print("Subproblem infeasible! Generation of a feasibility cut.")
        farkas_multipliers = []
        v = subproblem.hasdualray()
        print(v)
        subproblem.getdualray(farkas_multipliers)
        #Non può accadere che sia unfeasible
        """
        print(f"Farkas Multipliers: {farkas_multipliers} ")
        k = sum(1 for _ in P for _ in X)
        u = sum(1 for p in X for q in X if p < q)
        pairs = [(p, q) for p in P for q in X]  # Generiamo tutte le coppie (p, q)
        coord_squares = {
            p: sum(coordinates_p[p]['X' if k == 0 else 'Y'] ** 2 for k in range(d))
            for p in P
        }

        feasibility_cut = (
                sum(farkas_multipliers[j+u] * (coord_squares[p] + (1e-3) ** 2)
                    for j, (p, q) in enumerate((p, q) for p in P for q in X))
                + xp.Sum(farkas_multipliers[j] * (1e-3) ** 2
                         for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q))
                - xp.Sum(farkas_multipliers[j+k+u] * (Mp[p] * (1 - ypq[p, q]))
                         for j, (p, q) in enumerate((p, q) for p in P for q in X))
                - xp.Sum(farkas_multipliers[j+2*k+u] * (M * (1 - zpq[p, q]))
                         for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q))
                <= 0
        )

        problem.addConstraint(feasibility_cut)
        print(f"Aggiunto feasibility cut: {feasibility_cut}")
        """
    print(f"UB: {UB}, LB: {LB}, Difference: {np.abs(UB - LB) / UB}")
    print(Mp)
    print(tpq_solution)
    print(spq_solution)
    print(M)
    print(ypq_solution)
    print(wpq_solution)
    iteration=iteration+1
    print(iteration)
for q in X:
    grado_q = sum(ypq_solution[p, q] for p in P) + \
              sum(zpq_solution[p, q] for p in X if p < q) + \
              sum(zpq_solution[q, p] for p in X if p > q)
    print(f"Nodo {q}, Grado: {grado_q}")
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