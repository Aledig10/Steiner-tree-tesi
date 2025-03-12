import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Leggi i dati dal file CSV
data = pd.read_csv('istanza2.csv', sep='\s+')
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
xp.init('C:/xpressmp/bin/xpauth.xpr')
Mp=[]
distanza =0;
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

ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}
zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X }
theta =  xp.var(vartype=xp.continuous)


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
    if q > 1:
        problem.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)


for q in X:
    problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)


max_iters = 1000
UB=100
LB=0
iteration = 0
while iteration <= max_iters and np.abs(UB-LB)/UB>=0.01:
    print(f"\nIterazione {iteration + 1}")

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
    spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}
    tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}  # additive variable
    vpq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}  # additive variable

    subproblem = xp.problem(name="Subproblem")
    subproblem.addVariable([xp_var[key] for key in xp_var])
    subproblem.addVariable([spq[key] for key in spq])
    subproblem.addVariable([tpq[key] for key in tpq])
    subproblem.addVariable([wpq[key] for key in wpq])
    subproblem.addVariable([vpq[key] for key in vpq])

    # Objective function
    obj = (xp.Sum(vpq[p, q] for p in X for q in X ) + xp.Sum(wpq[p, q] for p in P for q in X))
    subproblem.setObjective(obj, sense=xp.minimize)
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
                constraint=lhs <= spq[p, q] ** 2
                subproblem.addConstraint(constraint)
                constraints.append(constraint)
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
            constraint1=lhs2 <= tpq[p, q] ** 2
            subproblem.addConstraint(constraint1)
            constraints1.append(constraint1)
    for p in P:
        for q in X:
            constraint2=wpq[p, q] >= tpq[p, q] - Mp[p] * (1 - ypq_solution[p, q])
            subproblem.addConstraint(constraint2)
            constraints2.append(constraint2)
    for p in X:
        for q in X:
            if p < q:
                constraint3=vpq[p, q] >= spq[p, q] - M * (1 - zpq_solution[p, q])
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
        print("xp:", xp_solution)
        multipliers = subproblem.getDual(constraints)
        multipliers1 = subproblem.getDual(constraints1)
        multipliers2 = subproblem.getDual(constraints2)
        multipliers3 = subproblem.getDual(constraints3)
        print(f" Multipliers: {multipliers} {multipliers1} {multipliers2} {multipliers3} ")
        optimality_cut =  xp.Sum(
            -multipliers1[j] * (
                    xp.Sum(coordinates_p[p]['X' if k == 0 else 'Y'] for k in range(d)) ** 2
            )
            for j, (p, q) in enumerate((p, q) for p in P for q in X)
        ) + xp.Sum(
            multipliers2[j] * (-Mp[p]*(1-ypq[p,q])
            )
            for j, (p, q) in enumerate((p, q) for p in P for q in X)
        ) + xp.Sum(
            multipliers3[j] * (-M*(1-zpq[p,q])
            )
            for j, (p, q) in enumerate((p, q) for p in X for q in X if p<q )
        )<= theta
        problem.addConstraint(optimality_cut)
        problem.write("modello.lp")
        print(f"optimality cut: {optimality_cut}")

    if subproblem.getProbStatus() == xp.lp_infeas:
        print(subproblem.getProbStatus())
        print("Subproblem infeasible! Generation of a feasibility cut.")

        num_constraints = subproblem.attributes.rows

        farkas_multipliers = [0.0] * num_constraints
        subproblem.getdualray(farkas_multipliers)

        print(f"Farkas Multipliers: {farkas_multipliers} ")
        k = sum(1 for _ in P for _ in X)
        u = sum(1 for p in X for q in X if p < q)
        feasibility_cut = xp.Sum(
            farkas_multipliers[j+u] * (
                    xp.Sum(-coordinates_p[p]['X' if k == 0 else 'Y'] for k in range(d)) ** 2
            )
            for j, (p, q) in enumerate((p, q) for p in P for q in X)
        )+ xp.Sum(
            farkas_multipliers[j+u+k] * (-Mp[p]*(1-ypq[p,q])
            )
            for j, (p, q) in enumerate((p, q) for p in P for q in X)
        ) + xp.Sum(
            farkas_multipliers[j+u+2*k] * (-M*(1-zpq[p,q])
            )
            for j, (p, q) in enumerate((p, q) for p in X for q in X if p<q )
        )<= 0
        problem.addConstraint(feasibility_cut)
        print(f"Aggiunto feasibility cut: {feasibility_cut}")
        break
    iteration=iteration+1
    print(iteration)
for q in X:
    grado_q = sum(ypq_solution[p, q] for p in P) + \
              sum(zpq_solution[p, q] for p in X if p < q) + \
              sum(zpq_solution[q, p] for p in X if p > q)
    print(f"Nodo {q}, Grado: {grado_q}")
"""
xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous),
        'Y': xp.var(vartype=xp.continuous)
    }
    for k in X
} # x^p in R^d
spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X} #additive variable
tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X} #additive variable
wpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}#additive variable
vpq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}#additive variable


#problem creation
problem1 = xp.problem(name="Steiner Tree")
# Aggiunta delle variabili al modello
problem1.addVariable([xp_var[key] for key in xp_var])
problem1.addVariable([spq[key] for key in spq])
problem1.addVariable([tpq[key] for key in tpq])
problem1.addVariable([wpq[key] for key in wpq])
problem1.addVariable([vpq[key] for key in vpq])


# Objective function
obj = (xp.Sum( vpq[p,q] for p in X for q in X if p<q) + xp.Sum(wpq[p,q] for p in P for q in X))
problem1.setObjective(obj, sense=xp.minimize)
for p in P:
    for q in X:
        problem1.addConstraint(wpq[p,q] >= tpq[p,q] - Mp[p]*(1 - ypq_solution[p,q]))
for p in X:
    for q in X:
        if p<q:
            problem1.addConstraint(vpq[p, q] >= spq[p, q] - M * (1 - zpq_solution[p, q]))

for p in X:
   for q in X:
     if p<q:
        lhs = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y' ]) ** 2 for k in range(d))
        problem1.addConstraint(lhs <= spq[p, q]**2)
for p in P:
    for q in X:
         lhs2 = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
         problem1.addConstraint(lhs2 <= tpq[p, q] ** 2)



# Problem solution
problem1.solve()
status = problem1.getProbStatus()
print(problem1.getProbStatusString())
if status == xp.mip_optimal:
    xp_solution = {key: problem1.getSolution(xp_var[key]) for key in xp_var}

    print("Optimal solution found")
    print("xp:", xp_solution)
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Problem status: {status}")
"""
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
