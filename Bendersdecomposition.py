import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Leggi i dati dal file CSV
data = pd.read_csv('istanza4.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Set of given nodes
num_steiner_nodes = len(data) - 2  # Number of Steiner Nodes
X = range(num_steiner_nodes)  # Set of Steiner nodes
d = 2
xp.init('C:/xpressmp/bin/xpauth.xpr')

d = 2  # Space dimension R^d

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
iteration =0
while iteration <= max_iters and UB-LB>=0.0001:
    print(f"\nIterazione {iteration + 1}")

    problem.solve()
    LB = problem.getObjVal()
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
    spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in
           X}
    tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in
           X}

    subproblem = xp.problem(name="Subproblem")
    subproblem.addVariable([xp_var[key] for key in xp_var])
    subproblem.addVariable([spq[key] for key in spq])
    subproblem.addVariable([tpq[key] for key in tpq])

    # Objective function
    obj = (xp.Sum(spq[p, q] * zpq_solution[p, q] for p in X for q in X) + xp.Sum(
        tpq[p, q] * ypq_solution[p, q] for p in P for q in X))
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints=[]
    constraints1=[]
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

    subproblem.solve()
    UB = subproblem.getObjVal()
    status = subproblem.getProbStatus()
    print(subproblem.getProbStatusString())
    if status == xp.lp_optimal:
        xp_solution = {key: subproblem.getSolution(xp_var[key]) for key in xp_var}
        tpq_solution = {key: subproblem.getSolution(tpq[key]) for key in tpq}
        spq_solution = {key: subproblem.getSolution(spq[key]) for key in spq}

        print("Optimal solution found")
        print("xp:", xp_solution)
        print("ypq:", tpq_solution)
        print("zpq:", spq_solution)
        multipliers = subproblem.getDual(constraints)
        multipliers1 = subproblem.getDual(constraints1)
        optimality_cut = xp.Sum(
            multipliers[i] * (
                    xp.Sum(
                        (xp_solution[q]['X' if k == 0 else 'Y'] - xp_solution[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d)
                    ) - spq_solution[p, q] ** 2
            ) for i, (p, q) in enumerate((p, q) for p in X for q in X if p < q)
        ) + xp.Sum(
            multipliers1[j] * (
                    xp.Sum(
                        (xp_solution[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in
                        range(d)
                    ) - tpq_solution[p, q] ** 2
            ) for j, (p, q) in enumerate((p, q) for p in P for q in X)
        ) <= theta
        problem.addConstraint(optimality_cut)
        print(f"feasibility cut: {optimality_cut}")

    if subproblem.getProbStatus() == xp.lp_infeas:
        print(subproblem.getProbStatus())
        print("Subproblem infeasible! Generation of a feasibility cut.")

        farkas_multipliers=[]
        farkas_multipliers1=[]
        farkas_multipliers= subproblem.getdualray(constraints)
        farkas_multipliers1 = subproblem.getdualray(constraints1)
        print(f"Farkas Multipliers: {farkas_multipliers} {farkas_multipliers1}  ")

        feasibility_cut = xp.Sum(
            farkas_multipliers[i] * (
                    xp.Sum(
                        (xp_solution[q]['X' if k == 0 else 'Y'] - xp_solution[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d)
                    ) - spq[p, q] ** 2
            ) for i, (p, q) in enumerate((p, q) for p in X for q in X if p < q)
        ) + xp.Sum(
            farkas_multipliers1[j] * (
                    xp.Sum(
                        (xp_solution[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in
                        range(d)
                    ) - tpq[p, q] ** 2
            ) for j, (p, q) in enumerate((p, q) for p in P for q in X)
        ) <= 0
        problem.addConstraint(feasibility_cut)
        print(f"Aggiunto feasibility cut: {feasibility_cut}")

    iteration=iteration+1
    print(iteration)
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
