import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time



def plot_the_graph(coordinates_p,xp_solution,ypq_solution,zpq_solution,P,X,d,coordinate_columns):
    #Building the plots
    if d == 2:
        dim1, dim2 = coordinate_columns[:2]
        plt.figure(figsize=(8, 8))

        for p in P:
            plt.scatter(coordinates_p[p][dim1], coordinates_p[p][dim2], color='blue', s=50,
                        label='Points P' if p == list(P)[0] else "")

        for p in X:
            plt.scatter(xp_solution[p][dim1], xp_solution[p][dim2], color='red', s=50,
                        label='Points X' if p == list(X)[0] else "")

        for (p, q), val in ypq_solution.items():
            if val == 1.0:
                x1, y1 = coordinates_p[p][dim1], coordinates_p[p][dim2]
                x2, y2 = xp_solution[q][dim1], xp_solution[q][dim2]
                plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1,
                         label="ypq" if (p, q) == (0, 1) else "")

        for (p, q), val in zpq_solution.items():
            if val == 1.0:
                x1, y1 = xp_solution[p][dim1], xp_solution[p][dim2]
                x2, y2 = xp_solution[q][dim1], xp_solution[q][dim2]
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2,
                         label="zpq" if (p, q) == (0, 1) else "")

        plt.title(f"Projection on {dim1}-{dim2} plane")
        plt.xlabel(dim1)
        plt.ylabel(dim2)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.show()

    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D

        dim1, dim2, dim3 = coordinate_columns[:3]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for p in P:
            ax.scatter(coordinates_p[p][dim1], coordinates_p[p][dim2], coordinates_p[p][dim3],
                       color='blue', s=50, label='Points P' if p == list(P)[0] else "")

        for p in X:
            ax.scatter(xp_solution[p][dim1], xp_solution[p][dim2], xp_solution[p][dim3],
                       color='red', s=50, label='Points X' if p == list(X)[0] else "")

        for (p, q), val in ypq_solution.items():
            if val == 1.0:
                x1, y1, z1 = coordinates_p[p][dim1], coordinates_p[p][dim2], coordinates_p[p][dim3]
                x2, y2, z2 = xp_solution[q][dim1], xp_solution[q][dim2], xp_solution[q][dim3]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'g--', linewidth=1)

        for (p, q), val in zpq_solution.items():
            if val == 1.0:
                x1, y1, z1 = xp_solution[p][dim1], xp_solution[p][dim2], xp_solution[p][dim3]
                x2, y2, z2 = xp_solution[q][dim1], xp_solution[q][dim2], xp_solution[q][dim3]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=2)

        ax.set_title(f"3D {dim1}-{dim2}-{dim3}")
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        ax.set_zlabel(dim3)
        plt.show()

start_time = time.time()
# Leggi i dati dal file CSV
data = pd.read_csv('istanza5.csv', sep='\s+')
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

xp.init('C:/xpressmp/bin/xpauth.xpr')
Mp = []
distanza = 0
d = 2  # Space dimension
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
# Start defining the MASTER PROBLEM
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

#for q in X:
 #   problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)
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
        constraint2 = wpq[p, q] - tpq[p, q] >= 0
        subproblem.addConstraint(constraint2)
        constraints2.append(constraint2)
for p in X:
    for q in X:
        if p < q:
            constraint3 = vpq[p, q] - spq[p, q] >=0
            subproblem.addConstraint(constraint3)
            constraints3.append(constraint3)
subproblem.setControl('outputlog', 0)

max_iters = 100000
UB = 100
LB = 0
iteration = 0
while iteration <= max_iters and np.abs(UB - LB) / abs(UB) >= 0.01:
    print(f"\nIterazione {iteration + 1}")
    problem.setControl("presolve", 0)
    problem.solve()
    LB = problem.getObjVal()
    print("LB", LB)
    status = problem.getProbStatus()
    if status == xp.mip_optimal:
        ypq_solution = problem.getSolution(ypq)
        zpq_solution = problem.getSolution(zpq)

        print("Optimal solution found")
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
    else:
        print(f"Error status: {status}")
    # There we pass to the subproblem
    rowind = list(constraints2)
    rhs_values = []

    j = 0
    for p in P:
        for q in X:
            #
            if j < len(rowind):
                rhs_value = -Mp[p] * (1 - ypq_solution[p, q])
                rhs_values.append(rhs_value)
            j += 1

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
    optimum = subproblem.getObjVal()
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
        multipliers_delta = {
            coord: subproblem.getDuals(delta_constraints[coord])
            for coord in coordinate_columns
        }
        epsilon=1e-7
        rhs_constant=0
        for j, (p, q) in enumerate((p, q) for p in P for q in X):
            for coord in coordinate_columns:
                rhs_constant -= multipliers_delta[coord][j] * coordinates_p[p][coord]

        part_ypq = -xp.Sum(
            multipliers2[j] * (
                Mp[p] * (1 - ypq[p, q]) if abs(multipliers2[j]) >= epsilon
                else Mp[p] if multipliers2[j] <= 0
                else 0
            )
            for j, (p, q) in enumerate((p, q) for p in P for q in X)
        )

        part_zpq = -xp.Sum(
            multipliers3[j] * (
                M * (1 - zpq[p, q]) if abs(multipliers3[j]) >= epsilon
                else M if multipliers3[j] <= 0
                else 0
            )
            for j, (p, q) in enumerate((p, q) for p in P for q in X if p < q)
        )

        optimality_cut = (rhs_constant+ part_ypq + part_zpq <= theta)
        problem.addConstraint(optimality_cut)
        problem.write('problema2.lp')
    if subproblem.getProbStatus() == xp.lp_infeas:
        print(subproblem.getProbStatus())
        print("Subproblem infeasible! Generation of a feasibility cut.")
        farkas_multipliers = []
        v = subproblem.hasdualray()
        print(v)
        subproblem.getdualray(farkas_multipliers)
        # Non può accadere che sia unfeasible
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
    iteration = iteration + 1
    print(iteration)
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
plot_the_graph(coordinates_p, xp_solution, ypq_solution, zpq_solution, P, X, d, coordinate_columns)
