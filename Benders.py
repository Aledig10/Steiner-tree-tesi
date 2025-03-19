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


max_iters = 100
UB=100
LB=0
iteration = 0
while iteration <= max_iters and np.abs(UB-LB)/abs(UB)>=0.01:
    print(f"\nIterazione {iteration + 1}")

    problem.solve()
    LB = problem.getObjVal()
    print("LB", LB)
    status = problem.getProbStatus()
    print(status)
    if status == xp.mip_optimal:
        ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

        print("Optimal solution found")
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
    else:
        print(f"Error status: {status}")
    #There we pass to the subproblem
    sum_x = sum(coordinates_p[key]['X'] for key in coordinates_p)
    sum_y = sum(coordinates_p[key]['Y'] for key in coordinates_p)
    num_elements = len(coordinates_p)
    xpo = {}  # Dizionario vuoto

    for p in X:
        xpo[p] = {'X': sum_x / num_elements, 'Y': sum_y / num_elements}  # Inizializza xp[p] come dizionario

    spq = {}  # Dizionario per memorizzare i valori di spq
    tpq = {}
    for p in X:
        for q in X:
            if p < q:
                lhs = sum(
                      (xpo[q]['X' if k == 0 else 'Y'] - xpo[p]['X' if k == 0 else 'Y']) ** 2
                        for k in range(d)
                )
                spq[p, q] =np.sqrt(lhs)


    for p in P:
        for q in X:
    # Calculate lhs2 for tpq
             lhs2 = sum((xpo[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
             tpq[p, q] = np.sqrt(lhs2)
    UB=sum(spq[p, q] - M * (1 - zpq_solution[p, q]) for p in X for q in X if p < q) + sum(tpq[p, q] - Mp[p] * (1 - ypq_solution[p, q]) for p in P for q in X)
    print("UB", UB)
    print(spq)
    print(tpq)
    print(Mp)
    print(M)

    optimality_cut = (
            sum(
                +((1 / (2 * tpq[p, q])) * (sum(coordinates_p[p]['X' if k == 0 else 'Y'] for k in range(d)) ** 2))
                for p in P for q in X
            )

            - sum(
        Mp[p] * (1 - ypq[p, q])
        for p in P for q in X
    )
            - sum(
        M * (1 - zpq[p, q])
        for p in X for q in X if p < q
    )
            <= theta
    )

    problem.addConstraint(optimality_cut)
    print(f"optimality cut: {optimality_cut}")
    problem.write("modello1.lp")
    iteration=iteration+1
    print(iteration)
    print(f"UB: {UB}, LB: {LB}, Difference: {np.abs(UB - LB) / UB}")
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
    plt.scatter(xpo[p]['X'], xpo[p]['Y'], color='red', s=50, label='Points X' if p == list(X)[0] else "")

for (p, q), val in ypq_solution.items():
    if val==1.0:
        x1, y1 = coordinates_p[p]["X"], coordinates_p[p]["Y"]
        x2, y2 = xpo[q]["X"], xpo[q]["Y"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")


for (p, q), val in zpq_solution.items():
    if val==1:
        x1, y1 = xpo[p]["X"], xpo[p]["Y"]
        x2, y2 = xpo[q]["X"], xpo[q]["Y"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

plt.title("Points in the 2D space")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.show()
