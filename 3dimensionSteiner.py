import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Read CSV file
data = pd.read_csv('istanza3d_1.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y','Z']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Set of given nodes
num_steiner_nodes = len(data) - 2  # Number of Steiner node3s
X = range(num_steiner_nodes)  # Set of Steiner nodes
xp.init('C:/xpressmp/bin/xpauth.xpr')
# Creation of the model
Mp=[]
distanza =0
d = 3 # Space dimension
for p in P:
    for z in P:
            if z != p:
                distanza1 = np.sqrt((coordinates_p[p]['X'] - coordinates_p[z]['X']) ** 2 + (
                            coordinates_p[p]['Y'] - coordinates_p[z]['Y']) ** 2 + (coordinates_p[p]['Z'] - coordinates_p[z]['Z']) ** 2)
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
                            coordinates_p[p]['Y'] - coordinates_p[z]['Y']) ** 2
                                    + (coordinates_p[p]['Z'] - coordinates_p[z]['Z'])**2)
                if M < distanza1:
                    M= distanza1
print(M)
# Decision variable
xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous),
        'Y': xp.var(vartype=xp.continuous),
        'Z': xp.var(vartype=xp.continuous)
    }
    for k in X
} # x^p in R^d
ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}     # y_pq binary
zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X }    # z_pq binary
spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X} #additive variable
tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X} #additive variable
wpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}#additive variable
vpq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}#additive variable


#problem creation
problem = xp.problem(name="Steiner Tree")
# Aggiunta delle variabili al modello
problem.addVariable([xp_var[key] for key in xp_var])
problem.addVariable([ypq[key] for key in ypq])
problem.addVariable([zpq[key] for key in zpq])
problem.addVariable([spq[key] for key in spq])
problem.addVariable([tpq[key] for key in tpq])
problem.addVariable([wpq[key] for key in wpq])
problem.addVariable([vpq[key] for key in vpq])


# Objective function
obj = (xp.Sum( vpq[p,q] for p in X for q in X if p<q) + xp.Sum(wpq[p,q] for p in P for q in X))
problem.setObjective(obj, sense=xp.minimize)

# Constraints
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


for p in X:
   for q in X:
     if p<q:
        lhs = xp.Sum((xp_var[q]['X' if k == 0 else 'Y' if k == 1 else 'Z']
                     - xp_var[p]['X' if k == 0 else 'Y' if k == 1 else 'Z' ]) ** 2 for k in range(d))
        problem.addConstraint(lhs <= spq[p, q]**2)
for p in P:
    for q in X:
         lhs2 = xp.Sum((xp_var[q]['X' if k == 0 else 'Y' if k == 1 else 'Z'] - coordinates_p[p]['X' if k == 0 else 'Y' if k == 1 else 'Z']) ** 2 for k in range(d))
         problem.addConstraint(lhs2 <= tpq[p, q] ** 2)


for p in P:
    for q in X:
        problem.addConstraint(wpq[p,q] >= tpq[p,q] - Mp[p]*(1 - ypq[p,q]))
for p in X:
    for q in X:
        if p<q:
            problem.addConstraint(vpq[p, q] >= spq[p, q] - M * (1 - zpq[p, q]))


# Problem solution
problem.solve()
status = problem.getProbStatus()
print(problem.getProbStatusString())
if status == xp.mip_optimal:
    xp_solution = {key: problem.getSolution(xp_var[key]) for key in xp_var}
    ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
    zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

    print("Optimal solution found")
    print("xp:", xp_solution)
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Problem status: {status}")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot per i punti P
for p in P:
    ax.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], coordinates_p[p]['Z'],
               color='blue', s=50, label='Points P' if p == list(P)[0] else "")

# Scatter plot per i punti X
for p in X:
    ax.scatter(xp_solution[p]['X'], xp_solution[p]['Y'], xp_solution[p]['Z'],
               color='red', s=50, label='Points X' if p == list(X)[0] else "")

# Linee per ypq_solution
for (p, q), val in ypq_solution.items():
    if val == 1.0:
        x1, y1, z1 = coordinates_p[p]["X"], coordinates_p[p]["Y"], coordinates_p[p]["Z"]
        x2, y2, z2 = xp_solution[q]["X"], xp_solution[q]["Y"], xp_solution[q]["Z"]
        if x1 is not None and x2 is not None:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")

# Linee per zpq_solution
for (p, q), val in zpq_solution.items():
    if val == 1:
        x1, y1, z1 = xp_solution[p]["X"], xp_solution[p]["Y"], xp_solution[p]["Z"]
        x2, y2, z2 = xp_solution[q]["X"], xp_solution[q]["Y"], xp_solution[q]["Z"]
        if x1 is not None and x2 is not None:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

ax.set_title("Points in the 3D space")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)

plt.show()