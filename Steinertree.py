
import xpress as xp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Leggi i dati dal file CSV
data = pd.read_csv('istanza2.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Insieme di nodi dati
num_steiner_nodes = len(data) - 2  # Numero di nodi di Steiner (adatta al problema)
X = range(num_steiner_nodes)  # Nodi di Steiner
d = 2
M=10000
xp.init('C:/xpressmp/bin/xpauth.xpr')
# Crea il modello

d = 2  # Dimensione dello spazio R^d

# Variabili di decisione
xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous),
        'Y': xp.var(vartype=xp.continuous)
    }
    for k in X
} # x^p in R^d
ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}                # y_pq binarie
zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X }       # z_pq binarie
spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
wpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}
vpq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}


# Creazione del problema
problem = xp.problem(name="Steiner Tree")
# Aggiunta delle variabili al modello
problem.addVariable([xp_var[key] for key in xp_var])
problem.addVariable([ypq[key] for key in ypq])
problem.addVariable([zpq[key] for key in zpq])
problem.addVariable([spq[key] for key in spq])
#problem.addVariable([aux1_squared[key] for key in aux1_squared])
problem.addVariable([tpq[key] for key in tpq])
#problem.addVariable([aux2_squared[key] for key in aux2_squared])
problem.addVariable([wpq[key] for key in wpq])
problem.addVariable([vpq[key] for key in vpq])


# Funzione obiettivo
obj = (xp.Sum( vpq[p,q] for p in X for q in X) + xp.Sum(wpq[p,q] for p in P for q in X))
problem.setObjective(obj, sense=xp.minimize)

# Vincolo : Somma di ypq per ogni p
for p in P:
    problem.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

# Vincolo: Somma combinata per ogni q
for q in X:
    problem.addConstraint(
        xp.Sum(ypq[p, q] for p in P)
        + xp.Sum(zpq[p, q] for p in X if p < q)
        + xp.Sum(zpq[q, p] for p in X if p > q) == 3
    )

# Vincolo: Somma di zpq per ogni q > 1
for q in X:
    if q > 1:
        problem.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

# Vincolo: Somma di ypq per ogni q <= 2
for q in X:
    problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)


# Aggiungi il vincolo SOC per ogni coppia (p, q)
for p in X:
   for q in X:
     if p<q:
        # Vincolo SOC riformulato
        lhs = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y' ]) ** 2 for k in range(d))
        problem.addConstraint(lhs <= spq[p, q]**2)
for p in P:
    for q in X:
         lhs2 = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
         problem.addConstraint(lhs2 <= tpq[p, q] ** 2)

# Aggiungo il vincolo per farlo diventare un problema mixed-integer second-order cone optimization (MISOCO)
for p in P:
    for q in X:
        problem.addConstraint(wpq[p,q] >= tpq[p,q] - M*(1 - ypq[p,q]))
for p in X:
    for q in X:
            problem.addConstraint(vpq[p, q] >= spq[p, q] - M * (1 - zpq[p, q]))


# Risoluzione del problema
problem.solve()
status = problem.getProbStatus()
if status == xp.mip_optimal:
    xp_solution = {key: problem.getSolution(xp_var[key]) for key in xp_var}
    ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
    zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

    print("Soluzione trovata")
    print("xp:", xp_solution)
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Stato del problema: {status}")

#
plt.figure(figsize=(8, 8))
for p in P:
    plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50, label='Punti P' if p == list(P)[0] else "")


for p in X:
    plt.scatter(xp_solution[p]['X'], xp_solution[p]['Y'], color='red', s=50, label='Punti X' if p == list(X)[0] else "")

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

plt.title("Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
