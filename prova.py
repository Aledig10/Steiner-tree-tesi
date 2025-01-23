"""import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Leggi i dati dal file CSV
data = pd.read_csv('istanza1.csv', sep=',')
data['id'] = data.index

# Prepara i dati
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
P = range(len(data))
num_steiner_nodes = len(data) - 2  # Modifica questo valore a seconda del problema
X = range(num_steiner_nodes)
indici = ['X', 'Y']

# Crea il modello
model = gp.Model("Steiner Tree")

# Variabili
y = model.addVars(P, X, vtype=GRB.BINARY, name="y")
z = model.addVars(X, X, vtype=GRB.BINARY, name="z")
x_coords = model.addVars(X, indici, vtype=GRB.CONTINUOUS, name="x")
w=model.addVars(X, indici, vtype=GRB.CONTINUOUS, name="w")
z1=model.addVars(X, indici, vtype=GRB.CONTINUOUS, name="z")

# Funzione obiettivo
model.setObjective(
    gp.quicksum(y[p, q] * gp.quicksum(z1 for coord in indici) for p in P for q in X) +
    gp.quicksum(z[p, q] * gp.quicksum(z1  for coord in indici) for p in X for q in X if p < q),
    GRB.MINIMIZE
)
# Vincolo: ogni nodo P deve essere assegnato a esattamente un nodo Steiner
model.addConstrs(
    gp.quicksum(y[p, q] for q in X) == 1 for p in P
)

# Vincolo: grado massimo dei nodi Steiner
model.addConstrs(
    gp.quicksum(y[p, q] for p in P) + \
    gp.quicksum(z[p, q] for p in X if p < q) + \
    gp.quicksum(z[q, p] for p in X if p > q) <= 3 for q in X
)

# Vincolo: flusso tra i nodi Steiner
model.addConstrs(
    gp.quicksum(z[p, q] for p in X if p < q) == 1 for q in X if q > 1
)

# Vincolo: ogni nodo Steiner non può essere assegnato a più di due nodi P
model.addConstrs(
    gp.quicksum(y[p, q] for p in P) <= 2 for q in X
)

model.addConstrs(
    z1**2>=w**2
)
model.addConstrs(
    w=x_coords[p,q]
)
# Risolvi il modello
model.optimize()

# Stampa i risultati
if model.status == GRB.OPTIMAL:
    print("Soluzione ottimale trovata:")
    for p in P:
        for q in X:
            if y[p, q].x > 0.5:
                print(f"y[{p},{q}] = {y[p, q].x}")
    for p in X:
        for q in X:
            if z[p, q].x > 0.5:
                print(f"z[{p},{q}] = {z[p, q].x}")
    for q in X:
        print(f"x[{q},X] = {x_coords[q, 'X'].x}, x[{q},Y] = {x_coords[q, 'Y'].x}")
else:
    print("Nessuna soluzione ottimale trovata.")
"""
import xpress as xp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Leggi i dati dal file CSV
data = pd.read_csv('istanza2.csv', sep=',')
data['id'] = data.index

# Prepara i dati
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
P = range(len(data))
num_steiner_nodes = len(data) - 2  # Modifica questo valore a seconda del problema
X = range(num_steiner_nodes)
indici = ['X', 'Y']
xp.init('C:/xpressmp/bin/xpauth.xpr')
# Crea il modello

d = 2  # Dimensione dello spazio R^d

# Variabili di decisione
xp_var = {(p, k): xp.var(vartype=xp.continuous) for p in P for k in range(d)}  # x^p in R^d
ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}                # y_pq binarie
zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X if p < q}       # z_pq binarie
zpq_aux = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
zpq_aux_squared = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}

# Creazione del problema
problem = xp.problem(name="Steiner Tree")
# Aggiunta delle variabili al modello
problem.addVariable([xp_var[key] for key in xp_var])
problem.addVariable([ypq[key] for key in ypq])
problem.addVariable([zpq[key] for key in zpq])
problem.addVariable([zpq_aux[key] for key in zpq_aux])
problem.addVariable([zpq_aux_squared[key] for key in zpq_aux_squared])

# Aggiungi il vincolo SOC per ogni coppia (p, q)
for p in P:
    for q in X:
        # Vincolo per zpq_aux_squared = zpq_aux^2
        problem.addConstraint(zpq_aux_squared[p, q] >= zpq_aux[p, q] * zpq_aux[p, q])

        # Vincolo SOC riformulato
        lhs = xp.Sum((xp_var[q, k] - xp_var[p, k]) ** 2 for k in range(d))
        problem.addConstraint(lhs <= zpq_aux_squared[p, q])

# Funzione obiettivo
obj = (
    xp.Sum(
        ypq[p, q] * zpq_aux[p, q] # 2-norm squared
        for p in P for q in X
    )
    + xp.Sum(
        zpq[p, q] * zpq_aux[p, q] # 2-norm squared
        for p in X for q in X if p < q
    )
)
problem.setObjective(obj, sense=xp.minimize)

# Vincolo (2): Somma di ypq per ogni p
for p in P:
    problem.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

# Vincolo (3): Somma combinata per ogni q
for q in X:
    problem.addConstraint(
        xp.Sum(ypq[p, q] for p in P)
        + xp.Sum(zpq[p, q] for p in X if p < q)
        + xp.Sum(zpq[q, p] for p in X if p > q) == 3
    )

# Vincolo (4): Somma di zpq per ogni q > 1
for q in X:
    if q > 1:
        problem.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

# Vincolo (5): Somma di ypq per ogni q <= 2
for q in X:
    problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)




# Vincolo (6): Definizione delle variabili x^p come continue in R^d
# Già gestito tramite la creazione di `xp_var` come variabili continue.

# Vincolo (7): ypq sono variabili binarie
# Vincolo (8): zpq sono variabili binarie
# Già definito tramite `vartype=xp.binary` nella dichiarazione delle variabili.

# Risoluzione del problema
problem.solve()

# Recupero della soluzione
status = problem.getProbStatus()
if status == xp.mip_optimal:
    xp_solution = {key: problem.getSolution(xp_var[key]) for key in xp_var}
    ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
    zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

    print("Soluzione ottimale trovata!")
    print("xp:", xp_solution)
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Stato del problema: {status}")
#

"""
# Variabili
y = {(p, q): xp.var(vartype=xp.binary, name=f"y_{p}_{q}") for p in P for q in X}
z = {(p, q): xp.var(vartype=xp.binary, name=f"z_{p}_{q}") for p in X for q in X if p < q}
x_coords = {(q, coord): xp.var(vartype=xp.continuous, name=f"x_{q}_{coord}") for q in X for coord in indici}

# Aggiungi variabili al modello
model.addVariable(y)
model.addVariable(z)
model.addVariable(x_coords)

# Funzione obiettivo
objective = (
    xp.Sum(y[p, q] * xp.sqrt(xp.Sum((x_coords[q, coord] - coordinates_p[p][coord])**2 for coord in indici))
           for p in P for q in X) +
    xp.Sum(z[p, q] * xp.sqrt(xp.Sum((x_coords[q, coord] - x_coords[p, coord])**2 for coord in indici))
           for p in X for q in X if p < q)
)
model.setObjective(objective, sense=xp.minimize)

# Vincolo: ogni nodo P deve essere assegnato a esattamente un nodo Steiner
for p in P:
    model.addConstraint(xp.Sum(y[p, q] for q in X) == 1)

# Vincolo: grado massimo dei nodi Steiner
for q in X:
    model.addConstraint(
        xp.Sum(y[p, q] for p in P) +
        xp.Sum(z[p, q] for p in X if p < q) +
        xp.Sum(z[q, p] for p in X if p > q) <= 3
    )

# Vincolo: flusso tra i nodi Steiner
for q in X:
    if q > 1:
        model.addConstraint(
            xp.Sum(z[p, q] for p in X if p < q) == 1
        )

# Vincolo: ogni nodo Steiner non può essere assegnato a più di due nodi P
for q in X:
    model.addConstraint(
        xp.Sum(y[p, q] for p in P) <= 2
    )

# Risolvi il modello
model.solve()

# Stampa i risultati
if model.getProbStatus() == xp.optimal:
    print("Soluzione ottimale trovata:")
    for p in P:
        for q in X:
            if y[p, q].value > 0.5:
                print(f"y[{p},{q}] = {y[p, q].value}")
    for p in X:
        for q in X:
            if (p, q) in z and z[p, q].value > 0.5:
                print(f"z[{p},{q}] = {z[p, q].value}")
    for q in X:
        print(f"x[{q},X] = {x_coords[q, 'X'].value}, x[{q},Y] = {x_coords[q, 'Y'].value}")
else:
    print("Nessuna soluzione ottimale trovata.")

import xpress as xp

# Creazione del modello
model = xp.problem()

# Definizione delle variabili (lista di variabili)
n = 5  # Numero di variabili
x = [xp.var(name=f"x{i}") for i in range(n)]

# Aggiunta delle variabili al modello
model.addVariable(x)

# Vincolo sulla norma L2: somma dei quadrati <= 10
model.addConstraint(xp.Sum(x[i]**2 for i in range(n)) <= 10)

# Definizione dell'obiettivo
model.setObjective(xp.Sum(x[i] for i in range(n)) + 0.5 * xp.Sum(x[i]**2 for i in range(n)), sense=xp.minimize)

# Risoluzione del problema
model.solve()

# Output dei risultati
solution = model.getSolution(x)
print("Soluzione:", solution)

"""