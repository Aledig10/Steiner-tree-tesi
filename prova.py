"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt

# Leggi i dati dal file CSV
data = pd.read_csv('istanza1.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Insieme di nodi dati
num_steiner_nodes = len(data) - 2  # Numero di nodi di Steiner (adatta al problema)
X = range(num_steiner_nodes)  # Nodi di Steiner
d = 2  # Dimensione dello spazio
plt.figure(figsize=(8, 8))
for p in P:
    plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50, label='Punti P' )
plt.title("Punti nello spazio 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

model = gp.Model("Steiner problem")

# Variabili di decisione
x = model.addVars(P, range(d), lb=-GRB.INFINITY, name="x")  # Variabili di posizione

y = model.addVars(P, X, vtype=GRB.BINARY, name="y")  # Variabili binarie y_pq
z = model.addVars(X, X, vtype=GRB.BINARY, name="z")  # Variabili binarie z_pq

# Variabili per i vincoli SOC
t = model.addVars(P, X, lb=0, name="t")  # Variabili ausiliarie per norme y_pq
s = model.addVars(X, X, lb=0, name="s")  # Variabili ausiliarie per norme z_pq

# Funzione obiettivo
model.setObjective(
    gp.quicksum(y[p, q] * t[p, q] for p in P for q in X) +
    gp.quicksum(z[p, q] * s[p, q] for p in X for q in X if p < q),
    GRB.MINIMIZE
)

# Vincoli
for p in P:
    model.addConstr(gp.quicksum(y[p, q] for q in X) == 1, name=f"assign_{p}")

for q in X:
    model.addConstr(
        gp.quicksum(y[p, q] for p in P) + gp.quicksum(z[p, q] for p in X if p < q) + gp.quicksum(z[q, p] for p in X if p > q) == 3,
        name=f"degree_{q}"
    )

for q in X:
    if q > 1:
        model.addConstr(gp.quicksum(z[p, q] for p in X if p < q) == 1, name=f"chain_{q}")

for q in X:
    model.addConstr(gp.quicksum(y[p, q] for p in P) <= 2, name=f"capacity_{q}")

# Vincoli SOC per ||x^q - ζ^p||_2
for p in P:
    for q in X:
        model.addConstr(t[p, q] * t[p, q] >= gp.quicksum((x[q, i] - coordinates_p[p, i]) * (x[q, i] - x[p, i]) for i in range(d)), name=f"soc_y_{p}_{q}")

# Vincoli SOC per ||x^q - x^p||_2
for p in X:
    for q in X:
        if p < q:
            model.addConstr(s[p, q] * s[p, q] >= gp.quicksum((x[q, i] - x[p, i]) * (x[q, i] - x[p, i]) for i in range(d)), name=f"soc_z_{p}_{q}")

# Ottimizzazione
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found!\n")

    # Stampa delle variabili x
    print("Variable x (positions):")
    for p in P:
        print(f"x[{p}] =", [x[p, i].X for i in range(d)])

    # Stampa delle variabili y
    print("\nVariable y (assignments):")
    for p in P:
        for q in X:
            if y[p, q].X > 0.5:  # Consideriamo solo i valori binari attivi
                print(f"y[{p},{q}] =", y[p, q].X)
            print(f"t[{p},{q}] =", t[p, q].X)

    # Stampa delle variabili z
    print("\nVariable z (connections):")
    for p in X:
        for q in X:
            if p < q and z[p, q].X > 0.5:  # Consideriamo solo i valori binari attivi
                print(f"z[{p},{q}] =", z[p, q].X)
                print(f"s[{p},{q}] =", s[p, q].X)
else:
    print("No optimal solution found.")
"""
"""
# Creazione del modello
model = gp.Model()

# Variabili di decisione
xp_var = {
    k: {
        'X': model.addVar(vtype=GRB.CONTINUOUS, name=f"xp[{k}].X"),
        'Y': model.addVar(vtype=GRB.CONTINUOUS, name=f"xp[{k}].Y")
    }
    for k in X
}

ypq = model.addVars(P, X, vtype=GRB.BINARY, name="ypq")  # y_pq binarie
zpq = model.addVars([(p, q) for p in X for q in X if p < q], vtype=GRB.BINARY, name="zpq")
aux2 = model.addVars(X, X,lb=0, vtype=GRB.CONTINUOUS, name="aux2")  # Variabili ausiliarie per norme 2
aux1 = model.addVars(P, X,lb=0, vtype=GRB.CONTINUOUS, name="aux1")  # Variabili ausiliarie per norme 2



# Funzione obiettivo
model.setObjective(
    sum(ypq[p, q] * aux1[p, q] for p in P for q in X) +
    sum(zpq[p, q] * aux2[p, q] for p in X for q in X if p < q),
    GRB.MINIMIZE
)
# Vincolo (2): Somma di ypq per ogni p
for p in P:
    model.addConstr(gp.quicksum(ypq[p, q] for q in X) == 1, name=f"c2[{p}]")

# Vincolo (3): Somma combinata per ogni q
for q in X:
    model.addConstr(
        (gp.quicksum(ypq[p, q] for p in P) +
        gp.quicksum(zpq[p, q] for p in X if p < q) +
        gp.quicksum(zpq[q, p] for p in X if p > q)) == 3,
        name=f"c3[{q}]"
    )

# Vincolo (4): Somma di zpq per ogni q > 1
for q in X:
    if q > 1:
        model.addConstr(
            gp.quicksum(zpq[p, q] for p in X if p < q) == 1,
            name=f"c4[{q}]"
        )


# Vincolo (5): Somma di ypq per ogni q <= 2
for q in X:
    model.addConstr(
        gp.quicksum(ypq[p, q] for p in P) <= 2,
        name=f"c5[{q}]"
    )
# Vincolo SOC per ogni coppia (p, q)
for p in X:
    for q in X:
        if p<q:
            model.addQConstr(
                aux2[p, q] ** 2 >=
                ((xp_var[q]['X'] - xp_var[p]['X']) ** 2 +
                 (xp_var[q]['Y'] - xp_var[p]['Y']) ** 2),
                name=f"soc_aux2[{p},{q}]"
            )

            # Norme euclidee per z_pq
for p in P:
    for q in X:
        # Norme euclidee per y_pq
        model.addQConstr(
            aux1[p, q] * aux1[p, q] >=
            (xp_var[q]['X'] - coordinates_p[p]['X']) ** 2 +
            (xp_var[q]['Y'] - coordinates_p[p]['Y']) ** 2,
            name=f"soc_aux1[{p},{q}]"
        )
# Risoluzione del modello
model.write("debug_model.lp")
print("Numero di nodi P:", len(P))
print("Numero di nodi Steiner X:", len(X))
print("Numero di variabili xp_var:", len(xp_var))
print("Numero di variabili ypq:", len(ypq))
print("Numero di variabili zpq:", len(zpq))
print("Numero di vincoli nel modello:", model.NumConstrs)
print("Numero di vincoli quadratici:", len([c for c in model.getQConstrs()]))


model.optimize()

# Controllo dello stato
if model.status == GRB.OPTIMAL:
    for c in model.getConstrs():
        print(f"Vincolo {c.ConstrName}: valore slack = {c.Slack}")
    print("Soluzione ottimale trovata!")
    xp_solution = {
        key: (
            model.getVarByName(f"xp[{key}].X").X,
            model.getVarByName(f"xp[{key}].Y").X
        )
        for key in X
    }
    ypq_solution = {key: ypq[key].X for key in ypq}
    zpq_solution = {key: zpq[key].X for key in zpq}

    print("xp:", xp_solution)
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Stato del modello: {model.status}")
print(coordinates_p[0]['X'])
print(coordinates_p[0]['Y'])
plt.figure(figsize=(8, 8))
for p in P:
    plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50, label='Punti P' )

# Plotta i punti rossi (insieme X)
for p in X:
    plt.scatter(xp_solution[p][0], xp_solution[p][1], color='red', s=50, label='Punti X' if p == list(X)[0] else "")

for (p, q), value in ypq_solution.items():
    if value == 1.0:
        x1 = xp_solution[p][0]
        y1 = xp_solution[p][1]
        x2=coordinates_p[q]['X']
        y2=coordinates_p[q]['Y']
        plt.plot([x1, x2], [y1, y2], 'b-')

for (p, q), value in zpq_solution.items():
    if value == 1.0:
        x1 = xp_solution[p][0]
        y1 = xp_solution[p][1]
        x2=xp_solution[q][0]
        y2=xp_solution[q][1]
        plt.plot([x1, x2], [y1, y2], 'b-')

plt.title("Punti nello spazio 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()


"""
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


# Vincolo (6): Definizione delle variabili x^p come continue in R^d
# Già gestito tramite la creazione di `xp_var` come variabili continue.

# Risoluzione del problema
problem.solve()
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
plt.figure(figsize=(8, 8))
for p in P:
    plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50, label='Punti P' if p == list(P)[0] else "")

# Plotta i punti rossi (insieme X)
for p in X:
    plt.scatter(xp_solution[p]['X'], xp_solution[p]['Y'], color='red', s=50, label='Punti X' if p == list(X)[0] else "")

for (p, q), val in ypq_solution.items():
    if val==1.0:  # Solo connessioni attive
        x1, y1 = coordinates_p[p]["X"], coordinates_p[p]["Y"]
        x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")

# Disegnare archi basati su zpq (connessioni tra X)
for (p, q), val in zpq_solution.items():
    if val==1:  # Solo connessioni attive
        x1, y1 = xp_solution[p]["X"], xp_solution[p]["Y"]
        x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

plt.title("Punti nello spazio 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
