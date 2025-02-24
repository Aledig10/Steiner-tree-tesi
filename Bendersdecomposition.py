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
P = range(len(data))  # Insieme di nodi dati
num_steiner_nodes = len(data) - 2  # Numero di nodi di Steiner (adatta al problema)
X = range(num_steiner_nodes)  # Nodi di Steiner
d = 2
xp.init('C:/xpressmp/bin/xpauth.xpr')
# Crea il modello

d = 2  # Dimensione dello spazio R^d

#Start defining the MASTER PROBLEM
# Variabili di decisione

ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}                # y_pq binarie
zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X }       # z_pq binarie
theta =  xp.var(vartype=xp.continuous)

# Creazione del problema
problem = xp.problem(name="Master problem")
problem.addVariable([ypq[key] for key in ypq])
problem.addVariable([zpq[key] for key in zpq])
problem.addVariable(theta)


# Funzione obiettivo
obj = (theta)
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

# Risoluzione del problema
problem.solve()
status = problem.getProbStatus()
if status == xp.mip_optimal:
    ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
    zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

    print("Soluzione ottimale Master trovata!")
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
else:
    print(f"Stato del problema: {status}")

#Now we pass to the subproblem
xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous),
        'Y': xp.var(vartype=xp.continuous)
    }
    for k in X
} # x^p in R^d
spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP


# Creazione del problema
subproblem = xp.problem(name="Subproblem")
# Aggiunta delle variabili al modello
subproblem.addVariable([xp_var[key] for key in xp_var])
subproblem.addVariable([spq[key] for key in spq])
subproblem.addVariable([tpq[key] for key in tpq])

# Funzione obiettivo
obj = (xp.Sum( spq[p,q]*zpq_solution[p,q] for p in X for q in X) + xp.Sum(tpq[p,q]*ypq_solution[p,q] for p in P for q in X))
subproblem.setObjective(obj, sense=xp.minimize)

for p in X:
   for q in X:
     if p<q:
        # Vincolo SOC riformulato
        lhs = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y' ]) ** 2 for k in range(d))
        subproblem.addConstraint(lhs <= spq[p, q]**2)
for p in P:
    for q in X:
         lhs2 = xp.Sum((xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
         subproblem.addConstraint(lhs2 <= tpq[p, q] ** 2)


# Risoluzione del problema
subproblem.solve()
status = subproblem.getProbStatus()
if status == xp.mip_optimal:
    xp_solution = {key: subproblem.getSolution(xp_var[key]) for key in xp_var}
    tpq_solution = {key: problem.getSolution(tpq[key]) for key in ypq}
    spq_solution = {key: problem.getSolution(spq[key]) for key in zpq}

    print("Soluzione ottimale trovata!")
    print("xp:", xp_solution)
    print("ypq:", tpq_solution)
    print("zpq:", spq_solution)
else:
    print(f"Stato del problema: {status}")


max_iters = 2
for iteration in range(max_iters):
    print(f"\nIterazione {iteration + 1}")

    problem.solve()

    status = problem.getProbStatus()
    if status == xp.mip_optimal:
        ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

        print("Soluzione ottimale Master trovata!")
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
    else:
        print(f"Stato del problema: {status}")

    xp_var = {
        k: {
            'X': xp.var(vartype=xp.continuous),
            'Y': xp.var(vartype=xp.continuous)
        }
        for k in X
    }  # x^p in R^d
    spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in
           X}  # dichiarazione ulteriore variabili per far si che diventi MISOCP
    tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in
           X}  # dichiarazione ulteriore variabili per far si che diventi MISOCP

    # Creazione del problema
    subproblem = xp.problem(name="Subproblem")
    # Aggiunta delle variabili al modello
    subproblem.addVariable([xp_var[key] for key in xp_var])
    subproblem.addVariable([spq[key] for key in spq])
    subproblem.addVariable([tpq[key] for key in tpq])

    # Funzione obiettivo
    obj = (xp.Sum(spq[p, q] * zpq_solution[p, q] for p in X for q in X) + xp.Sum(
        tpq[p, q] * ypq_solution[p, q] for p in P for q in X))
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints=[]
    constraints1=[]
    for p in X:
        for q in X:
            if p < q:
                # Vincolo SOC riformulato
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

    # Risoluzione del problema
    subproblem.solve()
    status = subproblem.getProbStatus()
    if status == xp.mip_optimal:
        xp_solution = {key: subproblem.getSolution(xp_var[key]) for key in xp_var}
        tpq_solution = {key: problem.getSolution(tpq[key]) for key in ypq}
        spq_solution = {key: problem.getSolution(spq[key]) for key in zpq}

        print("Soluzione ottimale trovata!")
        print("xp:", xp_solution)
        print("ypq:", tpq_solution)
        print("zpq:", spq_solution)
        #problem.addConstraint(optimality_cut)
        #print(f"Aggiunto feasibility cut: {optimality_cut}")

    # Controlla infeasibility del subproblem
    if subproblem.getProbStatus() == 1:
        print("Subproblem infeasible! Generazione di un feasibility cut.")

        # Ottieni i Farkas Multipliers per tutti i vincoli
        farkas_multipliers = subproblem.getDual(constraints)
        farkas_multipliers1 = subproblem.getDual(constraints1)
        print(f"Farkas Multipliers: {farkas_multipliers} {farkas_multipliers1}  ")

        #Costruisci il feasibility cut per tutti i vincoli
        feasibility_cut = xp.Sum(
            farkas_multipliers[i] * (
                    xp.Sum(
                        (xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d)
                    ) - spq[p, q] ** 2
            ) for i, (p, q) in enumerate((p, q) for p in X for q in X if p < q)
        ) + xp.Sum(
            farkas_multipliers1[j] * (
                    xp.Sum(
                        (xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in
                        range(d)
                    ) - tpq[p, q] ** 2
            ) for j, (p, q) in enumerate((p, q) for p in P for q in X)
        ) <= 0
        problem.addConstraint(feasibility_cut)
        print(f"Aggiunto feasibility cut: {feasibility_cut}")


