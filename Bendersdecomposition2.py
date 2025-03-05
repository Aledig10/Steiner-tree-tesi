import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Read the data from CSV
data = pd.read_csv('istanza4.csv', sep='\s+')
data = data.drop(data.columns[0], axis=1)
data = data.reset_index()

print(data.index)
data['id'] = data.index
coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
print(coordinates_p)
P = range(len(data))  # Number of nodes
num_steiner_nodes = len(data) - 2  # Number of Steiner Nodes
X = range(num_steiner_nodes)  # Steiner nodes
d = 2
xp.init('C:/xpressmp/bin/xpauth.xpr')
Mp=[]
distanza =0;
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
print(M)
d = 2  # Space dimension
#Start defining the MASTER PROBLEM
#Decision Variable
xp_var = {
    k: {
        'X': xp.var(vartype=xp.continuous),
        'Y': xp.var(vartype=xp.continuous)
    }
    for k in X
} # x^p in R^d
spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X} #dichiarazione ulteriore variabili per far si che diventi MISOCP
wpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}#additive variable
vpq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}#additive variable

#Create the problem
problem = xp.problem(name="problem")
# Aggiunta delle variabili al modello
problem.addVariable([xp_var[key] for key in xp_var])
problem.addVariable([spq[key] for key in spq])
problem.addVariable([tpq[key] for key in tpq])
problem.addVariable([vpq[key] for key in vpq])
problem.addVariable([wpq[key] for key in wpq])


# Funzione obiettivo
obj = (xp.Sum(vpq[p,q] for p in X for q in X) + xp.Sum(wpq[p,q]for p in P for q in X))
problem.setObjective(obj, sense=xp.minimize)

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

max_iters = 1000
UB=100
LB=0
iteration =0
while iteration <= max_iters and UB-LB>=0.0001:
    print(f"\nIterazione {iteration + 1}")

    problem.solve()
    # Ottieni e stampa il valore della funzione obiettivo
    LB = problem.getObjVal()
    status = problem.getProbStatus()
    if status == xp.lp_optimal:
        xp_solution = {key: problem.getSolution(xp_var[key]) for key in xp_var}
        spq_solution = {key: problem.getSolution(spq[key]) for key in spq}
        tpq_solution = {key: problem.getSolution(tpq[key]) for key in tpq}
        wpq_solution= {key: problem.getSolution(wpq[key]) for key in wpq}
        vpq_solution= {key: problem.getSolution(vpq[key]) for key in vpq}
        print("Soluzione ottimale Master trovata!")
        print("xpq:", xp_solution)
    else:
        print(f"Stato del problema: {status}")

    ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}
    zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X}

    subproblem = xp.problem(name="Subproblem")
    subproblem.addVariable([ypq[key] for key in ypq])
    subproblem.addVariable([zpq[key] for key in zpq])

    # Funzione obiettivo
    obj = 0
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints=[]
    constraints1=[]
    constraints2=[]
    constraints3=[]
    constraints4=[]
    constraints5=[]
    # constraints
    for p in P:
        constraint=xp.Sum(ypq[p, q] for q in X) == 1
        subproblem.addConstraint(constraint)
        constraints2.append(constraint)

    for q in X:
        constraint1= xp.Sum(ypq[p, q] for p in P)+ xp.Sum(zpq[p, q] for p in X if p < q)  + xp.Sum(zpq[q, p] for p in X if p > q) == 3
        subproblem.addConstraint(constraint1)
        constraints3.append(constraint1)

    for q in X:
        if q > 1:
            constraint2=xp.Sum(zpq[p, q] for p in X if p < q) == 1
            subproblem.addConstraint(constraint2)
            constraints4.append(constraint2)
    for q in X:
        constraint3=xp.Sum(ypq[p, q] for p in P) <= 2
        subproblem.addConstraint(constraint3)
        constraints5.append(constraint3)
    for p in P:
        for q in X:
            constraint4=wpq_solution[p, q] >= tpq_solution[p, q] - Mp[p] * (1 - ypq[p, q])
            subproblem.addConstraint(constraint4)
            constraints.append(constraint4)
    for p in X:
        for q in X:
            subproblem.addConstraint(vpq_solution[p, q] >= spq_solution[p, q] - M * (1 - zpq[p, q]))

    # Risoluzione del problema
    subproblem.solve()
    UB = subproblem.getObjVal()
    status = subproblem.getProbStatus()
    print(subproblem.getProbStatusString())
    if status == xp.mip_optimal:
        ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

        print("Soluzione ottimale trovata!")
        print("ypq:", tpq_solution)
        print("zpq:", spq_solution)
        break
    if subproblem.getProbStatus() == xp.lp_infeas:
        print(subproblem.getProbStatus())
        print("Subproblem infeasible! Generazione di un feasibility cut.")

        farkas_multipliers= subproblem.getdualray(constraints)
        farkas_multipliers1 = subproblem.getdualray(constraints1)
        farkas_multipliers2 = subproblem.getdualray(constraints2)
        farkas_multipliers3 = subproblem.getdualray(constraints3)
        farkas_multipliers4 = subproblem.getdualray(constraints4)
        farkas_multipliers5 = subproblem.getdualray(constraints5)
        print(f"Farkas Multipliers: {farkas_multipliers} {farkas_multipliers1} {farkas_multipliers2} {farkas_multipliers3}\
            {farkas_multipliers4} {farkas_multipliers5}" )

        #Costruisci il feasibility cut per tutti i vincoli
        feasibility_cut = (xp.Sum(
            farkas_multipliers[i] * (-wpq[p, q] + tpq[p, q] - Mp[p]
                                     ) for i, (p, q) in enumerate((p, q) for p in P for q in X)
        ) + xp.Sum(
            farkas_multipliers1[j] * (-vpq + spq[p, q] - M
            ) for j, (p, q) in enumerate((p, q) for p in X for q in X)
        )+ xp.Sum(
            farkas_multipliers2[i] for i in P
        ) + xp.Sum(-3*farkas_multipliers3[i] for i in X)
        +xp.Sum(-farkas_multipliers4[i] for i in X and i>1)
        +xp.Sum(-2*farkas_multipliers5[i] for i in X) <= 0)
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
