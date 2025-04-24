import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cplex
from cplex.exceptions import CplexError

start_time = time.time()

# Read the data from CSV
data = pd.read_csv('istanza5.csv', sep='\s+')
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
Mp = []
distanza = 0;
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
M = 0
for p in P:
    for z in P:
        if z != p:
            distanza1 = np.sqrt((coordinates_p[p]['X'] - coordinates_p[z]['X']) ** 2 + (
                    coordinates_p[p]['Y'] - coordinates_p[z]['Y']) ** 2)
            if M < distanza1:
                M = distanza1
print(M)
d = 2  # Space dimension
# Start defining the MASTER PROBLEM
# Decision Variable
import cplex
from cplex.exceptions import CplexError

# Definizione del modello
problem = cplex.Cplex()
problem.objective.set_sense(problem.objective.sense.minimize)

var_names = []
var_lb = []
var_ub = []
var_types = []
var_index = {}

for k in X:
    for coord in ['X', 'Y']:
        name = f"{coord}_{k}"
        var_names.append(name)
        var_lb.append(-cplex.infinity)
        var_ub.append(cplex.infinity)
        var_types.append('C')
        var_index[name] = len(var_index)

for p in X:
    for q in X:
        if p < q:
            name = f"s_{p}_{q}"
            var_names.append(name)
            var_lb.append(0.0)
            var_ub.append(cplex.infinity)
            var_types.append('C')
            var_index[name] = len(var_index)

for p in P:
    for q in X:
        name = f"t_{p}_{q}"
        var_names.append(name)
        var_lb.append(0.0)
        var_ub.append(cplex.infinity)
        var_types.append('C')
        var_index[name] = len(var_index)

# === 4. Variabili wpq e vpq (per funzione obiettivo) ===
for p in P:
    for q in X:
        name = f"w_{p}_{q}"
        var_names.append(name)
        var_lb.append(0.0)
        var_ub.append(cplex.infinity)
        var_types.append('C')
        var_index[name] = len(var_index)

for p in X:
    for q in X:
        if p < q:
            name = f"v_{p}_{q}"
            var_names.append(name)
            var_lb.append(0.0)
            var_ub.append(cplex.infinity)
            var_types.append('C')
            var_index[name] = len(var_index)
for p in P:
    name = f"Xc_{p}"
    var_names.append(name)
    var_lb.append(0.0)
    var_ub.append(cplex.infinity)
    var_types.append('C')
    var_index[name] = len(var_index)
for p in P:
    name = f"Yc_{p}"
    var_names.append(name)
    var_lb.append(0.0)
    var_ub.append(cplex.infinity)
    var_types.append('C')
    var_index[name] = len(var_index)

# Aggiunta delle variabili al modello
problem.variables.add(names=var_names, lb=var_lb, ub=var_ub)

objective_vars = []
objective_coeffs = []

for p in X:
    for q in X:
        if p < q:
            name = f"v_{p}_{q}"
            objective_vars.append(name)
            objective_coeffs.append(1.0)

for p in P:
    for q in X:
        name = f"w_{p}_{q}"
        objective_vars.append(name)
        objective_coeffs.append(1.0)

problem.objective.set_linear(list(zip(objective_vars, objective_coeffs)))
for p in P:
    problem.linear_constraints.add(
        lin_expr=[[[f'Xc_{p}'], [1.0]]],
        senses=["E"],
        rhs=[coordinates_p[p]['X']],
    )
for p in P:
    problem.linear_constraints.add(
        lin_expr=[[[f'Yc_{p}'], [1.0]]],
        senses=["E"],
        rhs=[coordinates_p[p]['Y']],
    )

for p in X:
    for q in X:
        if p < q:
            x1p = var_index[f"X_{p}"]
            y1p = var_index[f"Y_{p}"]
            x1q = var_index[f"X_{q}"]
            y1q = var_index[f"Y_{q}"]
            s_pq = var_index[f"s_{p}_{q}"]

            # (X_q - X_p)^2 + (Y_q - Y_p)^2 - s_pq^2 <= 0
            #  (X_q^2 - 2*X_q*X_p + X_p^2) + (Y_q^2 - 2*Y_q*Y_p + Y_p^2) - s_pq^2 <= 0
            quad_expr = cplex.SparseTriple(
                ind1=[x1p, x1q, x1p, y1p, y1q, y1p, s_pq],  # Indices of variables
                ind2=[x1p, x1q, x1q, y1p, y1q, y1q, s_pq],  # Indices of variables again
                val=[1.0, 1.0, -2.0, 1.0, 1.0, -2.0, -1.0]  # Coefficients for each pair (terms)
            )

            lin_expr = cplex.SparsePair(
                ind=[],
                val=[]
            )

            problem.quadratic_constraints.add(
                name=f"qc_spq_{p}_{q}",
                quad_expr=quad_expr,
                lin_expr=lin_expr,
                sense="L",
                rhs=0.0
            )
for p in P:
    for q in X:
        x1q = var_index[f"X_{q}"]
        y1q = var_index[f"Y_{q}"]
        t_pq = var_index[f"t_{p}_{q}"]
        x_p = var_index[f"Xc_{q}"]
        y_p = var_index[f"Yc_{q}"]

        #  t_pq^2 >= (X_q - x_p)^2 + (Y_q - y_p)^2
        # t_pq^2 - (X_q^2 - 2x_p X_q + x_p^2 + Y_q^2 - 2y_p Y_q + y_p^2) >= 0

        quad_expr = cplex.SparseTriple(
            ind1=[x1q, x_p, x1q, y1q, y_p, y1q, t_pq],
            ind2=[x1q, x_p, x_p, y1q, y_p, y_p, t_pq],
            val=[1.0, 1.0, -2.0, 1.0, 1.0, -2.0, -1.0]
        )

        # Espressione lineare: -2*x_p*X_q -2*y_p*Y_q
        lin_expr = cplex.SparsePair(
            ind=[],
            val=[]
        )

        # Aggiungi il vincolo
        problem.quadratic_constraints.add(
            name=f"qc_tpq_{p}_{q}",
            quad_expr=quad_expr,
            lin_expr=lin_expr,
            sense="L",
            rhs=0
        )
problem.write("modellocplex.lp")
max_iters = 10
UB = 100
LB = 0
iteration = 0

while iteration < max_iters:
    try:
        problem.solve()
        print("Stato:", problem.solution.get_status_string())
        print("Obiettivo:", problem.solution.get_objective_value())
    except CplexError as e:
        print("Errore nella risoluzione:", e)
    status = problem.solution.get_status()
    print("Stato:", status)
    if status == 1:

        valori = problem.solution.get_values()
        nomi = problem.variables.get_names()

        soluzione_completa = dict(zip(nomi, valori))

        xp_solution = {k: v for k, v in soluzione_completa.items() if k.startswith("X_") or k.startswith("Y_")}
        spq_solution = {k: v for k, v in soluzione_completa.items() if k.startswith("s_")}
        tpq_solution = {k: v for k, v in soluzione_completa.items() if k.startswith("t_")}
        wpq_solution = {k: v for k, v in soluzione_completa.items() if k.startswith("w_")}
        vpq_solution = {k: v for k, v in soluzione_completa.items() if k.startswith("v_")}

        print("Soluzione ottimale Master trovata!")
        print("xpq:", xp_solution)
        print("wpq:", wpq_solution)
        print("tpq:", tpq_solution)
        print("vpq:", vpq_solution)
        print("spq:", spq_solution)
        print("Obj:", problem.solution.get_objective_value())

    else:
        print(f"Stato del problema: {status}")

    # SUBPROBLEM
    subproblem = cplex.Cplex()
    subproblem.set_problem_type(cplex.Cplex().problem_type.LP)
    subproblem.objective.set_sense(subproblem.objective.sense.minimize)

    ypq = {(p, q): f"y_{p}_{q}" for p in P for q in X}
    zpq = {(p, q): f"z_{p}_{q}" for p in X for q in X if p < q}

    subproblem.variables.add(names=list(ypq.values()),
                             lb=[0.0] * len(ypq),
                             ub=[1.0] * len(ypq))
    subproblem.variables.add(names=list(zpq.values()),
                             lb=[0.0] * len(zpq),
                             ub=[1.0] * len(zpq))

    # Vincoli
    constraints = []
    rhs = []
    senses = []
    names = []

    for p in P:
        for q in X:
            name = f"c_wpq_{p}_{q}"
            y_name = f"y_{p}_{q}"
            w_name = f"w_{p}_{q}"
            t_name = f"t_{p}_{q}"

            rhs_value = -wpq_solution[w_name] + tpq_solution[t_name] - Mp[p]

            subproblem.linear_constraints.add(
                lin_expr=[[[y_name], [-Mp[p]]]],
                senses=["G"],
                rhs=[rhs_value],
                names=[name]
            )

    for p in X:
        for q in X:
            if p < q:
                name = f"c_vpq_{p}_{q}"
                z_name = f"z_{p}_{q}"
                v_name = f"v_{p}_{q}"
                s_name = f"s_{p}_{q}"

                rhs_value = -vpq_solution[v_name] + spq_solution[s_name] - M

                subproblem.linear_constraints.add(
                    lin_expr=[[[z_name], [-M]]],
                    senses=["G"],
                    rhs=[rhs_value],
                    names=[name]
                )

    for p in P:
        name = f"assign_{p}"
        indices = [ypq[p, q] for q in X]
        coefs = [1.0] * len(indices)
        subproblem.linear_constraints.add(
            lin_expr=[[indices, coefs]],
            senses=["E"],
            rhs=[1.0],
            names=[name]
        )

    for q in X:
        name = f"flow_{q}"
        expr_vars = [ypq[p, q] for p in P] + [zpq[p, q] for p in X if p < q] + [zpq[q, p] for p in X if p > q]
        coefs = [1.0] * len(expr_vars)
        subproblem.linear_constraints.add(
            lin_expr=[[expr_vars, coefs]],
            senses=["E"],
            rhs=[3.0],
            names=[name]
        )

    for q in X:
        if q > 0:
            name = f"one_in_{q}"
            indices = [zpq[p, q] for p in X if p < q]
            coefs = [1.0] * len(indices)
            subproblem.linear_constraints.add(
                lin_expr=[[indices, coefs]],
                senses=["E"],
                rhs=[1.0],
                names=[name]
            )

    for q in X:
        name = f"capacity_{q}"
        indices = [ypq[p, q] for p in P]
        subproblem.linear_constraints.add(
            lin_expr=[[indices, [1.0] * len(indices)]],
            senses=["L"],
            rhs=[2.0],
            names=[name]
        )

    for p, q in ypq:
        subproblem.linear_constraints.add(
            lin_expr=[[[ypq[p, q]], [1.0]]],
            senses=["L"],
            rhs=[1.0]
        )

    for p, q in zpq:
        subproblem.linear_constraints.add(
            lin_expr=[[[zpq[p, q]], [1.0]]],
            senses=["L"],
            rhs=[1.0]
        )
    subproblem.write("modello.lp")
    print("Il problema è MIP?", subproblem.problem_type == cplex.ProblemType.MILP)
    subproblem.solve()
    if subproblem.solution.get_status() == 3:
        print("Subproblem infeasible, retrieving dual Farkas ray...")
        ray = subproblem.solution.advanced.dual_farkas()
        print("Dual Farkas:", ray)
        # Let's build the feasibility cut
        k = sum(1 for _ in P for _ in X)
        u = sum(1 for p in X for q in X if p < q)
        pairs = [(p, q) for p in P for q in X]
        var_namess = []
        coeffs = []

        index = 0
        for p in P:
            for q in X:
                name = f"farkas_{index}"
                var_namess.append(name)
                coeffs.append(-var_index[f"w_{p}_{q}"] + var_index[f"t_{p}_{q}"] - Mp[p])
                index += 1

        for p in X:
            for q in X:
                if p < q:
                    name = f"farkas_{index}"
                    var_names.append(name)
                    coeffs.append(-var_index[f"v_{p}_{q}"] + var_index[f"s_{p}_{q}"] - M)
                    index += 1

        for i in P:
            name = f"farkas_{index}"
            var_names.append(name)
            coeffs.append(1.0)
            index += 1

        for i in X:
            name = f"farkas_{index}"
            var_names.append(name)
            coeffs.append(3.0)
            index += 1

        for i in X:
            if i > 1:
                name = f"farkas_{index}"
                var_names.append(name)
                coeffs.append(1.0)
                index += 1

        for i in X:
            name = f"farkas_{index}"
            var_names.append(name)
            coeffs.append(-2.0)
            index += 1

        problem.linear_constraints.add(
            lin_expr=[[var_names, coeffs]],
            senses=["L"],
            rhs=[0.0],
            names=["feasibility_cut"]
        )


    else:
        sol_status = subproblem.solution.get_status()
        if sol_status == 101:
            print("Soluzione subproblem trovata.")
            ypq_solution = {key: subproblem.solution.get_values(ypq[key]) for key in ypq}
            zpq_solution = {key: subproblem.solution.get_values(zpq[key]) for key in zpq}
            print("ypq:", ypq_solution)
            print("zpq:", zpq_solution)
            break
        else:
            print(f"Subproblem risolto ma senza soluzione accessibile. Status: {sol_status}")
    iteration += 1
    print(iteration)
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
plt.figure(figsize=(8, 8))
for p in P:
    plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50,
                label='Points P' if p == list(P)[0] else "")

for i, p in enumerate(X):
    x = xp_solution[f"X_{p}"]
    y = xp_solution[f"Y_{p}"]
    plt.scatter(x, y, color='red', s=50, label='Points X' if i == 0 else "")

for (p, q), val in ypq_solution.items():
    if val == 1.0:
        x1, y1 = coordinates_p[p]["X"], coordinates_p[p]["Y"]
        x2, y2 = xp_solution[f"X_{q}"], xp_solution[f"Y_{q}"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")

for (p, q), val in zpq_solution.items():
    if val == 1:  # Active connection
        x1, y1 = xp_solution[f"X_{p}"], xp_solution[f"Y_{p}"]
        x2, y2 = xp_solution[f"X_{q}"], xp_solution[f"Y_{q}"]
        if x1 is not None and x2 is not None:
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

plt.title("Points in the 2D space")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
