import xpress as xp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import sys

# This Python script solves the Euclidean Steiner Tree problem using Benders Decomposition,
# applied to the Fampa-Maculan mathematical programming model, with FICO Xpress 9.6.
# Benders Decomposition divides the original problem into two smaller problems,
# each involving a subset of the variables. These are known as the master problem and the subproblem.
# In this implementation, all continuous variables are included in the master problem,
# while the integer are handled in the subproblem.
# The method works by iteratively solving a relaxed master problem, using its solution
# to generate cuts from the subproblem, and adding these cuts back to the master problem
# to progressively improve it until convergence is achieved.

#Let's look on how the problem is defined

#Master problem


# Minimize:  sum_{p<q ∈ X} v_pq + sum_{p ∈ P, q ∈ X} w_pq
#s.t
# ||x_q - x_p||^2 ≤ s_pq²   ∀ p < q ∈ X
# ||x_q[k] - zeta_p[k]||^2 ≤ t_pq²   ∀ p ∈ P, q ∈ X
# x_p ∈  R^n
#Optimality cut and feasibility cut added at every iteration


#Subproblem

# Minimize: 0
#s.t
# sum_{q ∈ X} y_pq == 1   ∀ p ∈ P
# sum_{p ∈ P} y_pq + sum_{p < q ∈ X} z_pq + sum_{p > q ∈ X} z_qp == 3   ∀ q ∈ X
# sum_{p < q ∈ X} z_pq == 1   ∀ q ∈ X, q > 1
# w_pq ≥ t_pq - M_p(1 - y_pq)   ∀ p ∈ P, q ∈ X
# v_pq ≥ s_pq - M(1 - z_pq)   ∀ p < q ∈ X
# y_pq ∈ {0,1}  ∀ p ∈ P, q ∈ X
# z_pq ∈ {0,1}  ∀ p,q ∈ X, p<q


def plot_the_graph(coordinates_p, xp_solution, ypq_solution, zpq_solution, P, X, d, coordinate_columns):
    # This function builds the plot of the Steiner tree solution,
    # distinguishing between 2D and 3D cases, and displaying an error message
    # if the dimension is greater than 3.

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


def Subproblem(X, P, Mp,M,coordinate_columns, coordinates_p):
    #In this function we are going to define the subproblem
    subproblem = xp.problem(name="Subproblem")
    ypq = {(p, q): subproblem.addVariable(name=f"y_{p}_{q}", vartype=xp.continuous) for p in P for q in X}
    zpq = {(p, q): subproblem.addVariable(name=f"z_{p}_{q}", vartype=xp.continuous) for p in X for q in X if p < q}
    obj = 0
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints = []
    constraints1 = []
    constraints2 = []
    constraints3 = []
    constraints4 = []
    # constraints
    for p in P:
        for q in X:
            constraint4 =  Mp[p] * (1 - ypq[p, q]) >= 0
            subproblem.addConstraint(constraint4)
            constraints.append(constraint4)
    for p in X:
        for q in X:
            if p < q:
                constraint5 =  M * (1 - zpq[p, q]) >= 0
                subproblem.addConstraint(constraint5)
                constraints1.append(constraint5)
    for p in P:
        constraint = xp.Sum(ypq[p, q] for q in X) == 1
        subproblem.addConstraint(constraint)
        constraints2.append(constraint)

    for q in X:
        constraint1 = xp.Sum(ypq[p, q] for p in P) + xp.Sum(zpq[p, q] for p in X if p < q) + xp.Sum(
            zpq[q, p] for p in X if p > q) == 3
        subproblem.addConstraint(constraint1)
        constraints3.append(constraint1)

    for q in X:
        if q > 0:
            constraint2 = xp.Sum(zpq[p, q] for p in X if p < q) == 1
            subproblem.addConstraint(constraint2)
            constraints4.append(constraint2)
    for p in P:
        for q in X:
            subproblem.addConstraint(ypq[p, q] <= 1)
    for p in X:
        for q in X:
            if p < q:
                subproblem.addConstraint(zpq[p, q] <= 1)

    Subproblem_variable=(ypq,zpq)
    return(subproblem, constraints, constraints1, constraints2, constraints3, constraints4,Subproblem_variable )


def Master(X,P,coordinate_columns,coordinates_p,d):
    # MASTER PROBLEM
    # Decision variable
    problem = xp.problem(name="Master problem")
    xp_var = {}
    for k in X:
        xp_var[k] = {}
        for dim in coordinate_columns:
            xp_var[k][dim] = problem.addVariable(name=f"xp_{k}_{dim}", vartype=xp.continuous)
    spq = {(p, q): problem.addVariable(name=f"s_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    tpq = {(p, q): problem.addVariable(name=f"t_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): problem.addVariable(name=f"w_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    vpq = {(p, q): problem.addVariable(name=f"v_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}

    obj = (xp.Sum(vpq[p, q] for p in X for q in X if p < q) + xp.Sum(wpq[p, q] for p in P for q in X))
    problem.setObjective(obj, sense=xp.minimize)

    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (xp_var[q][coordinate_columns[k]] - xp_var[p][coordinate_columns[k]]) ** 2
                    for k in range(d)
                )
                problem.addConstraint(lhs <= spq[p, q] ** 2)

        # Vincoli tra punti dati (P) e Steiner
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (xp_var[q][coordinate_columns[k]] - coordinates_p[p][coordinate_columns[k]]) ** 2
                for k in range(d)
            )
            problem.addConstraint(lhs2 <= tpq[p, q] ** 2)

    Master_variable=(xp_var,wpq,vpq,spq,tpq)
    return (problem,Master_variable)

def Benders2():
    file_name = sys.argv[1]
    start_time = time.time()
    # Read data from csv file
    data = pd.read_csv(file_name, sep='\s+')
    data = data.drop(data.columns[0], axis=1)
    data = data.reset_index()
    data['id'] = data.index
    exclude_keywords = ['level', 'index', 'id']
    coordinate_columns = [col for col in data.select_dtypes(include=[np.number]).columns
                          if not any(key in col for key in exclude_keywords)]
    d = len(coordinate_columns)  # Space dimension
    coordinates_p = data.set_index('id')[coordinate_columns].T.to_dict()
    P = range(len(data))
    num_steiner_nodes = len(data) - 2
    X = range(num_steiner_nodes)

    xp.init('C:/xpressmp/bin/xpauth.xpr')
    # Value of first Big-M coefficient
    Mp = []
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
    # Value of second Big-M coefficient
    M = 0
    for p in P:
        for z in P:
            if z != p:
                distanza1 = np.linalg.norm(
                    np.array([coordinates_p[p][dim] - coordinates_p[z][dim] for dim in coordinate_columns])
                )
                if distanza1 > M:
                    M = distanza1
    (problem, Master_variable) = Master(X, P, coordinate_columns, coordinates_p,d)
    xp_var=Master_variable[0]
    wpq=Master_variable[1]
    vpq=Master_variable[2]
    spq=Master_variable[3]
    tpq=Master_variable[4]
    (subproblem, constraints, constraints1, constraints2, constraints3, constraints4,Subproblem_variable) = Subproblem(X, P, Mp,M,coordinate_columns, coordinates_p)
    ypq=Subproblem_variable[0]
    zpq=Subproblem_variable[1]
    start_time = time.time()
    max_iters = 3000
    UB=100
    LB=0
    iteration =0
    while iteration <= max_iters:
        print(f"\nIterazione {iteration + 1}")
        problem.solve()
        LB = problem.getObjVal()
        print(LB)
        status = problem.getProbStatus()
        if status == xp.lp_optimal:
            xp_solution = {key: problem.getSolution(xp_var[key]) for key in xp_var}
            spq_solution = {key: problem.getSolution(spq[key]) for key in spq}
            tpq_solution = {key: problem.getSolution(tpq[key]) for key in tpq}
            wpq_solution= {key: problem.getSolution(wpq[key]) for key in wpq}
            vpq_solution= {key: problem.getSolution(vpq[key]) for key in vpq}
            print("Soluzione ottimale Master trovata!")
            print("xpq:", xp_solution)
            print("wpq:", wpq_solution)
            print("tpq:", tpq_solution)
            print("vpq:", vpq_solution)
            print("spq:", spq_solution)
        else:
            print(f"Stato del problema: {status}")
        rowind=list(constraints)
        rhs_values = []
        j = 0
        for p in P:
            for q in X:
                if j < len(rowind):
                    rhs_value = -wpq_solution[p, q] + tpq_solution[p, q]
                    rhs_values.append(rhs_value)
                j += 1

        subproblem.chgrhs(rowind[:len(rhs_values)], rhs_values)
        rowind2 = list(constraints1)
        rhs_values = []
        for p in X:
            for q in X:
                if p < q:
                    rhs_value = -vpq_solution[p, q] + spq_solution[p, q]
                    rhs_values.append(rhs_value)

        subproblem.chgrhs(rowind2[:len(rhs_values)], rhs_values)
        subproblem.setControl({"presolve":0})
        subproblem.controls.scaling = 0

        subproblem.solve()

        UB = subproblem.getObjVal()
        print("UB", UB)
        status = subproblem.getProbStatus()
        print(subproblem.getProbStatusString())
        if status == xp.lp_optimal:
            ypq_solution = {key: problem.getSolution(ypq[key]) for key in ypq}
            zpq_solution = {key: problem.getSolution(zpq[key]) for key in zpq}

            print("Optimal solution found!")
            print("tpq:", tpq_solution)
            print("spq:", spq_solution)
            break
        if subproblem.getProbStatus() == xp.lp_infeas:
            print(subproblem.getProbStatus())
            print("Subproblem infeasible! Generation of a feasibility cut.")

            constraint = subproblem.getConstraint()
            num_constraints = subproblem.attributes.rows
            print(num_constraints)
            v=subproblem.hasdualray()
            print(v)
            farkas_multipliers = []
            subproblem.getdualray(farkas_multipliers)
            print(f"Farkas Multipliers: {farkas_multipliers} " )
            if iteration==2:
                break
            k = sum(1 for _ in P for _ in X)
            u = sum(1 for p in X for q in X if p<q)
            pairs = [(p, q) for p in P for q in X]
            # Costruisci il feasibility cut per tutti i vincoli
            feasibility_cut = xp.Sum(
                farkas_multipliers[i] * (-wpq[p, q] + tpq[p, q] - Mp[p]
                                         ) for i, (p, q) in enumerate((p, q) for p in P for q in X)
            ) + xp.Sum(
                farkas_multipliers[j + k] * (-vpq[p, q] + spq[p, q] - M
                                             ) for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q)
            ) + xp.Sum(
                farkas_multipliers[i + u + k] for i in P
            ) + xp.Sum(3 * farkas_multipliers[i + u + k + len(P)] for i in X) + xp.Sum(
                farkas_multipliers[i + u + k + len(P) + len(X)] for i in X if i > 1) - xp.Sum(
                2 * farkas_multipliers[i + u + k + len(P) + 2 * len(X) - 1] for i in X) <= 0
            problem.addConstraint(feasibility_cut)
        iteration=iteration+1
        print(iteration)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
    plot_the_graph(coordinates_p, xp_solution, ypq_solution, zpq_solution, P, X, d, coordinate_columns)



if __name__ == '__main__':
    Benders2()


#This approach is more critical respecct to the other Benders decomposition.
# Since the subproblem is always unfeasible, we need to build in every iteration Feasibility cut
#For feasibility cut we need the direction of infeasibility. This direction is only defined when the problem is continuous
#So to generate we need to consider only a linear relaxation of the subproblem. This will lead to a weaker cuts
#that could not make an improvement respect to the perfotmance of the model with all the variables
#proposed by Fampa-Maculan