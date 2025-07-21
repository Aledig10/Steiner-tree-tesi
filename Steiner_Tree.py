import xpress as xp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pdb

# In this Python file, we solve the Euclidean Steiner Tree problem using the Fampa-Maculan mathematical
# programming model implemented with FICO Xpress 9.6.
# The mathematical programming model is defined as follows:
# Given a set P of terminal points, and setting the number of Steiner points equal
# to the number of terminal points minus 2, the model is formulated as follows:

# Minimize:  sum_{p<q ∈ X} v_pq + sum_{p ∈ P, q ∈ X} w_pq
#s.t
# sum_{q ∈ X} y_pq == 1   ∀ p ∈ P
# sum_{p ∈ P} y_pq + sum_{p < q ∈ X} z_pq + sum_{p > q ∈ X} z_qp == 3   ∀ q ∈ X
# sum_{p < q ∈ X} z_pq == 1   ∀ q ∈ X, q > 1
# ||x_q - x_p||^2 ≤ s_pq²   ∀ p < q ∈ X
# ||x_q[k] - zeta_p[k]||^2 ≤ t_pq²   ∀ p ∈ P, q ∈ X
# w_pq ≥ t_pq - M_p(1 - y_pq)   ∀ p ∈ P, q ∈ X
# v_pq ≥ s_pq - M(1 - z_pq)   ∀ p < q ∈ X
# y_pq ∈ {0,1}  ∀ p ∈ P, q ∈ X
# z_pq ∈ {0,1}  ∀ p,q ∈ X, p<q
# x_p ∈  R^n



def MINLP_formulation(Mp,M,coordinates_p,X,P,d,coordinates_column):
    # problem creation
    total = xp.problem(name="Steiner Tree")
    xp_var = {}
    ypq = {(p, q): total.addVariable(name=f"y_{p}_{q}", vartype=xp.binary) for p in P for q in X}
    zpq = {(p, q): total.addVariable(name=f"z_{p}_{q}", vartype=xp.binary) for p in X for q in X if p < q}
    for k in X:
        xp_var[k] = {}
        for dim in coordinates_column:
            xp_var[k][dim] = total.addVariable(name=f"xp_{k}_{dim}", vartype=xp.continuous)
    spq = {(p, q): total.addVariable(name=f"s_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    tpq = {(p, q): total.addVariable(name=f"t_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): total.addVariable(name=f"w_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    vpq = {(p, q): total.addVariable(name=f"v_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}

    # Objective function
    #Minimize:  sum_{p<q ∈ X} v_pq + sum_{p ∈ P, q ∈ X} w_pq
    obj = (xp.Sum(vpq[p, q] for p in X for q in X if p < q) + xp.Sum(wpq[p, q] for p in P for q in X))
    total.setObjective(obj, sense=xp.minimize)

    # Constraints
    # sum_{q ∈ X} y_pq == 1   ∀ p ∈ P
    for p in P:
        total.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

    # sum_{p ∈ P} y_pq + sum_{p < q ∈ X} z_pq + sum_{p > q ∈ X} z_qp == 3   ∀ q ∈ X
    for q in X:
        total.addConstraint(
            xp.Sum(ypq[p, q] for p in P)
            + xp.Sum(zpq[p, q] for p in X if p < q)
            + xp.Sum(zpq[q, p] for p in X if p > q) == 3
        )

    # sum_{p < q ∈ X} z_pq == 1   ∀ q ∈ X, q > 1
    for q in X:
        if q > 1:
            total.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

    #for q in X:
     #   total.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)

    # ||x_q - x_p||^2 ≤ s_pq²   ∀ p < q ∈ X
    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (xp_var[q][coordinates_column[k]] - xp_var[p][coordinates_column[k]]) ** 2
                    for k in range(d)
                )
                total.addConstraint(lhs <= spq[p, q] ** 2)

    # ||x_q[k] - zeta_p[k]||^2 ≤ t_pq²   ∀ p ∈ P, q ∈ X
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (xp_var[q][coordinates_column[k]] - coordinates_p[p][coordinates_column[k]]) ** 2
                for k in range(d)
            )
            total.addConstraint(lhs2 <= tpq[p, q] ** 2)

    # w_pq ≥ t_pq - M_p(1 - y_pq)   ∀ p ∈ P, q ∈ X
    for p in P:
        for q in X:
            total.addConstraint(wpq[p, q] >= tpq[p, q] - Mp[p] * (1 - ypq[p, q]))

    # v_pq ≥ s_pq - M(1 - z_pq)   ∀ p < q ∈ X
    for p in X:
        for q in X:
            if p < q:
                total.addConstraint(vpq[p, q] >= spq[p, q] - M * (1 - zpq[p, q]))

    # Problem solution
    # We set the number of threads to 1 so that we can compare the runtime with the code using Benders decomposition.
    total.controls.threads = 1
    #The maximum computation time is set to 2 hours.
    total.controls.maxtime = 7200
    total.solve()
    status = total.attributes.solstatus
    LB = total.attributes.objval
    print("LB", LB)
    if status == 1:
        xp_solution = {key: total.getSolution(xp_var[key]) for key in xp_var}
        ypq_solution = {key: total.getSolution(ypq[key]) for key in ypq}
        zpq_solution = {key: total.getSolution(zpq[key]) for key in zpq}
        tpq_solution = {key: total.getSolution(tpq[key]) for key in tpq}
        print("Optimal solution found")
        print("xp:", xp_solution)
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
        print("tpq", tpq_solution)
    else:
        print(f"Problem status: {status}")
    return (ypq_solution, zpq_solution,xp_solution,LB)

def plot_the_graph(coordinates_p,xp_solution,ypq_solution,zpq_solution,P,X,d,coordinate_columns):
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

        plt.title(f"Graph")
        plt.xlabel(dim1)
        plt.ylabel(dim2)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='datalim')
        #plt.show()
        plt.savefig('grafo.png')

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
        #plt.show()
        plt.savefig('grafo.png')

    else:
        print(f" Impossible to do the plot for d = {d}.")

def solve():

    start_time = time.time()

    file_name = sys.argv[1]

    # Read files
    data = pd.read_csv(file_name, sep='\s+')
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

    Mp = []
    ObjAbsAccuracy = 0.0001

    # Define M_P, the value of the first big-M
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

    #Define M, the value for the second big-M constraint
    M = 0
    for p in P:
        for z in P:
            if z != p:
                distanza1 = np.linalg.norm(
                    np.array([coordinates_p[p][dim] - coordinates_p[z][dim] for dim in coordinate_columns])
                )
                if distanza1 > M:
                    M = distanza1
    print(M)
    print(Mp)

    (ypq_solution, zpq_solution,xp_solution, optimum_general) = MINLP_formulation(Mp, M, coordinates_p, X, P, d,coordinate_columns)
    print("Optimal solution found")
    print("ypq:", ypq_solution)
    print("zpq:", zpq_solution)
    print(xp_solution)
    print(optimum_general)


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.6f} secondi")
    plot_the_graph(coordinates_p, xp_solution, ypq_solution, zpq_solution, P, X, d, coordinate_columns)



if __name__ == '__main__':
   solve()