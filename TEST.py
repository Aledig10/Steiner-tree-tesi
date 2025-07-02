import xpress as xp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pdb



class Callbacks:
    def __init__(self):
        self.Integer_solution=[]
        self.Integer_index=[]
        self.best_solution=0
        self.best=[]

    def cb_preintsol(self,problem,data, soltype, cutoff):
        try:
            mindex=data[0]
            P=data[1]
            X=data[2]
            Mp=data[3]
            M=data[4]
            coordinates_p=data[5]
            ypq=data[6]
            zpq=data[7]
            theta=data[8]
            ypq_val=data[9]
            zpq_val=data[10]
            optimum_general=data[11]
            ObjAbsAccuracy=0.00001
            print('===================== PREINTSOL ========================')
            ypq_vals = problem.getCallbackSolution(list(ypq.values()))
            ypq_solution = dict(zip(ypq.keys(), ypq_vals))
            zpq_vals = problem.getCallbackSolution(list(zpq.values()))
            zpq_solution = dict(zip(zpq.keys(), zpq_vals))
            (optimum, valori, duali) = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution,X,P)
            (colind, cutcoef, rhs_final,index_y,index_z) = cut_generation(duali, coordinates_p, Mp, M,ypq,zpq, theta, P, X)
            cuttype = [1]
            rowtype = ['L']
            rhs = [rhs_final]
            start = [0, len(colind)]
            thetasol = problem.getCallbackSolution(theta)
            print(f"Check cut condition: theta = {thetasol}, subproblem value = {optimum}, ObjAbsAccuracy = {ObjAbsAccuracy}")
            if thetasol >= optimum - ObjAbsAccuracy:
                print('===================== PREINTSOL : FALSE ========================')
                return (False, optimum)
            else:
                value=0
                for j, (p, q) in enumerate(index_y):
                    value += ypq_val[p, q] * cutcoef[j]
                for j, (p, q) in enumerate(index_z):
                    value = value + zpq_val[p, q] * cutcoef[j + len(index_y)]
                if value - rhs_final - optimum_general > 0:
                    print("TAGLIO VIOLATO IN PREINTSOL")
                    pdb.set_trace()
                helplist = []
                problem.storecuts(2, cuttype, rowtype, rhs, start, helplist, colind, cutcoef)
                mindex.append(helplist[0])
                if thetasol >= self.best_solution + ObjAbsAccuracy:
                    self.best_solution = thetasol
                    soluzione = list(ypq_solution.values()) + list(zpq_solution.values()) + [optimum]
                    ypq_indices = [problem.getIndex(ypq[p, q]) for p in P for q in X]
                    zpq_indices = [problem.getIndex(zpq[p, q]) for p in X for q in X if p < q]
                    theta_index = problem.getIndex(theta)
                    indici = list(ypq_indices) + list(zpq_indices) + [theta_index]
                    self.Integer_index.append(indici)
                    self.Integer_solution.append(soluzione)
                print('===================== PREINTSOL:TRUE ========================')
                return (True,optimum)
        except Exception as e:
            print("ERRORE:", e)

    def cb_optnode(self,problem, data):
        try:
            print('===================== OPTNODE ========================')
            mindex=data[0]
            P=data[1]
            X=data[2]
            Mp=data[3]
            M=data[4]
            ypq=data[6]
            zpq=data[7]
            theta=data[8]
            ypq_val=data[9]
            zpq_val=data[10]
            optimum_general=data[11]
            ObjAbsAccuracy=0.00001
            coordinates_p=data[5]
            ypq_vals = problem.getCallbackSolution(list(ypq.values()))
            ypq_solution = dict(zip(ypq.keys(), ypq_vals))
            zpq_vals = problem.getCallbackSolution(list(zpq.values()))
            zpq_solution = dict(zip(zpq.keys(), zpq_vals))
            thetasol = problem.getCallbackSolution(theta)
            for i, val in zip(self.Integer_index, self.Integer_solution):
                print(i)
                problem.addmipsol(val,i, 'From_preintsol')
            print("soluzione")
            print(self.Integer_solution)
            print(self.best_solution)
            del self.Integer_index[:]
            del self.Integer_solution[:]
            if len(mindex) > 0:
                problem.loadcuts(0, -1, mindex)
                print(len(mindex), " cuts added")
                del mindex[:]
            (optimum, valori, duali) = subproblem(Mp, M, coordinates_p, ypq_solution, zpq_solution,X,P)
            soluzione = list(ypq_solution.values()) + list(zpq_solution.values()) + [optimum]
            ypq_indices = [problem.getIndex(ypq[p, q]) for p in P for q in X]
            zpq_indices = [problem.getIndex(zpq[p, q]) for p in X for q in X if p < q]
            theta_index = problem.getIndex(theta)
            indici = list(ypq_indices) + list(zpq_indices) + [theta_index]
            integer = self.is_solution_integer(soluzione)

            if integer == True:
                if thetasol >= self.best_solution + ObjAbsAccuracy:
                    problem.addmipsol(soluzione, indici, 'from_optnode')
                    self.best_solution = thetasol

            print(f"Check cut condition: theta = {thetasol}, subproblem value = {optimum}, ObjAbsAccuracy = {ObjAbsAccuracy}")
            if thetasol >= optimum - ObjAbsAccuracy:
                print('===================== OPTNODE: EXIT ========================')
            else:
                epsilon = 1e-7
                (colind, cutcoef, rhs_final, index_y, index_z) = cut_generation(duali, coordinates_p, Mp,M,ypq,zpq, theta, P, X)
                cuttype = [1]
                rowtype = ['L']
                rhs = [rhs_final]
                start = [0, len(colind)]
                print(f"Prima dei tagli - LB: {problem.attributes.lpobjval}, Best: {problem.attributes.bestbound}")
                problem.addcuts(cuttype, rowtype, rhs, start, colind, cutcoef)
                value=0
                for j, (p,q) in enumerate(index_y):
                    value += ypq_val[p, q] * cutcoef[j]
                for j, (p, q) in enumerate(index_z):
                    value= value+zpq_val[p,q]*cutcoef[j+len(index_y)]
                if value-rhs_final-optimum_general>0:
                    print("TAGLIO VIOLATO IN OPTNODE")
                    pdb.set_trace()
            incumbent = problem.attributes.mipobjval
            current_bound = problem.attributes.lpobjval
            global_bound = problem.attributes.bestbound
            print(f"UB: {incumbent}")
            print(f"Current LB: {current_bound}")
            print(f"Global LB: {global_bound}")
            print(f"Gap corrente calcolato: {(problem.attributes.mipobjval - problem.attributes.bestbound) / abs(problem.attributes.mipobjval)}")
            node = problem.attributes.currentnode
            print(f"Node: {node}")
            print('===================== OPTNODE: RIENTRO ========================')
            return 0
        except Exception as e:
            print("ERRORE:", e)

    def is_solution_integer(self,valori, tolerance=1e-6):
        for value in valori:
            if abs(value - round(value)) > tolerance:
                return False
        return True

def MINLP_formulation(Mp,M,coordinates_p,X,P,d):
    xp_var = {
        k: {
            'X': xp.var(vartype=xp.continuous),
            'Y': xp.var(vartype=xp.continuous)
        }
        for k in X
    }  # x^p in R^d
    ypq = {(p, q): xp.var(vartype=xp.binary) for p in P for q in X}  # y_pq binary
    zpq = {(p, q): xp.var(vartype=xp.binary) for p in X for q in X}  # z_pq binary
    spq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}  # additive variable
    tpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}  # additive variable
    wpq = {(p, q): xp.var(vartype=xp.continuous) for p in P for q in X}  # additive variable
    vpq = {(p, q): xp.var(vartype=xp.continuous) for p in X for q in X}  # additive variable

    # problem creation
    total = xp.problem(name="Steiner Tree")
    # Aggiunta delle variabili al modello
    total.addVariable([xp_var[key] for key in xp_var])
    total.addVariable([ypq[key] for key in ypq])
    total.addVariable([zpq[key] for key in zpq])
    total.addVariable([spq[key] for key in spq])
    total.addVariable([tpq[key] for key in tpq])
    total.addVariable([wpq[key] for key in wpq])
    total.addVariable([vpq[key] for key in vpq])

    # Objective function
    obj = (xp.Sum(vpq[p, q] for p in X for q in X if p < q) + xp.Sum(wpq[p, q] for p in P for q in X))
    total.setObjective(obj, sense=xp.minimize)

    # Constraints
    for p in P:
        total.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

    for q in X:
        total.addConstraint(
            xp.Sum(ypq[p, q] for p in P)
            + xp.Sum(zpq[p, q] for p in X if p < q)
            + xp.Sum(zpq[q, p] for p in X if p > q) == 3
        )

    for q in X:
        if q > 1:
            total.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

    for q in X:
        total.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)

    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (xp_var[q]['X' if k == 0 else 'Y'] - xp_var[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
                total.addConstraint(lhs <= spq[p, q] ** 2)
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (xp_var[q]['X' if k == 0 else 'Y'] - coordinates_p[p]['X' if k == 0 else 'Y']) ** 2 for k in range(d))
            total.addConstraint(lhs2 <= tpq[p, q] ** 2)

    for p in P:
        for q in X:
            total.addConstraint(wpq[p, q] >= tpq[p, q] - Mp[p] * (1 - ypq[p, q]))
    for p in X:
        for q in X:
            if p < q:
                total.addConstraint(vpq[p, q] >= spq[p, q] - M * (1 - zpq[p, q]))

    # Problem solution
    total.solve()
    status = total.getProbStatus()
    LB = total.getObjVal()
    print("LB", LB)
    print(total.getProbStatusString())
    if status == xp.mip_optimal:
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
    return (ypq_solution, zpq_solution,LB)



def subproblem(Mp,M,coordinates_p,ypq_values,zpq_values,X,P):
    subproblem = xp.problem(name="Subproblem")
    xp_var = {
        k: {
            'X': subproblem.addVariable(vartype=xp.continuous, name=f"xp_{k}"),
            'Y': subproblem.addVariable(vartype=xp.continuous, name=f"yp_{k}")
        }
        for k in X
    }  # x^p in R^d
    spq = {(p, q): subproblem.addVariable(name=f"s_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    tpq = {(p, q): subproblem.addVariable(name=f"t_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    wpq = {(p, q): subproblem.addVariable(name=f"w_{p}_{q}",
                                          vartype=xp.continuous) for p in P for q in X}
    vpq = {(p, q): subproblem.addVariable(name=f"v_{p}_{q}",
                                          vartype=xp.continuous) for p in X for q in X if p < q}
    deltapq = {(p, q, coord): subproblem.addVariable(vartype=xp.continuous, lb=-1e4, name=f"delta_{p}_{q}_{coord}")
               for p in P for q in X for coord in ['X', 'Y']}

    gammapq = {(p, q, coord): subproblem.addVariable(vartype=xp.continuous, lb=-1e4, name=f"gamma_{p}_{q}_{coord}")
               for p in X for q in X if p < q for coord in ['X', 'Y']}

    # Objective function
    obj = (xp.Sum(vpq[p, q] for p in X for q in X if p < q) +
           xp.Sum(wpq[p, q] for p in P for q in X))
    subproblem.setObjective(obj, sense=xp.minimize)
    constraints = []
    constraints1 = []
    constraints2 = []
    constraints3 = []
    constraints4 = []
    constraints5 = []
    constraints6 = []
    constraints7 = []
    for p in P:
        for q in X:
            constraint = deltapq[(p, q, 'X')] == (-coordinates_p[p]['X'] + xp_var[q]['X'])
            subproblem.addConstraint(constraint)
            constraints4.append(constraint)
            constraint = deltapq[(p, q, 'Y')] == (-coordinates_p[p]['Y'] + xp_var[q]['Y'])
            subproblem.addConstraint(constraint)
            constraints5.append(constraint)
    for p in X:
        for q in X:
            if p < q:
                constraint = gammapq[(p, q, 'X')] == (xp_var[q]['X'] - xp_var[p]['X'])
                subproblem.addConstraint(constraint)
                constraints6.append(constraint)
                constraint = gammapq[(p, q, 'Y')] == (xp_var[q]['Y'] - xp_var[p]['Y'])
                subproblem.addConstraint(constraint)
                constraints7.append(constraint)

    for p in X:
        for q in X:
            if p < q:
                lhs = xp.Sum(
                    (gammapq[(p, q, coord)]) ** 2 for coord in ['X', 'Y'])
                constraint = -lhs + spq[p, q] ** 2 >= 0
                subproblem.addConstraint(constraint)
                constraints.append(constraint)
    for p in P:
        for q in X:
            lhs2 = xp.Sum(
                (deltapq[(p, q, coord)]) ** 2 for coord in ['X', 'Y'])
            constraint1 = -lhs2 + tpq[p, q] ** 2 >= 0
            subproblem.addConstraint(constraint1)
            constraints1.append(constraint1)
    for p in P:
        for q in X:
            constraint2 = wpq[p, q] - tpq[p, q] >= -Mp[p] * (1 - ypq_values[p, q])
            subproblem.addConstraint(constraint2)
            constraints2.append(constraint2)
    for p in X:
        for q in X:
            if p < q:
                constraint3 = vpq[p, q] - spq[p, q] >= -M * (1 - zpq_values[p, q])
                subproblem.addConstraint(constraint3)
                constraints3.append(constraint3)
    subproblem.setControl('outputlog',0)
    subproblem.solve()
    optimum = subproblem.getObjVal()
    status = subproblem.getProbStatus()
    if status == xp.lp_optimal:
        xp_solution = subproblem.getSolution(xp_var)
        tpq_solution = subproblem.getSolution(tpq)
        spq_solution = subproblem.getSolution(spq)
        wpq_solution = subproblem.getSolution(wpq)
        vpq_solution = subproblem.getSolution(vpq)
        delta_solution = subproblem.getSolution(deltapq)
        gamma_solution = subproblem.getSolution(gammapq)
        multipliers = subproblem.getDuals(constraints)
        multipliers1 = subproblem.getDuals(constraints1)
        multipliers2 = subproblem.getDuals(constraints2)
        multipliers3 = subproblem.getDuals(constraints3)
        multipliers4 = subproblem.getDuals(constraints4)
        multipliers5 = subproblem.getDuals(constraints5)

    valori=(xp_solution, tpq_solution,spq_solution,wpq_solution,vpq_solution,delta_solution,gamma_solution)
    duali=(multipliers,multipliers1,multipliers2,multipliers3,multipliers4,multipliers5)
    return (optimum,valori,duali)


def cut_generation(duali,coordinates_p,Mp,M,ypq,zpq,theta,P,X):
    colind = []
    cutcoef = []
    rhs_constant = 0.0
    epsilon=1e-7
    multipliers4=duali[4]
    multipliers5=duali[5]
    multipliers3=duali[3]
    multipliers2=duali[2]
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
         rhs_constant -= multipliers4[j] * coordinates_p[p]['X']

    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        rhs_constant -= multipliers5[j] * coordinates_p[p]['Y']
    index_y=[]
    for j, (p, q) in enumerate((p, q) for p in P for q in X):
        if abs(multipliers2[j]) >= epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
            colind.append(ypq[p, q])
            cutcoef.append(multipliers2[j] * Mp[p])
            index_y.append([p,q])
        elif multipliers2[j] <= -epsilon:
            rhs_constant -= multipliers2[j] * Mp[p]
    index_z=[]
    for j, (p, q) in enumerate((p, q) for p in X for q in X if p < q):
        if abs(multipliers3[j]) >= epsilon:
            rhs_constant -= multipliers3[j] * M
            colind.append(zpq[p, q])
            cutcoef.append(multipliers3[j] * M)
            index_z.append([p, q])
        elif multipliers3[j] <= -epsilon:
            rhs_constant -= multipliers3[j] * M

    colind.append(theta)
    cutcoef.append(-1.0)
    rhs_final = -rhs_constant
    return(colind,cutcoef,rhs_final,index_y,index_z)


def node_cutoff_callback(problem, mindex,node):
    print("SONO IN NODE CUTOFF")
    print(f"LP obj: {problem.attributes.lpobjval}")
    print(f"MIP obj: {problem.attributes.mipobjval}")
    print(f"Best bound: {problem.attributes.bestbound}")
    print(f"Active nodes: {problem.attributes.activenodes}")
    print(f"LP status: {problem.attributes.lpstatus}")

def log_callback(problem, mindex, msg, msgtype):
    print(f"LOG: {msg}")
    print(f"Final incumbent: {problem.attributes.mipobjval}")
    print(f"Final bound: {problem.attributes.bestbound}")

def cb_intsol(problem,mindex):
    incumbent = problem.attributes.mipobjval
    current_bound = problem.attributes.lpobjval
    global_bound = problem.attributes.bestbound
    print("Sono in INTSOL")
    print(f"UB: {incumbent}")
    print(f"Current LB: {current_bound}")
    print(f"Global LB: {global_bound}")
    print(f"Gap: {incumbent - global_bound}")
    print(f"Gap relativo target: {problem.controls.miprelstop}")
    print(f"Gap assoluto target: {problem.controls.mipabsstop}")
    print(
        f"Gap corrente calcolato: {(problem.attributes.mipobjval - problem.attributes.bestbound) / abs(problem.attributes.mipobjval)}")


def nodeInfeasible(problem, object):
    print("SONO IN NODE INFEASIBLE")
    node = problem.attributes.currentnode
    print("Node {0} infeasible".format(node))


def plot_the_graph(coordinates_p,xp_solution,ypq_solution,zpq_solution,P,X):
    #Building the plots
    plt.figure(figsize=(8, 8))
    for p in P:
        plt.scatter(coordinates_p[p]['X'], coordinates_p[p]['Y'], color='blue', s=50,
                    label='Points P' if p == list(P)[0] else "")

    for p in X:
        plt.scatter(xp_solution[p]['X'], xp_solution[p]['Y'], color='red', s=50,
                    label='Points X' if p == list(X)[0] else "")

    for (p, q), val in ypq_solution.items():
        if val == 1.0:
            x1, y1 = coordinates_p[p]["X"], coordinates_p[p]["Y"]
            x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
            if x1 is not None and x2 is not None:
                plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1, label="ypq" if (p, q) == (0, 1) else "")

    for (p, q), val in zpq_solution.items():
        if val == 1:
            x1, y1 = xp_solution[p]["X"], xp_solution[p]["Y"]
            x2, y2 = xp_solution[q]["X"], xp_solution[q]["Y"]
            if x1 is not None and x2 is not None:
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label="zpq" if (p, q) == (0, 1) else "")

    plt.title("Points in the 2D space")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()



def solve():
    start_time = time.time()
    file_name= sys.argv[1]
    data = pd.read_csv(file_name, sep='\s+')
    data = data.drop(data.columns[0], axis=1)
    data = data.reset_index()
    data['id'] = data.index
    coordinates_p = data.set_index('id')[['X', 'Y']].T.to_dict()
    P = range(len(data))  # Set of given nodes
    num_steiner_nodes = len(data) - 2  # Number of Steiner Nodes
    X = range(num_steiner_nodes)  # Set of Steiner nodes
    xp.init('C:/xpressmp/bin/xpauth.xpr')
    Mp=[]
    ObjAbsAccuracy=0.00001
    distanza =0
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
    ypq = {(p, q): xp.var(vartype=xp.binary, ub=1, lb=0, name=f"y_{p}_{q}") for p in P for q in X}
    zpq = {(p, q): xp.var(vartype=xp.binary, ub=1, lb=0, name=f"z_{p}_{q}") for p in X for q in X if p < q}
    theta = xp.var(vartype=xp.continuous, name="theta")

    problem = xp.problem(name="Master problem")
    problem.addVariable([ypq[key] for key in ypq])
    problem.addVariable([zpq[key] for key in zpq])
    problem.addVariable(theta)

    # Objective function
    obj = (theta)
    problem.setObjective(obj, sense=xp.minimize)
    # constraints
    for p in P:
        problem.addConstraint(xp.Sum(ypq[p, q] for q in X) == 1)

    for q in X:
        problem.addConstraint(
            xp.Sum(ypq[p, q] for p in P)
            + xp.Sum(zpq[p, q] for p in X if p < q)
            + xp.Sum(zpq[q, p] for p in X if p > q) == 3
        )

    for q in X:
        if q > 0:
            problem.addConstraint(xp.Sum(zpq[p, q] for p in X if p < q) == 1)

    for q in X:
        problem.addConstraint(xp.Sum(ypq[p, q] for p in P) <= 2)

    g = Callbacks()
    mindex=[]
    problem.controls.cutfreq = 0
    problem.controls.miprelstop = 0.0001
    problem.controls.threads = 1
    (ypq_val,zpq_val,optimum_general)=MINLP_formulation(Mp,M,coordinates_p,X,P,d)
    data = (mindex, P, X, Mp,M,coordinates_p,ypq,zpq,theta,ypq_val,zpq_val,optimum_general)
    problem.setControl("presolve", 0)
    problem.setControl('CUTSTRATEGY', 0)
    problem.setControl("miprefineiterlimit", 0)
    problem.addcbpreintsol(g.cb_preintsol,data,0)
    #problem.addcbnodecutoff(node_cutoff_callback,mindex,0)
    #problem.addcbmessage(log_callback, mindex, 0)
    #problem.addcbintsol(cb_intsol, None,0)
    #problem.addcbnodecutoff(cb_nodecutoff,None,0)
    #problem.addcbinfnode(nodeInfeasible, None, 0)
    problem.addcboptnode(g.cb_optnode,data, 0)
    problem.solve()
    LB = problem.getObjVal()
    print("LB", LB)
    status = problem.getProbStatus()
    if status == xp.mip_optimal:
        ypq_solution=problem.getSolution(ypq)
        zpq_solution = problem.getSolution(zpq)

        print("Optimal solution found")
        print("ypq:", ypq_solution)
        print("zpq:", zpq_solution)
    else:
        print(f"Error status: {status}")
    #Now we need to extract the values of the subproblem
    (optimum,valori,duali)=subproblem(Mp,M,coordinates_p,ypq_solution,zpq_solution,X, P)
    xp_solution = valori[0]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
    plot_the_graph(coordinates_p,xp_solution,ypq_solution,zpq_solution,P,X)


if __name__ == '__main__':
   solve()