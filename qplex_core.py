import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx

from qiskit import BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.aqua import Operator, run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import max_cut, tsp
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
# set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

from docplex.mp.model import Model
from qiskit.aqua.translators.ising import docplex


def createMatrixRestrictionDefaultOrig(qbits_code, input_vars, input_rest, beta):
    num_vars = len(input_vars)
    tamMatrix = num_vars * qbits_code  # matrix is num_vars * qbits used for each var
    p = 1000  # very high number
    matrix = np.zeros([tamMatrix, tamMatrix])
    for row in range(tamMatrix):
        for col in range(row, tamMatrix):
            dom_row = row // qbits_code
            dom_col = col // qbits_code
            if row == col:
                matrix[row, col] = -input_vars[dom_row] * 2 ** (row % qbits_code)
            else:
                if dom_row == dom_col:  # Si estamos en un X0 X1 o X0 X2... no hay relación, se pone 0
                    matrix[row, col] = 0
                    matrix[col, row] = 0
                else:
                    r = p / beta * (input_rest[dom_row] * input_rest[dom_col])
                    matrix[row, col] = r
                    matrix[col, row] = r
    return matrix

def wrapper(qbits_encode, input_vars, options):

    if options['restriction'] == 'default':
        matrix = createMatrixRestrictionDefault(qbits_encode, input_vars, options['input_rest'],  options['beta'])
        return matrix

def createMatrixSinRestrictionDefault(qbits_code, input_vars, input_rest, beta):
    num_vars = len(input_vars)
    tamMatrix = num_vars * qbits_code  # matrix is num_vars * qbits used for each var
    p = 1000  # very high number
    matrix = np.zeros([tamMatrix, tamMatrix])
    for row in range(tamMatrix):
        for col in range(row, tamMatrix):
            dom_row = row // qbits_code
            dom_col = col // qbits_code
            if row == col:
                matrix[row, col] = 1#-input_vars[dom_row] * 2 ** (row % qbits_code)
            else:
                if dom_row == dom_col:  # Si estamos en un X0 X1 o X0 X2... no hay relación, se pone 0
                    matrix[row, col] = 0
                    matrix[col, row] = 0
                else:
                    r = -p / beta * (input_rest[dom_row] * input_rest[dom_col])
                    matrix[row, col] = r
                    matrix[col, row] = r
    #print(matrix)
    for row in range(tamMatrix):
        acu = 0.
        for col in range(row+1, tamMatrix):
            acu += matrix[row, col]
        matrix[row, row] = (matrix[row, row]/2) + (acu/4)
    return matrix




def createMatrixRestrictionDefaultOrig2(qbits_code, input_vars, input_rest, beta):
    num_vars = len(input_vars)
    tamMatrix = num_vars * qbits_code  # matrix is num_vars * qbits used for each var
    p = 1000  # very high number
    matrix = np.zeros([tamMatrix, tamMatrix])
    for row in range(tamMatrix):
        for col in range(row, tamMatrix):
            dom_row = row // qbits_code
            dom_col = col // qbits_code
            if row == col:
                matrix[row, col] = input_vars[dom_row] * 2 ** (row % qbits_code)
            else:
                if dom_row == dom_col:  # Si estamos en un X0 X1 o X0 X2... no hay relación, se pone 0
                    matrix[row, col] = 0
                    matrix[col, row] = 0
                else:
                    r = p / beta * (input_rest[dom_row] * input_rest[dom_col])
                    matrix[row, col] = r
                    matrix[col, row] = r
    for row in range(tamMatrix):
        acu = 0.
        for col in range(row+1, tamMatrix):
            acu += matrix[row, col]
        matrix[row, row] = (matrix[row, row]/2) + (acu/4)
    return matrix

def createMatrixRestrictionDefault(qbits_code, input_vars, input_rest, beta):
    num_vars = len(input_vars)
    tamMatrix = num_vars * qbits_code  # matrix is num_vars * qbits used for each var
    p = 1000  # very high number
    matrix = np.zeros([tamMatrix, tamMatrix])
    for row in range(tamMatrix):
        for col in range(row, tamMatrix):
            dom_row = row // qbits_code
            dom_col = col // qbits_code
            if row == col:
                matrix[row, col] = -input_vars[dom_row] * 2 ** (row % qbits_code)
            else:
                if dom_row == dom_col:  # Si estamos en un X0 X1 o X0 X2... no hay relación, se pone 0
                    matrix[row, col] = 0
                    matrix[col, row] = 0
                else:
                    r = -p / beta * (input_rest[dom_row] * input_rest[dom_col])
                    matrix[row, col] = -r
                    matrix[col, row] = -r
    #print(matrix)
    for row in range(tamMatrix):
        acu = 0.
        for col in range(row+1, tamMatrix):
            acu += matrix[row, col]
        matrix[row, row] = (matrix[row, row]/2) + (acu/4)
    return matrix

def createMatrixRestrictionBackup1(qbits_code, input_vars, input_rest, beta):
    num_vars = len(input_vars)
    tamMatrix = num_vars * qbits_code  # matrix is num_vars * qbits used for each var
    p = 1  # very high number
    matrix = np.zeros([tamMatrix, tamMatrix])
    for row in range(tamMatrix):
        for col in range(row, tamMatrix):
            dom_row = row // qbits_code
            dom_col = col // qbits_code
            if row == col:
                matrix[row, col] = -input_vars[dom_row] * 2 ** (row % qbits_code)
            else:
                if dom_row == dom_col:  # Si estamos en un X0 X1 o X0 X2... no hay relación, se pone 0
                    matrix[row, col] = 0
                    matrix[col, row] = 0
                else:
                    r = -p / beta * (input_rest[dom_row] * input_rest[dom_col])
                    matrix[row, col] = -r
                    matrix[col, row] = -r
    #print(matrix)
    for row in range(tamMatrix):
        acu = 0.
        for col in range(row+1, tamMatrix):
            acu += matrix[row, col]
        matrix[row, row] = (matrix[row, row]/2) + (acu/4)
    return matrix


def getPauliMatrix(matrix):
    rows = matrix.shape[0]
    paulis = []
    for row_pos in range(rows):
        for col_pos in range(matrix.shape[1]):
            temp = {}
            temp["imag"] = 0.0
            temp["real"] = matrix[row_pos, col_pos]
            label_pauli = ["I" for _ in range(rows)]
            if row_pos != col_pos:
                label_pauli[row_pos] = 'Z'
                label_pauli[col_pos] = 'Z'
            label_pauli = "".join(label_pauli)
            paulis.append({"coeff": temp, "label": label_pauli})
    paulis_dict = {"paulis": paulis}
    for i in paulis_dict["paulis"]:
        print(i)
    pauli_matrix = Operator.load_from_dict(paulis_dict)
    return pauli_matrix


def dameInversoBinario(target, precision, num_vars):
    contador = {}
    for i in range(precision):
        contador[i] = 0
    while target > 0:
        for i in range(precision-1, -1, -1):
            while target >= 2**i:
                target -= 2**i
                contador[i] += 1

    print(contador)
    #Invertir bits
    long_buscada = num_vars * precision
    array = np.zeros(long_buscada)
    for i in range(precision):
        for j in range(contador[i]):
            array[j*num_vars + (i)] = 1

    invert_array = ['0' if x == 1 else '1' for x in array]

    arrays = np.array_split(np.array(invert_array), 3)
    acu = 0
    for i in arrays:
        en_binario = "".join(list(np.flip(i)))
        acu += int(en_binario, 2)
    return acu


def optimize_f(precision, coefs_param, beta):

    coefs = coefs_param.copy()
    coefs.append(0)
    coefs_restr = (1, 1, 1)

    lista_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    mdl = Model()
    variables = []
    for num_var in range(len(coefs)):
        tmp = {i: mdl.binary_var(name=(lista_vars[num_var] + '_{0}').format(i)) for i in range(precision)}
        variables.append(tmp)
    # x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(precision)}
    # y = {i: mdl.binary_var(name='y_{0}'.format(i)) for i in range(precision)}
    # z = {i: mdl.binary_var(name='z_{0}'.format(i)) for i in range(precision)}
    #print(variables)
    # Object function
    # my_func = mdl.sum(coefs[0]*(2**i)*x[i]+coefs[1]*(2**i)*y[i]+(2**i)*coefs[2]*z[i] for i in range(precision))

    # my_func = mdl.sum(coefs[j]*(2**i)*vars[j][i] for j in range(len(coefs)) for i in range(precision))

    my_func = mdl.sum(coefs[j] * (2 ** i) * variables[j][i] for j in range(len(coefs)) for i in range(precision))

    # (x[i] for i in range(precision)), (y[i] for i in range(precision)), (z[i] for i in range(precision)
    # tmp = {0:{'var':x,'coef':coefs_restr[0]},
    #       1:{'var':y,'coef':coefs_restr[1]},
    #       2:{'var':z,'coef':coefs_restr[2]}}

    mdl.maximize(my_func)

    inverted_beta = dameInversoBinario(beta, precision, len(coefs))

    # mdl.add_constraint(mdl.sum( tmp[v]['var'][i]*(2**i)*tmp[v]['coef'] for v in range(len(coefs)) for i in range(precision)) == beta)
    # mdl.add_constraint(mdl.sum( x[0] + x[1]*(2) + x[2]*(4) + y[0] + y[1]*(2) + y[2]*(4) + z[0] + z[1]*(2) + z[2]*(4) ) == inverted_beta)
    # mdl.add_constraint(mdl.sum( variables[0][0] + variables[0][1]*(2) + variables[0][2]*(4) + variables[1][0] + variables[1][1]
    # *(2) + variables[1][2]*(4) + variables[2][0] + variables[2][1]*(2) + variables[2][2]*(4) ) == inverted_beta)

    mdl.add_constraint(
        mdl.sum(variables[v][i] * 2 ** i for v in range(len(coefs)) for i in range(precision)) == inverted_beta)

    # mdl.add_constraint(mdl.sum( -x[0] - x[1]*(2) - x[2]*(4) - y[0] - y[1]*(2) - y[2]*(4) - z[0] - z[1]*(2) - z[2]*(4) ) == 6)

    # mdl.add_constraint(mdl.sum( -1*tmp[v]['var'][i]*(2**i)*tmp[v]['coef'] for v in range(len(coefs)) for i in range(precision)) == -beta)

    qubitOp_docplex, offset_docplex = docplex.get_qubitops(mdl)

    #print(qubitOp_docplex)

    # algo_input = EnergyInput(qubitOp_docplex)
    # print(algo_input.)

    # ee = VQE(qubitOp_docplex)
    # ee.run()
    ee = ExactEigensolver(qubitOp_docplex, k=1)
    result_ee = ee.run()
    x_ee = max_cut.sample_most_likely(result_ee['eigvecs'][0])
    print('solution:', max_cut.get_graph_solution(x_ee))
    solucion_ee = max_cut.get_graph_solution(x_ee)
    return (solucion_ee, None)
    """
    algorithm_cfg = {
        'name': 'ExactEigensolver',
    }

    params = {
        'problem': {'name': 'ising'},
        'algorithm': algorithm_cfg
    }
    result = run_algorithm(params,algo_input)
    """
    # x = max_cut.sample_most_likely(result['eigvecs'][0])
    # print('energy:', result['energy'])
    # print('max-cut objective:', result['energy'] + offset_docplex)
    # print('solution:', max_cut.get_graph_solution(x))
    # print('solution objective:', max_cut.max_cut_value(x, w))

    seed = 10598

    # change optimizer(spsa), change ry (riyc)
    spsa = SPSA(max_trials=300)
    ry = RY(qubitOp_docplex.num_qubits, depth=6, entanglement='linear')
    vqe = VQE(qubitOp_docplex, ry, spsa, 'matrix')

    backend = BasicAer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend, seed=seed, seed_transpiler=seed)

    result = vqe.run(quantum_instance)
    x = max_cut.sample_most_likely(result['eigvecs'][0])
    print('solution:', max_cut.get_graph_solution(x))
    return (solucion_ee, max_cut.get_graph_solution(x))

def wrapper_optimiza_f(precision, coefs_param, beta):
    r = optimize_f(precision, coefs_param, beta)
    r_ee = r[0]
    vars = np.array_split(r_ee, 3)
    tam = len(vars)
    answer = {}
    for i in range(tam):
        curr = vars[i]
        curr_i = np.flip(curr)
        curr_i = [str(int(k)) for k in curr_i]
        en_binario = "".join(list(curr_i))
        curr_var = int(en_binario, 2)
        if i != tam - 1:
            answer[i]=curr_var
    return answer

if __name__ == '__main__':
    precision = 6
    coefs_param = [2, -3]
    beta = 7
    a = wrapper_optimiza_f(precision, coefs_param, beta)
    print(a)
    exit()
    dameInversoBinario(6, 3, 3)
    exit()
    qbits_encode = 2  # Max number of qubits for coding a number
    input_vars = [2, -3]  # 2*x + 4*y +6*z

    # basic restriction
    options = {}
    options['restriction'] = 'default'  # a*x+b*y+c*z <= beta (que puede ser 1)
    options['input_rest'] = [1, 1]
    options['beta'] = 2
    matrix = wrapper(qbits_encode, input_vars, options)
    print(matrix)
    pauli_matrix = getPauliMatrix(matrix)
    print(pauli_matrix)
    from docplex.mp.model import Model
    mdl = Model()
    mdl.add_constraint