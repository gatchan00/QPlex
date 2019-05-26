# importamos Flask
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response, abort
# importamos qplex-core
import sys
sys.path.append('../')
from qplex_core import *

# Creamos el objeto app para enrutar las llamadas
app = Flask(__name__)

# Creamos función para manejar los datos V2
def handle_dataV2(variables, restriction, precision):
    # check variables
    for i in variables:
        if isinstance(i, int) != True:
            return [400, 'Bad variable value']
    # check restriction
    if isinstance(restriction, int) != True:
        return [400, 'Bad restriction value']
    # check precision
    if isinstance(precision, int) != True or precision > 6:
        return [400, 'Bad precision value']
    # output bueno
    return [200, [variables, restriction, precision]]

# API-REST
@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    # comprobamos si es un json
    if not request.json:
        return jsonify(
                status='ERROR',
                message='Bad request type'
            ), 400
    data = handle_dataV2(request.json['variables'], request.json['restriction'], request.json['precision'])
    if data[0] == 200:
        # function: 2*X-3*Y
        # restriction: X*y<=restriction
        output = wrapper_optimiza_f(data[1][2], data[1][0], data[1][1])
        output = {
            'x': output[0],
            'y': output[1]
        }
        return jsonify(
                status='OK',
                results=output
            ), 200
    elif data[0] == 400:
        return jsonify(
                status='ERROR',
                message=data[1]
            ), 400
    else:
        return jsonify(
                status='ERROR',
                message='Uknown error'
            ), 400

# Creamos función para manejar los datos V1
def handle_dataV1(variables, restrictions):
    # check variables
    for i in variables:
        if i[1] != 'int':
            return [400, 'Bad variable type']
    # check restrictions
    for r in restrictions:
        if not isinstance(r[0], str) or not isinstance(r[1], str):
            return [400, 'Bad restrinctions value']
    # output bueno
    return [200, [variables, restrictions]]

# Web front-end
@app.route('/', methods=['GET', 'POST'])
def home():
    error = None
    if request.method == 'POST':
        # comprobamos si quieren int o real
        print(bool(request.form['optionA']))
        print(bool(request.form['optionB']))
        print(bool(request.form['optionC']))
        print(bool(request.form['optionD']))
        # metemos variables "y demás" en array
        variables = [
            ["a", "int", request.form['a']],
            ["b", "int", request.form['b']],
            ["c", "int", request.form['c']],
            ["d", "int", request.form['d']]
        ]
        # metemos restrictions en array
        restrictions = [request.form.getlist('rest1[]'), request.form.getlist('rest2[]')]
        data = handle_dataV1(variables, restrictions)
        if data[0] == 200:
            #bien pepe bien
            print(data[1])
        elif data[0] == 400:
            error = data[1]
        else:
            error = 'Uknown error'
    return render_template('home.html', error=error)

# Iniciamos el servidor
if __name__ == '__main__':
    app.run(host ='127.0.0.1', port = 3333, debug = True)