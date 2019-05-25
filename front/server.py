# importamos Flask
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response, abort

# Creamos el objeto app para enrutar las llamadas
app = Flask(__name__)

# Creamos función para manejar los datos
def handle_data(variables, restrictions):
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

# Añadimos endpoint
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
        data = handle_data(variables, restrictions)
        if data[0] == 200:
            #bien pepe bien
            print(data[1])
        elif data[0] == 400:
            error = data[1]
        else:
            error = 'Uknown error'
    return render_template('home.html', error=error)

# API
@app.route('/api/post', methods=['POST'])
def api_post():
    # comprobamos si es un json
    if not request.json:
        return jsonify(
                status='ERROR',
                error='Bad request type'
            ), 400
    # handleamos data
    data = handle_data(request.json['variables'], request.json['restrictions'])
    if data[0] == 200:
        return jsonify(
                status='OK',
                data=data[1]
            ), 200
    elif data[0] == 400:
        return jsonify(
                status='ERROR',
                error=data[1]
            ), 400
    else:
        return jsonify(
                status='ERROR',
                error='Uknown error'
            ), 400

# Iniciamos el servidor
if __name__ == '__main__':
    app.run(host ='127.0.0.1', port = 3333, debug = True)