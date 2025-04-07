from flask import Flask, render_template, request, jsonify, redirect, url_for
import random
import pandas as pd
from rendimiento import grafico
from analisis.modelo_academico3 import entrenar_modelo_academico
from models import db, DatosConcluidos
from flask_sqlalchemy import SQLAlchemy
from analisis.modelo_personales import entrenar_modelo_personal

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:password@localhost/desercion_escolar'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Ruta principal con gráfico y predicción
@app.route('/formulario_academicos', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'p2a': request.form.get('p2a'),
            'p4': request.form.get('p4'),
            'p5': request.form.get('p5'),
            'p6': request.form.get('p6'),
            'p7': request.form.get('p7'),
            'p10h': request.form.get('p10h'),
            'p10m': request.form.get('p10m'),
            'p11_1': request.form.get('p11_1'),
            'p14': request.form.get('p14'),
            'p15': request.form.get('p15'),
            'p16': request.form.get('p16'),
            'p17': request.form.get('p17'),
            'p18': request.form.get('p18'),
            'p13_1': request.form.get('p13_1'),
            'p13_2': request.form.get('p13_2'),
            'p13_3': request.form.get('p13_3'),
            'p24_7': request.form.get('p24_7'),
            'p24_8': request.form.get('p24_8')
        }
        print("Datos recibidos:", data)
        
        input_data = pd.DataFrame([data])
        modelo = entrenar_modelo_academico()
        prediction_proba = modelo.predict_proba(input_data)

        if prediction_proba[0][1] > 0.1:
            result = "El estudiante tiene alto riesgo de deserción."
        else:
            result = "El estudiante no tiene alto riesgo de deserción."
        
        print("Probabilidad de deserción (clase 1):", prediction_proba[0][1])
        print("Resultado:", result)
        
        return redirect(url_for('resultado', resultado=result))

    return render_template('gestion.html')

@app.route('/formulario_personales', methods=['GET', 'POST'])
def formulario_personales():
    if request.method == 'POST':
        data = {
            's1': request.form.get('p2a'),
            'p12_1': request.form.get('p4'),
            'p12_2': request.form.get('p5'),
            'p12_3': request.form.get('p6'),
            'p12_4': request.form.get('p7'),
            'p12_5': request.form.get('p10h'),
            'p12_6': request.form.get('p10m'),
            'p12_7': request.form.get('p11_1'),
            'p12_8': request.form.get('p14'),
            'p12_9': request.form.get('p15'),
            'p12_10': request.form.get('p16'),
            'p13_4': request.form.get('p17'),
            'p13_5': request.form.get('p18'),
            'p13_6': request.form.get('p13_1'),
            'p13_7': request.form.get('p13_2'),
            'p23_1': request.form.get('p13_3'),
            'p41a': request.form.get('p41a'),
            'p41b': request.form.get('p41b'),
            'p41c': request.form.get('p41c'),
            'p41d': request.form.get('p41d'),
            'p41e': request.form.get('p41e'),
            'p41f': request.form.get('p41f'),
            'p41g': request.form.get('p41g'),
            'p41h': request.form.get('p41h'),
            'p41i': request.form.get('p41i'),
            'p24_1': request.form.get('p24_1'),
            'p24_10': request.form.get('p24_10'),
            'p24_12': request.form.get('p24_12'),
            'p24_13': request.form.get('p24_13'),
            'p24_14': request.form.get('p24_14'),
            'p24_18': request.form.get('p24_18'),
            'p24_22': request.form.get('p24_22'),
            'p24_17': request.form.get('p24_17'),
            'p24_11': request.form.get('p24_11'),
            's9p': request.form.get('s9p'),
            's9m': request.form.get('s9m')
        }   
        print("Datos recibidos:", data)
        
        input_data = pd.DataFrame([data])
        modelo = entrenar_modelo_personal() 
        prediction_proba = modelo.predict_proba(input_data)

        if prediction_proba[0][1] > 0.1:
            result = "El estudiante tiene alto riesgo de deserción."
        else:
            result = "El estudiante no tiene alto riesgo de deserción."
        
        print("Probabilidad de deserción (clase 1):", prediction_proba[0][1])
        print("Resultado:", result)
        
        return redirect(url_for('resultado', resultado=result))

    return render_template('gestion.html')


# Página de resultado
@app.route('/resultado')
def resultado():
    resultado = request.args.get('resultado', 'No hay resultado disponible.')
    return render_template('resultado.html', resultado=resultado)



# Endpoint de predicción dummy
@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.form
    riesgo = random.uniform(0, 1)  
    return jsonify({'riesgo': round(riesgo * 100, 2)})


# Página de gestión (vista inicial)
@app.route('/gestion.html')
def gestion():
    return render_template('gestion.html')


# Prueba de consulta a la base de datos
@app.route('/prueba', methods=['GET'])
def prueba():
    datos = DatosConcluidos.query.all()
    return jsonify([{"edo": d.edo, "muni": d.muni} for d in datos])


if __name__ == '__main__':
    app.run(debug=True)
