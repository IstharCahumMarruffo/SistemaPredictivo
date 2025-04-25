from flask import Flask, render_template, request, jsonify, send_file
import random
import traceback
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
from models import db, DatosConcluidos
import pandas as pd 
from analisis.modelo_academico import entrenar_modelo_academico
from analisis.modelo_personales import entrenar_modelo_personal
from analisis.modelo_general import entrenar_modelo_general
from analisis.estadisticas_generales import generar_estadisticas
from analisis.modelo_economico import entrenar_modelo_economico
from modelos.evaluar_modelos import evaluar_modelos
from flask import Flask, render_template, request, redirect, url_for
from weasyprint import HTML
import tempfile

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:password@localhost/desercion_escolar'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

modelo_g = entrenar_modelo_general()
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'f8e_1': request.form.get('f8e_1'),
            's1': request.form.get('s1'),
            'p15': request.form.get('p15'),
            'p24_1': request.form.get('p24_1')
        }

        df_data = pd.DataFrame([data])

        df_data = df_data[['f8e_1', 's1', 'p15', 'p24_1']] 

        prediccion = modelo_g.predict(df_data)

        resultado = "⚠️ El estudiante tiene alto riesgo de deserción." if prediccion[0] == 1 else "✅Sin riesgo de deserción"
        return redirect(url_for('resultadoG', resultado=resultado))

    return render_template('index.html', resultado=None)
    
    
modelo = entrenar_modelo_academico()
@app.route('/academico', methods=['GET', 'POST'])
def academico():
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
       
        prediction_proba = modelo.predict_proba(input_data)
        probabilidad = prediction_proba[0][1]
        
        porcentaje = probabilidad * 100
        
        if probabilidad > 0.7:
            result = f"⚠️ ⚠️ Riesgo muy alto de deserción⚠️ ⚠️\n {porcentaje:.2f}%"
        elif probabilidad > 0.5:
            result = f"⚠️ Riesgo moderado de deserción ⚠️\n{porcentaje:.2f}%"
        else:
            result = f"✅ Riesgo bajo de deserción✅\n {porcentaje:.2f}%"   

        
        print("Probabilidad de deserción (clase 1):", prediction_proba[0][1])
        print("Resultado:", result)
        
        return redirect(url_for('resultado', resultado=result))

    return render_template('gestion.html')


modeloP = entrenar_modelo_personal()
@app.route('/personal', methods=['GET', 'POST'])
def personal():
    if request.method == 'POST':
        data = {
            's1': request.form.get('s1'),
            'p12_1': request.form.get('p12_1', 0),
            'p12_2': request.form.get('p12_2', 0),
            'p12_3': request.form.get('p12_3', 0),
            'p12_4': request.form.get('p12_4', 0),
            'p12_5': request.form.get('p12_5', 0),
            'p12_6': request.form.get('p12_6', 0),
            'p12_7': request.form.get('p12_7', 0),
            'p12_8': request.form.get('p12_8', 0),
            'p12_9': request.form.get('p12_9', 0),
            'p12_10': request.form.get('p12_10', 0),
            'p13_4': request.form.get('p13_4'),
            'p13_5': request.form.get('p13_5'),
            'p13_6': request.form.get('p13_6'),
            'p13_7': request.form.get('p13_7'),
            'p23_1': request.form.get('p23_1'),
            'p41a': request.form.get('p41a'),
            'p41b': request.form.get('p41b'),
            'p41c': request.form.get('p41c'),
            'p41d': request.form.get('p41d'),
            'p41e': request.form.get('p41e'),
            'p41f': request.form.get('p41f'),
            'p41g': request.form.get('p41g'),
            'p41h': request.form.get('p41h'),
            'p41i': request.form.get('p41i'),
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

       
        prediction_proba = modeloP.predict_proba(input_data)
        probabilidad = prediction_proba[0][1]
        porcentaje = probabilidad * 100
       
        if probabilidad > 0.7:
            result = f"⚠️ ⚠️ Riesgo P muy alto de deserción⚠️ ⚠️\n {porcentaje:.2f}%"
        elif probabilidad > 0.5:
            result = f"⚠️ Riesgo P moderado de deserción ⚠️\n{porcentaje:.2f}%"
        else:
            result = f"✅ Riesgo P bajo de deserción✅\n {porcentaje:.2f}%"  
        
        print("Probabilidad de deserción (clase 1):", prediction_proba[0][1])
        print("Resultado:", result)
        
        return redirect(url_for('resultado', resultado=result))

    return render_template('gestion.html')

modeloE = entrenar_modelo_economico()
@app.route('/economico', methods=['GET', 'POST'])
def economico():
    if request.method == 'POST':
        # Función para manejar la conversión y valores predeterminados
        def safe_float(value, default=-1):
            try:
                return float(value) if value != '' else default
            except ValueError:
                return default

        # Obtener los datos del formulario
        data = {
            'p27': request.form.get('p27'),
            'p29': safe_float(request.form.get('p29')),
            'p30': safe_float(request.form.get('p30')),
            'p31': safe_float(request.form.get('p31')),
            'p24_6': safe_float(request.form.get('p24_6')),
            'p24_1': safe_float(request.form.get('p24_1')),
        
        }

        # Si la respuesta en p27 es "No", asignar -1 a las otras preguntas
        if data['p27'] == '2':
            data['p29'] = -1
            data['p30'] = -1
            data['p31'] = -1

        # Imprimir los datos para depuración
        print("Datos recibidos:", data)

        # Crear un DataFrame de pandas para usarlo con el modelo
        input_data = pd.DataFrame([data])

        # Predecir la probabilidad de deserción
        prediction_proba = modeloE.predict_proba(input_data)
        
        probabilidad = prediction_proba[0][1]
        
        porcentaje = probabilidad * 100
        
        if probabilidad > 0.7:
            result = f"⚠️ ⚠️ Riesgo E muy alto de deserción⚠️ ⚠️\n {porcentaje:.2f}%"
        elif probabilidad > 0.5:
            result = f"⚠️ Riesgo E moderado de deserción ⚠️\n{porcentaje:.2f}%"
        else:
            result = f"✅ Riesgo E bajo de deserción✅\n {porcentaje:.2f}%"  
        
        
        print("Probabilidad de deserción (clase 1):", prediction_proba[0][1])
        print("Resultado:", result)

        return redirect(url_for('resultado', resultado=result))

    return render_template('gestion.html')

@app.route('/resultado')
def resultado():
    resultado = request.args.get('resultado', 'No hay resultado disponible.')
    
    return render_template('resultado.html', resultado=resultado)


@app.route('/resultadoG')
def resultadoG():
    resultado = request.args.get('resultado', 'No hay resultado disponible.')
    
    return render_template('resultadoG.html', resultado=resultado)

@app.route('/formulario_academicos.html')
def formulario():
    return render_template('formulario_academicos.html')

@app.route('/formulario_personales.html')
def formularioP():
    return render_template('formulario_personales.html')

@app.route('/formulario_economicos.html')
def formularioE():
    return render_template('formulario_economicos.html')

@app.route('/archivos.html')
def archivosSu():
    return render_template('archivos.html')


@app.route('/i1')
def i1():
    return render_template('i1.html')


@app.route('/gestion.html')
def gestion():
    return render_template('gestion.html')

@app.route('/estadisticas')
def estadisticas():
    return render_template('estadisticas.html')

@app.route('/estadisticas/<tipo>')
def grafico(tipo):
    grafico_genero, grafico_edad, grafico_promedio, grafico_beca, grafico_dinero= generar_estadisticas()
    imagenes = {
        'genero': grafico_genero,
        'edad': grafico_edad,
        'promedio': grafico_promedio,
        'beca': grafico_beca,
        'dinero': grafico_dinero
    }
    if tipo not in imagenes:
        return "Gráfico no encontrado", 404
    return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
            <title>Gráfico {tipo}</title>
        </head>
        <body>
            <img src="data:image/png;base64,{imagenes[tipo]}" alt="Gráfico {tipo}">
        </body>
        </html>
    '''




@app.route('/cargar_archivo', methods=['POST'])
def cargar_archivo():
    archivo = request.files['archivo']
    if not archivo:
        return "No se subió ningún archivo", 400

    df = pd.read_csv(archivo)
    resultados = evaluar_modelos(df)

    return render_template('resultadoA.html', resultados=resultados)

@app.route('/descargar_pdf', methods=['POST'])
def descargar_pdf():
    try:
        resultados = request.get_json().get('resultados')
        if not resultados:
            return "No se proporcionaron resultados", 400

        html_str = render_template('pdf.html', resultados=resultados)
        print(html_str)

        pdf_file = BytesIO()
        HTML(string=html_str).write_pdf(pdf_file)
        pdf_file.seek(0)

        return send_file(
            pdf_file, 
            as_attachment=True, 
            download_name='predicciones.pdf', 
            mimetype='application/pdf'
        )
    except Exception as e:
        traceback.print_exc()
        return f"Ocurrió un error al generar el PDF: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
