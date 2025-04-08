# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from limpieza import cargar_datos_generales

plt.rcParams.update({
    'font.size': 12,          # Cambiar tamaño de fuente
    'axes.labelsize': 12,     # Cambiar tamaño de las etiquetas de los ejes
    'xtick.labelsize': 10,    # Cambiar tamaño de las etiquetas del eje X
    'ytick.labelsize': 10,    # Cambiar tamaño de las etiquetas del eje Y
    'axes.facecolor': '#f0f0f0',  # Color de fondo de los ejes
    'figure.facecolor': '#e0e0e0' # Color de fondo de la figura
})

def generar_estadisticas():
    df = cargar_datos_generales()

    df['f21'] = df['f21'].map({1: 1, 2: 0})  # 1: desertó, 0: concluyó
    df['s1'] = df['s1'].map({1: 'Hombre', 2: 'Mujer'})
    df['p18'] = df['p18'].map({1: 'Si', 2: 'No'})
    df['p24_1'] = df['p24_1'].map({1: 'Si', 2: 'No'})

    df['Rango_Edad'] = pd.cut(df['f8e_1'], bins=[0, 17, 20, 23, 30],
                              labels=['≤17', '18-20', '21-23', '24-30'])

    df['Rango_Promedio'] = pd.cut(df['p15'], bins=[0, 6, 7, 8, 9, 10],
                                   labels=['<6', '6-7', '7-8', '8-9', '9-10'])

    riesgo_genero = df.groupby('s1')['f21'].mean()
    riesgo_edad = df.groupby('Rango_Edad')['f21'].mean()
    riesgo_beca = df.groupby('p18')['f21'].mean()
    riesgo_dinero = df.groupby('p24_1')['f21'].mean()
    riesgo_promedio = df.groupby('Rango_Promedio')['f21'].mean()

    def plot_to_base64(serie, title, color, xlabel):
        fig, ax = plt.subplots(figsize=(7, 7))
        serie.plot(kind='bar', color=color, ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Probabilidad de Deserción", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)

        return base64.b64encode(image_png).decode('utf-8')
    
    

    grafico_genero = plot_to_base64(riesgo_genero, 'Por Género', 'indigo', 'Género')
    grafico_edad = plot_to_base64(riesgo_edad, 'Por Edad', 'magenta', 'Edad')
    grafico_promedio = plot_to_base64(riesgo_promedio, 'Por Promedio', 'aqua', 'Promedio')
    grafico_beca = plot_to_base64(riesgo_beca, '¿Tiene beca?', 'pink', '')
    grafico_dinero = plot_to_base64(riesgo_dinero, '¿Falta dinero?', 'violet', '')
    resumen = df.groupby(['s1', 'Rango_Edad', 'Rango_Promedio'])['f21'].mean().reset_index()
    resumen.rename(columns={'s1': 'Sexo', 'f21': 'Deserto'}, inplace=True)

    return grafico_genero, grafico_edad, grafico_promedio, grafico_beca, grafico_dinero
