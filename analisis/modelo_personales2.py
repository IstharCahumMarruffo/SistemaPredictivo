from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from limpieza import cargar_datos_personales

def entrenar_modelo_personal():
    # Cargar los datos
    df = cargar_datos_personales()
    if df is None:
        print("No se pudieron cargar los datos.")
        return

    # Definir las variables personales y la variable objetivo
    variables_personales = [
       's1','p12_1','p12_2','p12_3','p12_4','p12_5','p12_6','p12_7','p12_8','p12_9','p12_10','p13_4','p13_5','p13_6','p13_7',
       'p23_1','p41a','p41b','p41c','p41d','p41e','p41f','p41g','p41h','p41i','p24_10','p24_12','p24_13','p24_14',
        'p24_18','p24_22','p24_17','p24_11','s9p','s9m'
    ]

    # Filtrar el DataFrame
    df_personal = df[variables_personales + ['f21']].copy()
    df_personal = df_personal.dropna()

    # Crear la columna 'estado' para la variable objetivo
    df_personal["estado"] = df_personal["f21"].map({1: 1, 2: 0})

    # Definir X y y
    X = df_personal[variables_personales]
    y = df_personal["estado"]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE en el conjunto de entrenamiento
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Inicializar el modelo
    modelo = DecisionTreeClassifier(random_state=42)

    # Realizar validación cruzada en el conjunto de entrenamiento
    scores = cross_val_score(modelo, X_train_res, y_train_res, cv=5, scoring='accuracy')
    print("=== Resultados de la validación cruzada modelo personal ===")
    print(f"Precisión en cada pliegue: {scores}")
    print(f"Precisión media: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")

    # Entrenar el modelo en el conjunto de entrenamiento
    modelo.fit(X_train_res, y_train_res)
   
    # Predecir en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    print("\n=== Evaluación del modelo personal ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # Guardar el modelo entrenado
    joblib.dump(modelo, 'modelo_personal.pkl')
    
    return modelo

# Llamar a la función para entrenar el modelo
#entrenar_modelo_personal()
