import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from limpieza import cargar_datos_generales

def entrenar_modelo_general():
    df = cargar_datos_generales()
    if df is None:
        print("No se pudieron cargar los datos.")
        return
    # Definir las características (X) y l variable objetivo (y)
    X = df[['f8e_1', 's1', 'p15']]  # Variables predictoras
    df["estado"] = df["f21"].map({1: 1, 2: 0})
    y = df["estado"]  # Variable objetivo (0 = no desertó, 1 = desertó)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de regresión logística
    modelo_g = LogisticRegression()
    modelo_g.fit(X_train, y_train)

    # Hacer predicciones y evaluar el modelo
    y_pred = modelo_g.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy del modelo: {accuracy:.2f}')


# Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(cm)

# Reporte de clasificación
    report = classification_report(y_test, y_pred)
    print("Reporte de Clasificación:")
    print(report)
    joblib.dump(modelo_g, 'modelo_general.pkl')
    return modelo_g

entrenar_modelo_general()