"""from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
from limpieza import cargar_datos_economicos


def entrenar_modelo_economico():
    df = cargar_datos_economicos

    if df is None:
            print("No se pudieron cargar los datos")
            return
    
    variables_economicas = ['p27','p29','p30','p31','p24_6', 'p24_1']

    df_economicos = df[variables_economicas+['f21', 'estado']].copy()

    X = df_economicos[variables_economicas]
    y=df_economicos["estado"]

    modelo = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
    print("=== Validación cruzada ===")
    print(f"Precisión en cada pliegue: {scores}")
    print(f"Precisión media: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n=== Evaluación del modelo ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(modelo, 'modelo_economico.pkl')
    print("Modelo guardado en 'modelo_economico.pkl'")

    return modelo

modelo = joblib.load('modelo_economico.pkl')

nuevo_estudiante = pd.DataFrame([{
    'p27': 1,
    'p29': 4,
    'p30': 1000,
    'p31': 40,
    'p24_6': 1,
    'p24_1': 1,
}])

prediccion = modelo.predict(nuevo_estudiante)[0]
proba = modelo.predict_proba(nuevo_estudiante)[0]

print(f"Predicción: {prediccion}")
print(f"Probabilidad de cada clase: {proba}")"""