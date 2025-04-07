from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from limpieza import cargar_datos_personales
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_modelo_academico():
    # Cargar datos
    df = cargar_datos_personales()
    
    if df is None:
        print("No se pudieron cargar los datos.")
        return
    print(df)
    variables_academicas = [
        'p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1',
        'p14', 'p15', 'p16', 'p17', 'p18', 'p13_1', 'p13_2',
        'p13_3', 'p24_7', 'p24_8', 'p24_19'
    ]

    df_academico = df[variables_academicas + ['f21']].copy()
    df_academico = df_academico.dropna()

    df_academico["estado"] = df_academico["f21"].map({1: 1, 2: 0})
    print(df_academico['estado'].value_counts())
    sns.boxplot(data=df_academico, x="estado", y="p4")
    plt.title("Distribución de p14 por clase")
    plt.show()

    X = df_academico[variables_academicas]
    y = df_academico["estado"]

    modelo = DecisionTreeClassifier(random_state=42)

    scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
    print("=== Resultados de la validación cruzada ===")
    print(f"Precisión en cada pliegue: {scores}")
    print(f"Precisión media: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")

    # Entrenamiento y prueba simple
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("Predicciones únicas:", np.unique(y_pred, return_counts=True))

    print("\n📈 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("✅ Precisión del modelo:", accuracy_score(y_test, y_pred))
    print(y.value_counts(normalize=True))

    return modelo

entrenar_modelo_academico()

