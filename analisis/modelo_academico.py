from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import (
#   classification_report,
#    accuracy_score,
#    confusion_matrix,
#    roc_auc_score,
#    roc_curve,
#    auc
#)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import seaborn as sns
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from limpieza import cargar_datos_academicos


def entrenar_modelo_academico():
    df = cargar_datos_academicos()

    if df is None:
        print("No se pudieron cargar los datos")
        return
    
    variables_academicas = [
        'p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1',
        'p14', 'p15', 'p16', 'p17', 'p18', 'p13_1', 'p13_2',
        'p13_3', 'p24_7', 'p24_8'
    ]

    df_academico = df[variables_academicas + ['f21']].copy()
    df_academico = df_academico.dropna(subset=variables_academicas+['f21'])
    df_academico["estado"] = df_academico["f21"].map({1: 1, 2:0})

    X = df_academico[variables_academicas]
    y = df_academico["estado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    
    #print("\n\n=== Validación cruzada Random Forest ===")
    #print(f"Precisión en cada pliegue: {scores}")
    #print(f"Precisión media: {scores.mean():.4f}")
    #print(f"Desviación estándar: {scores.std():.4f}")

    pipeline.fit(X_train, y_train)


    y_pred = pipeline.predict(X_test)

    #print("\n=== Evaluación del modelo Random Forest ===")
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    #print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    #print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # Guardar el modelo entrenado
    joblib.dump(pipeline, 'modelo_random_forest.pkl')
    #print("Modelo guardado en 'modelo_random_forest.pkl'")

    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu")
    plt.title("Matriz de Confusión Modelo Académico")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # === Curva ROC ===
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
    plt.title("Curva ROC Modelo Académico")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.show()

    # === Visualización de un árbol individual del Random Forest (opcional) ===
    rf_model = pipeline.named_steps['rf']
    plt.figure(figsize=(20, 10))
    plot_tree(rf_model.estimators_[0], 
              feature_names=variables_academicas, 
              class_names=["Desertor", "No Desertor"], 
              filled=True, 
              rounded=True)
    plt.title("Árbol de Decisión Individual del Random Forest Modelo Académico")
    plt.show()"""

    return pipeline

   

"""
modelo = joblib.load('modelo_academico.pkl')

nuevo_estudiante = pd.DataFrame([{
    'p2a': 2024,
    'p4': 20,
    'p5': 6,
    'p6': 2,
    'p7': 3,
    'p10h': 2,
    'p10m': 20,
    'p11_1': 3,
    'p14': 4,
    'p15': 6,
    'p16': 3,
    'p17': 4,
    'p18': 2,
    'p13_1': 4,
    'p13_2': 4,
    'p13_3': 1, 
    'p24_7': 1,
    'p24_8': 1
}])

nuevo_estudiante2 = pd.DataFrame([{
    'p2a': 2024,
    'p4': 80,
    'p5': 8,
    'p6': 2,
    'p7': 4,
    'p10h': 0,
    'p10m': 20,
    'p11_1': 1,
    'p14': 2,
    'p15': 8,
    'p16': 1,
    'p17': 2,
    'p18': 2,
    'p13_1': 1,
    'p13_2': 1,
    'p13_3': 2, 
    'p24_7': 2,
    'p24_8': 2
}])

prediccion = modelo.predict(nuevo_estudiante2)[0]
proba = modelo.predict_proba(nuevo_estudiante2)[0]

print(f"Predicción: {prediccion}")
print(f"Probabilidad de cada clase: {proba}")
"""