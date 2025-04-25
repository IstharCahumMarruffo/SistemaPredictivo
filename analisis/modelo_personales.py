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
from limpieza import cargar_datos_personales

def entrenar_modelo_personal():
    df = cargar_datos_personales()
    if df is None:
        print("No se pudieron cargar los datos.")
        return

    variables_personales = [
        's1','p12_1','p12_2','p12_3','p12_4','p12_5','p12_6','p12_7','p12_8','p12_9','p12_10','p13_4','p13_5','p13_6','p13_7',
        'p23_1','p41a','p41b','p41c','p41d','p41e','p41f','p41g','p41h','p41i','p24_10','p24_12','p24_13','p24_14',
        'p24_18','p24_22','p24_17','p24_11','s9p','s9m'
    ]

    df_personal = df[variables_personales + ['f21']].copy()
    df_personal["estado"] = df_personal["f21"].map({1: 1, 2: 0})
    df_personal = df_personal.dropna()

    X = df_personal[variables_personales]
    y = df_personal["estado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline ([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
   
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    #print("=== Resultados de la validación cruzada modelo personal ===")
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
    plt.title("Matriz de Confusión Modelo Personal")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # === Curva ROC ===
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
    plt.title("Curva ROC Modelo Personal")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.show()

    # === Visualización de un árbol individual del Random Forest (opcional) ===
    rf_model = pipeline.named_steps['rf']
    plt.figure(figsize=(20, 10))
    plot_tree(rf_model.estimators_[0], 
              feature_names=variables_personales, 
              class_names=["Desertor", "No Desertor"], 
              filled=True, 
              rounded=True)
    plt.title("Árbol de Decisión Individual del Random Forest Modelo Personal")
    plt.show()
"""
    return pipeline

#entrenar_modelo_personal()
