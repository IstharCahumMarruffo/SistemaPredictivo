from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from limpieza import cargar_datos_academicos
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib

def entrenar_modelo_academico():
    # Cargar datos
    df = cargar_datos_academicos()
    
    if df is None:
        print("No se pudieron cargar los datos.")
        return
    
    variables_academicas = [
        'p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1',
        'p14', 'p15', 'p16', 'p17', 'p18', 'p13_1', 'p13_2',
        'p13_3', 'p24_7', 'p24_8'
    ]

    df_academico = df[variables_academicas + ['f21']].copy()
    df_academico = df_academico.dropna()

    df_academico["estado"] = df_academico["f21"].map({1: 1, 2: 0})

    X = df_academico[variables_academicas]
    y = df_academico["estado"]

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"Distribución de clases después del sobremuestreo: {np.bincount(y_res)}")

    modelo = DecisionTreeClassifier(random_state=42)

    # Evaluación del modelo con validación cruzada
    scores = cross_val_score(modelo, X_res, y_res, cv=5, scoring='accuracy')
    print("=== Resultados de la validación cruzada ===")
    print(f"Precisión en cada pliegue: {scores}")
    print(f"Precisión media: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")

    # Entrenamiento y prueba simple
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)

    # Obtener las probabilidades de predicción para la clase positiva
    probas = modelo.predict_proba(X_test)[:, 1]

    # Calcular la curva ROC y el índice de Youden
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n🔍 Umbral óptimo según índice de Youden: {optimal_threshold:.4f}")

    # Ajustar las predicciones usando el umbral óptimo
    y_pred_opt = (probas >= optimal_threshold).astype(int)

    print("\n📊 Reporte con umbral óptimo:")
    print(classification_report(y_test, y_pred_opt))
    print("✅ Precisión del modelo (umbral óptimo):", accuracy_score(y_test, y_pred_opt))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
    plt.title(f"Matriz de Confusión (umbral óptimo: {optimal_threshold:.2f})")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Curva ROC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.2f})")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Umbral óptimo = {optimal_threshold:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()

    # Guardar el modelo entrenado
    joblib.dump(modelo, 'modelo_academico.pkl')
    print("Modelo guardado en 'modelo_academico.pkl'")

    return modelo

entrenar_modelo_academico()
