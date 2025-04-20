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
    df_personal = df_personal.dropna()

    df_personal["estado"] = df_personal["f21"].map({1: 1, 2: 0})
    
    print(df_personal['estado'].value_counts())

    X = df_personal[variables_personales]
    y = df_personal["estado"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    modelo = DecisionTreeClassifier(random_state=42)

    scores = cross_val_score(modelo, X_res, y_res, cv=5, scoring='accuracy')
   # print("=== Resultados de la validación cruzada modelo personal ===")
    #print(f"Precisión en cada pliegue: {scores}")
    #print(f"Precisión media: {scores.mean():.4f}")
    #print(f"Desviación estándar: {scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)
   
    y_pred = modelo.predict(X_test)
   # print("\n=== Evaluación del modelo personal ===")
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    #print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
   
    """
    probas = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\n Umbral óptimo según índice de Youden: {optimal_threshold:.4f}")

    y_pred_opt = (probas >= optimal_threshold).astype(int)

    print("\n Reporte con umbral óptimo:")
    print(classification_report(y_test, y_pred_opt))
    print("Precisión del modelo (umbral óptimo):", accuracy_score(y_test, y_pred_opt))

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
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Umbral óptimo = {optimal_threshold:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()
    """
    joblib.dump(modelo, 'modelo_personal.pkl')
    return modelo

#entrenar_modelo_personal()