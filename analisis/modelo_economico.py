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
from limpieza import cargar_datos_economicos

def entrenar_modelo_economico():
    df = cargar_datos_economicos()

    if df is None:
        print("No se pudieron cargar los datos")
        return

    variables_economicas = ['p27', 'p29', 'p30', 'p31', 'p24_6', 'p24_1']
   
    df_economicos = df[variables_economicas + ['f21']].copy()

    
    df_economicos["estado"] = df_economicos["f21"].map({1: 1, 2: 0})
    df_economicos = df_economicos.dropna(subset=variables_economicas + ['f21'])


    X = df_economicos[variables_economicas]
    y = df_economicos["estado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    #print("=== Resultados de la validación cruzada ===")
    #print(f"Precisión en cada pliegue: {scores}")
    #print(f"Precisión media: {scores.mean():.4f}")
    #print(f"Desviación estándar: {scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    #print("\n=== Evaluación del modelo ===")
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    #print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    #print("ROC AUC:", roc_auc_score(y_test, y_prob))
    
    joblib.dump(pipeline, 'modelo_economico.pkl')
    #print("Modelo guardado como 'modelo_economico.pkl'")

    """cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu")
    plt.title("Matriz de Confusión Modelo Económico")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # === Curva ROC ===
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.title("Curva ROC Modelo Económico")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.show()

    rf_model = pipeline.named_steps['rf']
    plt.figure(figsize=(20, 10))
    plot_tree(rf_model.estimators_[0], 
              feature_names=variables_economicas, 
              class_names=["Desertor", "No Desertor"], 
              filled=True, 
              rounded=True)
    plt.title("Árbol de Decisión Individual del Random Forest Económico")
    plt.show()"""
    
    return pipeline

#entrenar_modelo_economico()


"""
nuevo_estudiante2 = pd.DataFrame([{
    'p27': 2,
    'p29': -1,
    'p30': -1,
    'p31': -1,
    'p24_6': 2,
    'p24_1': 2
}])


nuevo_estudiante = pd.DataFrame([{
    'p27': 1,
    'p29': 4,
    'p30': 2000,
    'p31': 25,
    'p24_6': 1,
    'p24_1': 1
}])


prediccion = modelo.predict(nuevo_estudiante)[0]

# Obtener la probabilidad de cada clase
proba = modelo.predict_proba(nuevo_estudiante)[0]

# Mostrar los resultados
print(f"Predicción: {prediccion}")
print(f"Probabilidad de cada clase: {proba}")"""