from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
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

    df_academico = df[variables_academicas + ['f21', 'estado']].copy()
  
   # print(df_academico["estado"].value_counts())
    X = df_academico[variables_academicas]
    y = df_academico["estado"]

    modelo = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
    #print("=== Validación cruzada modelo académico ===")
    #print(f"Precisión en cada pliegue: {scores}")
    #print(f"Precisión media: {scores.mean():.4f}")
    #print(f"Desviación estándar: {scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    #print("\n=== Evaluación del modelo academico ===")
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    #print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(modelo, 'modelo_academico.pkl')
    #print("Modelo guardado en 'modelo_academico.pkl'")

    return modelo

"""
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión Académicos')
    plt.colorbar()
    ticks = range(len(set(y)))
    plt.xticks(ticks, set(y), rotation=45)
    plt.yticks(ticks, set(y))
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()

   
    fpr, tpr, thresholds = roc_curve(y_test, modelo.predict_proba(X_test)[:,1], pos_label='desertor')
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC Académicos')
    plt.legend(loc='lower right')
    plt.show()

    youden_index = tpr - fpr
    optimal_threshold_index = youden_index.argmax()  
    optimal_threshold = thresholds[optimal_threshold_index]  

    print(f"Umbral óptimo: {optimal_threshold:.2f}")

   
    prediccion_optima = (modelo.predict_proba(X_test)[:,1] >= optimal_threshold).astype(int)
    print("Predicciones con umbral óptimo:")
    print(prediccion_optima)

    plt.figure(figsize=(15, 10))
    plot_tree(modelo, filled=True, feature_names=variables_academicas, class_names=[str(i) for i in set(y)], rounded=True)
    plt.title('Árbol de Decisión Académicos')
    plt.show()"""


#entrenar_modelo_academico()
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

prediccion = modelo.predict(nuevo_estudiante)[0]
proba = modelo.predict_proba(nuevo_estudiante)[0]

print(f"Predicción: {prediccion}")
print(f"Probabilidad de cada clase: {proba}")
"""