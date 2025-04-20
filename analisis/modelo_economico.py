from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from limpieza import cargar_datos_economicos

def entrenar_modelo_economico():
    df = cargar_datos_economicos()

    if df is None:
        print("No se pudieron cargar los datos")
        return

    variables_economicas = ['p27', 'p29', 'p30', 'p31', 'p24_6', 'p24_1']
    df_economicos = df[variables_economicas + ['f21']].copy()

    # Eliminar filas con valores faltantes
    df_economicos = df_economicos.dropna(subset=variables_economicas + ['f21'])

    # Asegurarse de que 'f21' sea entero
    df_economicos['f21'] = df_economicos['f21'].astype(int)

    # Crear la variable binaria 'estado'
    df_economicos["estado"] = df_economicos["f21"].map({1: 1, 2: 0})

    #print("Distribución de clases en 'estado':")
    #print(df_economicos["estado"].value_counts())
    #print("Valores únicos:", df_economicos["estado"].unique())

    # Definir variables predictoras y objetivo
    X = df_economicos[variables_economicas]
    y = df_economicos["estado"]

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE para balancear
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Definir y entrenar el modelo
    modelo = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(modelo, X_train_res, y_train_res, cv=5, scoring='accuracy')

    #print("=== Resultados de la validación cruzada ===")
    #print(f"Precisión en cada pliegue: {scores}")
    #print(f"Precisión media: {scores.mean():.4f}")
    #print(f"Desviación estándar: {scores.std():.4f}")

    # Entrenar el modelo final
    modelo.fit(X_train_res, y_train_res)

    # Evaluar en test
    y_pred = modelo.predict(X_test)
    #print("\n=== Evaluación del modelo ===")
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    #print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # Guardar el modelo
    joblib.dump(modelo, 'modelo_economico.pkl')
    #print("Modelo guardado como 'modelo_economico.pkl'")

    return modelo
# Ejecutar función
#entrenar_modelo_economico()

"""
modelo = joblib.load('modelo_economico.pkl')

# Datos de un nuevo estudiante

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


prediccion = modelo.predict(nuevo_estudiante2)[0]

# Obtener la probabilidad de cada clase
proba = modelo.predict_proba(nuevo_estudiante2)[0]

# Mostrar los resultados
print(f"Predicción: {prediccion}")
print(f"Probabilidad de cada clase: {proba}")"""