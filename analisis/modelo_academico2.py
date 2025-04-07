from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from limpieza import cargar_datos_personales
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

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

    # Aplicando SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"Tamaño del conjunto de datos después del sobremuestreo: {Counter(y_resampled)}")

    # Usando Random Forest
    modelo = RandomForestClassifier(random_state=42)

    # Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X_resampled, y_resampled, cv=cv, scoring='accuracy')
    print("=== Resultados de la validación cruzada ===")
    print(f"Precisión en cada pliegue: {scores}")
    print(f"Precisión media: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")

    # Ajuste de hiperparámetros con GridSearch
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_resampled, y_resampled)

    print("Mejores parámetros:", grid_search.best_params_)

    # Entrenamiento y prueba simple
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    modelo = grid_search.best_estimator_  # Usar el mejor modelo encontrado
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("Predicciones únicas:", np.unique(y_pred, return_counts=True))

    print("\n📈 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("✅ Precisión del modelo:", accuracy_score(y_test, y_pred))
    print(y_resampled.value_counts(normalize=True))

    return modelo

entrenar_modelo_academico()
