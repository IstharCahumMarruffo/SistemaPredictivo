import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# Configuración de la base de datos (mantener igual)
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "desercion_escolar"
}

# Cargar datos (mantener igual)
try:
    conn = pymysql.connect(**db_config)
    queryD = """
    SELECT f21, p2a, p4, p5, p6, p7, p10h, p10m, p11_1, p14, p15, p16, p17, p18, p13_1, p13_2, p13_3, p24_7, p24_8, p24_19
    FROM datos_desertores 
    WHERE f21 = 1 
    AND p2a != 9999 
    AND p4 < 997 
    AND p15 < 11;
    """
    queryC = """
    SELECT f21, p43a, p44, p45, p46, p47, p50h, p50m, p51_1, p56, p57, p58, p59, p60, p53_1, p53_2, p53_3, p63_7, p63_8, p63_19
    FROM datos_concluidos 
    WHERE f21 = 2 
    AND p43a != 9999 
    AND p44 < 997 
    AND p57 < 11
    ORDER BY RAND() LIMIT 2550;
    """

    df_desertores = pd.read_sql(queryD, conn)
    df_concluidos = pd.read_sql(queryC, conn)
    conn.close()
    print("✅ Datos cargados con éxito.")
except Exception as e:
    print(f"❌ Error al conectar a la base de datos: {e}")
    exit()

# Verificar datos y renombrar columnas (mantener igual)
if df_desertores.empty or df_concluidos.empty:
    raise ValueError("⚠️ Al menos una de las tablas está vacía. Verifica la base de datos.")

columnas_equivalentes = {
    'p43a':'p2a', 'p44':'p4', 'p45':'p5', 'p46':'p6', 'p47':'p7', 'p50h':'p10h', 'p50m':'p10m',
    'p51_1':'p11_1', 'p56':'p14', 'p57':'p15', 'p58':'p16', 'p59':'p17', 'p60':'p18', 'p53_1':'p13_1',
    'p53_2':'p13_2', 'p53_3':'p13_3', 'p63_7':'p24_7', 'p63_8':'p24_8', 'p63_19':'p24_19'
}
df_concluidos = df_concluidos.rename(columns=columnas_equivalentes)

# Combinar DataFrames y crear variable objetivo
df_total = pd.concat([df_desertores, df_concluidos], ignore_index=True)
df_total['desercion'] = df_total['f21'].apply(lambda x: 1 if x == 1 else 0)

# 1. Análisis exploratorio rápido
print("\n🔍 Distribución de clases:")
print(df_total['desercion'].value_counts(normalize=True))

# 2. Selección de características
columnas_academicas = ['p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1', 'p14', 'p15', 
                       'p16', 'p17', 'p18', 'p13_1', 'p13_2', 'p13_3', 'p24_7', 'p24_8', 'p24_19']
X = df_total[columnas_academicas]
y = df_total['desercion']

# 3. Manejo de valores nulos y escalado
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 4. División estratificada de datos
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# 5. Balanceo con SMOTE (solo en entrenamiento)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gbc = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# 7. Mejor modelo
best_model = grid_search.best_estimator_
print(f"\n🌟 Mejores parámetros: {grid_search.best_params_}")

# 8. Evaluación
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_proba)
print(f"\n🌟 AUC-ROC: {auc_score:.4f}")

# 9. Importancia de características
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\n🔝 Importancia de características:")
print(feature_importance.head(10))

# 10. Guardar modelo
joblib.dump(best_model, 'mejor_modelo_desercion_gbc.pkl')
print("\n💾 Modelo guardado como 'mejor_modelo_desercion_gbc.pkl'")