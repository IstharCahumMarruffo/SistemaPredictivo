import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la base de datos
db_config = {
    "host": "localhost",     
    "user": "root",   
    "password": "password", 
    "database": "desercion_escolar"   
}

try:
    # Conexión usando SQLAlchemy
    engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
    
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
    df_desertores = pd.read_sql(queryD, engine)
    df_concluidos = pd.read_sql(queryC, engine)
    
    print("\nDatos cargados con éxito:")
    print(f"Desertores: {df_desertores.shape[0]} registros")
    print(f"Concluidos: {df_concluidos.shape[0]} registros")
    print("\nValores faltantes en desertores:", df_desertores.isna().sum().sum())
    print("Valores faltantes en concluidos:", df_concluidos.isna().sum().sum())
    
except Exception as e:
    print(f"\nError al conectar a la base de datos: {e}")
    raise

if df_desertores.empty or df_concluidos.empty:
    raise ValueError("⚠️ Al menos una de las tablas está vacía. Verifica la base de datos.")

# Mapeo de columnas
columnas_equivalentes = {
    'p43a':'p2a', 'p44':'p4', 'p45':'p5', 'p46':'p6', 'p47':'p7', 'p50h':'p10h', 'p50m':'p10m',
    'p51_1':'p11_1', 'p56':'p14', 'p57':'p15', 'p58':'p16', 'p59':'p17', 'p60':'p18', 'p53_1':'p13_1',
    'p53_2':'p13_2', 'p53_3':'p13_3', 'p63_7':'p24_7', 'p63_8':'p24_8', 'p63_19':'p24_19'
}

df_concluidos = df_concluidos.rename(columns=columnas_equivalentes)
df_total = pd.concat([df_desertores, df_concluidos], ignore_index=True)

print("\nUnión exitosa. Filas totales:", df_total.shape[0])
print("Distribución de clases original:")
print(df_total['f21'].value_counts())

# Creación de la variable objetivo
df_total['desercion'] = df_total['f21'].apply(lambda x: 1 if x == 1 else 0)

# Selección de características
features = ['p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1', 'p14', 'p15', 'p16', 'p17',
                        'p18', 'p13_1', 'p13_2', 'p13_3',  'p24_7', 'p24_8', 'p24_19']


X = df_total[features]
y = df_total['desercion']

# Análisis de valores faltantes
print("\nValores faltantes por columna:")
print(X.isna().sum())

# División estratificada de los datos - CORRECCIÓN AQUÍ
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Usar corchetes [] en lugar de paréntesis ()
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print("\nDistribución en conjunto de entrenamiento:", np.bincount(y_train))
print("Distribución en conjunto de prueba:", np.bincount(y_test))

# Configuración del modelo HistGradientBoosting
hgbt = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    scoring='roc_auc',
    validation_fraction=0.1,
    random_state=42,
    class_weight='balanced'
)

# Entrenamiento del modelo
print("\nEntrenando modelo HistGradientBoosting...")
hgbt.fit(X_train, y_train)

# Predicción y evaluación
y_pred = hgbt.predict(X_test)
y_pred_proba = hgbt.predict_proba(X_test)[:, 1]

print("\n=== Resultados del Modelo ===")
print("Precisión:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))


# Guardar el modelo entrenado
joblib.dump(hgbt, 'modelo_entrenado_acad.pkl')
print("\nModelo HistGradientBoosting guardado exitosamente como 'modelo_entrenado_acad.pkl'")
# Cargar el modelo desde el archivo
modelo_entrenado_acad = joblib.load('modelo_entrenado_acad.pkl')
