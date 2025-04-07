import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Configuración de la base de datos
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "desercion_escolar"
}

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
    SELECT f21, p43a, s2a, p44, p45, p46, p47, p50h, p50m, p51_1, p56, p57, p58, p59, p60, p53_1, p53_2, p53_3, p63_7, p63_8, p63_19
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

if df_desertores.empty or df_concluidos.empty:
    raise ValueError("⚠️ Al menos una de las tablas está vacía. Verifica la base de datos.")

# Renombrar columnas para alinearlas entre las tablas
columnas_equivalentes = {
    'p43a':'p2a', 'p44':'p4', 'p45':'p5', 'p46':'p6', 'p47':'p7', 'p50h':'p10h', 'p50m':'p10m',
    'p51_1':'p11_1', 'p56':'p14', 'p57':'p15', 'p58':'p16', 'p59':'p17', 'p60':'p18', 'p53_1':'p13_1',
    'p53_2':'p13_2', 'p53_3':'p13_3','p63_7':'p24_7', 'p63_8':'p24_8', 'p63_19':'p24_19'
}
df_concluidos = df_concluidos.rename(columns=columnas_equivalentes)

# Combinar ambos DataFrames
df_total = pd.concat([df_desertores, df_concluidos], ignore_index=True)
print(f"✅ Unión exitosa. Filas totales: {df_total.shape[0]}")

# Verificar existencia de la columna objetivo
if 'f21' not in df_total.columns:
    raise ValueError("⚠️ La columna 'f21' no existe en la base de datos.")

df_total['desercion'] = df_total['f21'].apply(lambda x: 1 if x == 1 else 0)

# Seleccionar características académicas
columnas_academicas = ['p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1', 'p14', 'p15', 'p16', 'p17',
                        'p18', 'p13_1', 'p13_2', 'p13_3',  'p24_7', 'p24_8', 'p24_19']
X = df_total[columnas_academicas]
y = df_total['desercion']

# Manejo de valores NaN
print("\n🔍 Valores NaN antes del procesamiento:")
print(X.isna().sum())
X = X.fillna(X.median())  # Rellenar con la mediana de cada columna

# Dividir los datos en entrenamiento y prueba
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Entrenar el modelo con un árbol de decisión con profundidad limitada
# Incrementar profundidad del árbol
model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2) # Limitar la profundidad del árbol
model.fit(X_train_resampled, y_train_resampled)

# Predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo con las métricas de clasificación
print("\n📈 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("✅ Precisión del modelo:", accuracy_score(y_test, y_pred))

# Guardar el modelo
joblib.dump(model, 'modelo_desercion_ajustado.pkl')
print("💾 Modelo ajustado guardado como 'modelo_desercion_ajustado.pkl'")