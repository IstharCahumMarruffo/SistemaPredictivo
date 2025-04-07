import pandas as pd
import pymysql
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import seaborn as sns

# Configuración de la base de datos
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "desercion_escolar"
}

# 1. Carga y preparación de datos
try:
    conn = pymysql.connect(**db_config)
    
    # Mejoramos las consultas para incluir más variables relevantes
    queryD = """
    SELECT f21, p2a, p4, p5, p6, p7, p10h, p10m, p11_1, p14, p15, p16, p17, p18, p13_1, p13_2, p13_3, p24_7, p24_8, p24_19
    FROM datos_desertores 
    WHERE f21 = 1 
    AND p2a != 9999 
    AND p4 < 997 
    AND p15 < 11;
    """
    
    queryC = """
    SELECT f21, s2a,p43a as p2a, p44 as p4, p45 as p5, p46 as p6, p47 as p7, 
    p50h as p10h, p50m as p10m, p51_1 as p11_1, p56 as p14, p57 as p15, 
    p58 as p16, p59 as p17, p60 as p18, p53_1 as p13_1, p53_2 as p13_2, p53_3 as p13_3,  
    p63_7 as p24_7, p63_8 as p24_8, p63_19 as p24_19
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
    print("✅ Datos cargados con éxito. Desertores:", df_desertores.shape[0], "Concluidos:", df_concluidos.shape[0])
except Exception as e:
    print(f"❌ Error al conectar a la base de datos: {e}")
    exit()

# 2. Unificación y limpieza de datos
df_total = pd.concat([df_desertores, df_concluidos], ignore_index=True)
df_total['desercion'] = df_total['f21'].apply(lambda x: 1 if x == 1 else 0)

# Ingeniería de características: crear nuevas variables relevantes
df_total['horas_estudio_semana'] = df_total['p10h'] + df_total['p10m']/60
df_total['indice_dificultad'] = df_total[['p13_1', 'p13_2', 'p13_3']].mean(axis=1)
df_total['relacion_familia'] = df_total[['p24_7', 'p24_8', 'p24_19']].sum(axis=1)

# Selección de características más relevantes
features = ['p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1', 'p14', 'p15', 'p16', 'p17',
                        'p18', 'p13_1', 'p13_2', 'p13_3',  'p24_7', 'p24_8', 'p24_19', 'horas_estudio_semana', 'indice_dificultad', 
            'relacion_familia']

X = df_total[features]
y = df_total['desercion']

# 3. Análisis exploratorio (visualización)
plt.figure(figsize=(10,6))
sns.countplot(x='desercion', data=df_total)
plt.title('Distribución de clases')
plt.show()

# 4. Preprocesamiento
# Manejo de valores faltantes
X = X.fillna(X.median())

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 5. División de datos
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X_scaled, y):
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# 6. Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 7. Selección de características
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_res, y_train_res)
X_test_selected = selector.transform(X_test)

# 8. Modelado con RandomForest (mejor que árbol simple)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

# Entrenamiento
model.fit(X_train_selected, y_train_res)

# 9. Evaluación
y_pred = model.predict(X_test_selected)
y_proba = model.predict_proba(X_test_selected)[:,1]

print("\n📈 Métricas de evaluación:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# 10. Guardar modelo
joblib.dump(model, 'mejor_modelo_desercion.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
print("💾 Modelo guardado como 'mejor_modelo_desercion.pkl'")

# 11. Importancia de características
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features = np.array(features)[selector.get_support()]

plt.figure(figsize=(10,6))
plt.title("Importancia de las características")
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), selected_features[indices])
plt.gca().invert_yaxis()
plt.show()