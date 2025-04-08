"""import pandas as pd

# Ruta al archivo
file_path = '/home/ISTHAR8/Downloads/PT/ENDEMS.csv'

# Leer el archivo
df = pd.read_csv(file_path)

# Ver las primeras filas
print(df.head())

# Opcional: inspeccionar columnas
print(df.columns)

# También puedes revisar si hay comillas, saltos de línea, etc.
print(df.iloc[0])

with open(file_path, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    print(first_line)


# Condiciones para datos_desertores
condiciones_desertores = (df['f21'] == 1) 
# Contar las filas que cumplen las condiciones
contador_desertores = df[condiciones_desertores].shape[0]
print(f"Número de filas en datos_desertores: {contador_desertores}")

# Condiciones para datos_concluidos
condiciones_concluidos = (df['f21'] == 2) 

# Contar las filas que cumplen las condiciones
contador_concluidos = df[condiciones_concluidos].shape[0]
print(f"Número de filas en datos_concluidos: {contador_concluidos}")
"""