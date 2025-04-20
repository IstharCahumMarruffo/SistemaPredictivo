"""import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='password',
    database='desercion_escolar'
)

desertores = pd.read_sql('SELECT * FROM datos_desertores', conn)
concluyentes = pd.read_sql('SELECT * FROM datos_concluidos', conn)
no_matriculados = pd.read_sql('SELECT * FROM datos_no_matriculados', conn)

conn.close()

print("--------------------------------------------")
print("primeras filas de los tres DataFrames")
print(desertores.head())
print(concluyentes.head())
print(no_matriculados.head())

print("--------------------------------------------")
print("Información general de los tres DataFrames")
print("Desertores:")
print(desertores.info())

print("\nConcluyentes:")
print(concluyentes.info())

print("\nNo Matriculados:")
print(no_matriculados.info())


print("--------------------------------------------")
print("Resumen estadístico de los datos")
print("\nEstadísticas descriptivas - Desertores:")
print(desertores.describe())

print("\nEstadísticas descriptivas - Concluyentes:")
print(concluyentes.describe())

print("\nEstadísticas descriptivas - No Matriculados:")
print(no_matriculados.describe())

print("--------------------------------------------")
print("\nValores nulos en Desertores:")
print(desertores.isnull().sum())

print("\nValores nulos en Concluyentes:")
print(concluyentes.isnull().sum())

print("\nValores nulos en No Matriculados:")
print(no_matriculados.isnull().sum())

"""