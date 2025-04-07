import pandas as pd
import pymysql

def cargar_datos_academicos():
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "password",
        "database": "desercion_escolar"
    }

    columnas_equivalentes = {
        'p43a': 'p2a', 's2a': 'p2a',
        'p44': 'p4', 'p45': 'p5', 'p46': 'p6', 'p47': 'p7',
        'p50h': 'p10h', 'p50m': 'p10m', 'p51_1': 'p11_1',
        'p56': 'p14', 'p57': 'p15', 'p58': 'p16', 'p59': 'p17', 'p60': 'p18',
        'p53_1': 'p13_1', 'p53_2': 'p13_2', 'p53_3': 'p13_3',
        'p63_7': 'p24_7', 'p63_8': 'p24_8', 'p63_19': 'p24_19'
    }

    try:
        conn = pymysql.connect(**db_config)

        queryD = """
        SELECT f21, p2a, p4, p5, p6, p7, p10h, p10m, p11_1, p14, p15, p16, p17, p18,
               p13_1, p13_2, p13_3, p24_7, p24_8, p24_19
        FROM datos_desertores 
        WHERE f21 = 1 
          AND p2a != 9999 
          AND p4 < 997 
          AND p15 < 11;
        """

        queryC = """
        SELECT f21, p43a, s2a, p44, p45, p46, p47, p50h, p50m, p51_1, p56, p57, p58, p59, p60,
               p53_1, p53_2, p53_3, p63_7, p63_8, p63_19
        FROM datos_concluidos 
        WHERE f21 = 2 
          AND p43a != 9999 
          AND p44 < 997 
          AND p57 < 11;
        """

        df_desertores = pd.read_sql(queryD, conn)
        df_concluidos = pd.read_sql(queryC, conn)
        conn.close()
        print("✅ Datos cargados con éxito desde la base de datos.")

    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        return None

    # Revisar contenido inicial
    print("📊 Desertores - Primeros datos:")
    print(df_desertores)

    print("\n📊 Concluidos (antes del rename) - Primeros datos:")
    print(df_concluidos)

    # Renombrar columnas
    df_concluidos_renamed = df_concluidos.rename(columns=columnas_equivalentes)
    # Eliminar columnas duplicadas por nombre (solo deja la primera ocurrencia)
    df_concluidos_renamed = df_concluidos_renamed.loc[:, ~df_concluidos_renamed.columns.duplicated()]


    print("\n🔁 Concluidos (después del rename):")
    print(df_concluidos_renamed.head())

    # Revisar columnas en común
    columnas_comunes = df_desertores.columns.intersection(df_concluidos_renamed.columns)
    print("\n📌 Columnas comunes entre desertores y concluidos:", list(columnas_comunes))

    # Filtrar por columnas en común
    df_desertores = df_desertores[columnas_comunes].copy()
    df_concluidos_renamed = df_concluidos_renamed[columnas_comunes].copy()

    # Agregar columna de estado
    df_desertores["estado"] = "desertor"
    df_concluidos_renamed["estado"] = "concluido"

    # Unir datasets
    df_academicos = pd.concat([df_desertores, df_concluidos_renamed], ignore_index=True)

    print("\n✅ Datos combinados correctamente. Vista previa final:")
    print(df_academicos.head())

    print("\n🔍 Valores nulos por columna:")
    print(df_academicos.isnull().sum())

    for columna in df_academicos:
        valores_unicos = df_academicos[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")

    # Filtrar filas donde f21 == 1
    df_filtrado = df_desertores[df_desertores["f21"] == 1]

    print("\n🔍 Valores únicos por columna (solo donde f21 == 1):")
    for columna in df_filtrado.columns:
        valores_unicos = df_filtrado[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")


    return df_academicos

def cargar_datos_personales():
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "password",
        "database": "desercion_escolar"
    }

    columnas_equivalentes = {
        'p43a': 'p2a', 's2a': 'p2a', 
        'p44': 'p4', 'p45': 'p5', 'p46': 'p6', 'p47': 'p7',
        'p50h': 'p10h', 'p50m': 'p10m', 'p51_1': 'p11_1',
        'p56': 'p14', 'p57': 'p15', 'p58': 'p16', 'p59': 'p17', 'p60': 'p18',
        'p53_1': 'p13_1', 'p53_2': 'p13_2', 'p53_3': 'p13_3',
        'p63_7': 'p24_7', 'p63_8': 'p24_8', 'p63_19': 'p24_19'
    }

    try:
        conn = pymysql.connect(**db_config)

        queryD = """
        SELECT f21, p2a, p4, p5, p6, p7, p10h, p10m, p11_1, p14, p15, p16, p17, p18,
               p13_1, p13_2, p13_3, p24_7, p24_8, p24_19
        FROM datos_desertores 
        WHERE f21 = 1 
          AND p2a != 9999 
          AND p4 < 997 
          AND p15 < 11;
        """

        queryC = """
        SELECT f21, p43a, s2a, p44, p45, p46, p47, p50h, p50m, p51_1, p56, p57, p58, p59, p60,
               p53_1, p53_2, p53_3, p63_7, p63_8, p63_19
        FROM datos_concluidos 
        WHERE f21 = 2 
          AND p43a != 9999 
          AND p44 < 997 
          AND p57 < 11;
        """

        df_desertores = pd.read_sql(queryD, conn)
        df_concluidos = pd.read_sql(queryC, conn)
        conn.close()
        print("✅ Datos cargados con éxito desde la base de datos.")

    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        return None

    # Revisar contenido inicial
    print("📊 Desertores - Primeros datos:")
    print(df_desertores)

    print("\n📊 Concluidos (antes del rename) - Primeros datos:")
    print(df_concluidos)

    # Renombrar columnas
    df_concluidos_renamed = df_concluidos.rename(columns=columnas_equivalentes)
    # Eliminar columnas duplicadas por nombre (solo deja la primera ocurrencia)
    df_concluidos_renamed = df_concluidos_renamed.loc[:, ~df_concluidos_renamed.columns.duplicated()]


    print("\n🔁 Concluidos (después del rename):")
    print(df_concluidos_renamed.head())

    # Revisar columnas en común
    columnas_comunes = df_desertores.columns.intersection(df_concluidos_renamed.columns)
    #print("\n📌 Columnas comunes entre desertores y concluidos:", list(columnas_comunes))

    # Filtrar por columnas en común
    df_desertores = df_desertores[columnas_comunes].copy()
    df_concluidos_renamed = df_concluidos_renamed[columnas_comunes].copy()

    # Agregar columna de estado
    df_desertores["estado"] = "desertor"
    df_concluidos_renamed["estado"] = "concluido"

    # Unir datasets
    df_academicos = pd.concat([df_desertores, df_concluidos_renamed], ignore_index=True)

    #print("\n✅ Datos combinados correctamente. Vista previa final:")
    #print(df_academicos.head())

   # print("\n🔍 Valores nulos por columna:")
   # print(df_academicos.isnull().sum())

    for columna in df_academicos:
        valores_unicos = df_academicos[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")

    # Filtrar filas donde f21 == 1
    df_filtrado = df_desertores[df_desertores["f21"] == 1]

    print("\n🔍 Valores únicos por columna (solo donde f21 == 1):")
    for columna in df_filtrado.columns:
        valores_unicos = df_filtrado[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")

    print (df_academicos)
    return df_academicos
df=cargar_datos_academicos()
print(df)

def cargar_datos_personales():
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "password",
        "database": "desercion_escolar"
    }

    columnas_equivalentes = {  'p52_1': 'p12_1', 'p52_2': 'p12_2', 'p52_3': 'p12_3', 'p52_4': 'p12_4', 'p52_5': 'p12_5',
                 'p52_6': 'p12_6', 'p52_7': 'p12_7', 'p52_8': 'p12_8', 'p52_9': 'p12_9', 'p52_10': 'p12_10',
                 'p53_4': 'p13_4', 'p53_5': 'p13_5', 'p53_6': 'p13_6','p62_1':'p23_1', 'p53_7': 'p13_7',
                 'p63_1': 'p23_1', 'p63_2': 'p23_2',
                 'p74a': 'p41a', 'p74b': 'p41b', 'p74c': 'p41c', 'p74d': 'p41d', 'p74e': 'p41e',
                 'p74f': 'p41f', 'p74g': 'p41g', 'p74h': 'p41h', 'p74i': 'p41i', 'p63_1':'p24_1',
                 'p63_10': 'p24_10','p63_11':'p24_11', 'p63_12': 'p24_12', 'p63_13': 'p24_13', 'p63_14': 'p24_14',
                 'p63_15': 'p24_15', 'p63_17': 'p24_17', 'p63_18': 'p24_18', 'p63_22': 'p24_22'

    }

    try:
        conn = pymysql.connect(**db_config)

        queryD = """
       SELECT f21, s1,p12_1,p12_2,p12_3,p12_4,p12_5,p12_6,p12_7,p12_8,p12_9,p12_10,p13_4,p13_5,p13_6,p13_7,
              p23_1,p41a,p41b,p41c,p41d,p41e,p41f,p41g,p41h,p41i,p24_1,p24_10,p24_12,p24_13,p24_14,
              p24_18,p24_22,p24_17,p24_11,s9p,s9m FROM datos_desertores WHERE f21 = 1;
        """

        queryC = """
        SELECT f21, s1,p52_1,p52_2,p52_3,p52_4,p52_5,p52_6,p52_7,p52_8,p52_9,p52_10,p53_4,p53_5,p53_6,p53_7,
                p62_1,p63_1,p63_2,p74a,p74b,p74c,p74d,p74e,p74f,p74g,p74h,p74i,p63_1,p63_10,p63_12,p63_13,p63_14,
                p63_18,p63_22,p63_17,p63_11,p63_15,s9p,s9m FROM datos_concluidos WHERE f21 = 2;
        """ 

        df_desertores = pd.read_sql(queryD, conn)
        df_concluidos = pd.read_sql(queryC, conn)
        conn.close()
        print("✅ Datos cargados con éxito desde la base de datos.")

    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        return None

    # Revisar contenido inicial
    #print("📊 Desertores - Primeros datos:")
   # print(df_desertores)

    print("\n📊 Concluidos (antes del rename) - Primeros datos:")
    print(df_concluidos)

    # Renombrar columnas
    df_concluidos_renamed = df_concluidos.rename(columns=columnas_equivalentes)
    # Eliminar columnas duplicadas por nombre (solo deja la primera ocurrencia)
    df_concluidos_renamed = df_concluidos_renamed.loc[:, ~df_concluidos_renamed.columns.duplicated()]


    #print("\n🔁 Concluidos (después del rename):")
    #print(df_concluidos_renamed.head())

    # Revisar columnas en común
    columnas_comunes = df_desertores.columns.intersection(df_concluidos_renamed.columns)
    #print("\n📌 Columnas comunes entre desertores y concluidos:", list(columnas_comunes))

    # Filtrar por columnas en común
    df_desertores = df_desertores[columnas_comunes].copy()
    df_concluidos_renamed = df_concluidos_renamed[columnas_comunes].copy()

    # Agregar columna de estado
    df_desertores["estado"] = "desertor"
    df_concluidos_renamed["estado"] = "concluido"

    # Unir datasets
    df_personales = pd.concat([df_desertores, df_concluidos_renamed], ignore_index=True)

    #print("\n✅ Datos combinados correctamente. Vista previa final:")
    #print(df_personales.head())

    #print("\n🔍 Valores nulos por columna:")
    #print(df_personales.isnull().sum())

    for columna in df_personales:
        valores_unicos = df_personales[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")

    # Filtrar filas donde f21 == 1
    df_filtrado = df_desertores[df_desertores["f21"] == 1]

    #print("\n🔍 Valores únicos por columna (solo donde f21 == 1):")
    for columna in df_filtrado.columns:
        valores_unicos = df_filtrado[columna].unique()
        #print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")


    return df_personales
#cargar_datos_personales()

def cargar_datos_generales ():
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "password",
        "database": "desercion_escolar"
    }

    columnas_equivalentes = {
        'p57':'p15'   }

    try:
        conn = pymysql.connect(**db_config)

        queryD = """
        SELECT f21, f8e_1, s1, p15
        FROM datos_desertores 
        WHERE f21 = 1
        AND p15<11;
        """

        queryC = """
        SELECT f21, f8e_1, s1, p57
        FROM datos_concluidos 
        WHERE f21 = 2 
        AND p57<11;


        """

        df_desertores = pd.read_sql(queryD, conn)
        df_concluidos = pd.read_sql(queryC, conn)
        conn.close()
        print("✅ Datos cargados con éxito desde la base de datos.")

    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        return None

    # Revisar contenido inicial
    print("📊 Desertores - Primeros datos:")
    print(df_desertores)

    print("\n📊 Concluidos (antes del rename) - Primeros datos:")
    print(df_concluidos)

    # Renombrar columnas
    df_concluidos_renamed = df_concluidos.rename(columns=columnas_equivalentes)
    # Eliminar columnas duplicadas por nombre (solo deja la primera ocurrencia)
    df_concluidos_renamed = df_concluidos_renamed.loc[:, ~df_concluidos_renamed.columns.duplicated()]
    # Unir datasets
    df_general = pd.concat([df_desertores, df_concluidos_renamed], ignore_index=True)

    print("\n✅ Datos combinados correctamente. Vista previa final:")
    print(df_general.head())

    print("\n🔍 Valores nulos por columna:")
    print(df_general.isnull().sum())

    for columna in df_general:
        valores_unicos = df_general[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")

    # Filtrar filas donde f21 == 1
    df_filtrado = df_desertores[df_desertores["f21"] == 1]

    print("\n🔍 Valores únicos por columna (solo donde f21 == 1):")
    for columna in df_filtrado.columns:
        valores_unicos = df_filtrado[columna].unique()
        print(f"🔎 {columna} ({len(valores_unicos)} valores únicos): {valores_unicos}")


    return df_general

#cargar_datos_generales()
    