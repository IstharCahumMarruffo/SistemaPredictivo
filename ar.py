import pandas as pd

# Nombre del archivo Excel
archivo_excel = "ENDEMS_EXCEL.xlsx"  # Cambia esto por tu archivo

# Leer el archivo Excel
with pd.ExcelFile(archivo_excel) as xls:
    for sheet_name in xls.sheet_names:  # Iterar sobre todas las hojas
        df = pd.read_excel(xls, sheet_name=sheet_name)  
        archivo_csv = f"{sheet_name}.csv"  # Nombre del archivo CSV
        df.to_csv(archivo_csv, index=False, encoding='utf-8')  # Guardar como CSV con UTF-8
        print(f"Hoja '{sheet_name}' convertida a '{archivo_csv}'")

print("Conversión completada.")