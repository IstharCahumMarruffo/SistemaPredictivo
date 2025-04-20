from analisis.modelo_academico import entrenar_modelo_academico
from analisis.modelo_personales import entrenar_modelo_personal
from analisis.modelo_economico import entrenar_modelo_economico
import numpy as np

def evaluar_modelos(df):
    modelo_academico = entrenar_modelo_academico()
    modelo_personal = entrenar_modelo_personal()
    modelo_economico = entrenar_modelo_economico()
    
    CARACTERISTICAS_MODELOS = {
        modelo_academico: [
            'p2a', 'p4', 'p5', 'p6', 'p7', 'p10h', 'p10m', 'p11_1', 'p14', 'p15', 'p16', 'p17', 'p18', 
            'p13_1', 'p13_2', 'p13_3', 'p24_7', 'p24_8'
        ],
        modelo_personal: [
            's1','p12_1','p12_2','p12_3','p12_4','p12_5','p12_6','p12_7','p12_8','p12_9','p12_10',
            'p13_4','p13_5','p13_6','p13_7','p23_1','p41a','p41b','p41c','p41d','p41e','p41f','p41g', 
            'p41h','p41i','p24_10','p24_12','p24_13','p24_14','p24_18','p24_22','p24_17','p24_11',
            's9p','s9m'
        ],
        modelo_economico: [
            'p27', 'p29', 'p30', 'p31', 'p24_6', 'p24_1'
        ]
    }

    conversor = {0: 'concluido', 1: 'desertor'}
    
    resultados = []
    
    for _, estudiante in df.iterrows():
        predicciones = []
        
        for modelo, columnas in CARACTERISTICAS_MODELOS.items():
            try:
                caracteristicas = estudiante[columnas].values.reshape(1, -1)

                if caracteristicas.shape[1] != modelo.n_features_in_:
                    raise ValueError(f"El modelo espera {modelo.n_features_in_} características, pero recibió {caracteristicas.shape[1]}.")
                
                prediccion = modelo.predict(caracteristicas)
                pred_val = prediccion[0]

                # Convertir predicción correctamente
                if isinstance(pred_val, (int, float, np.integer, np.floating)):
                    palabra = conversor.get(int(pred_val), str(pred_val))
                elif isinstance(pred_val, str) and pred_val.lower() in conversor.values():
                    palabra = pred_val.lower()
                else:
                    palabra = str(pred_val)

                predicciones.append(palabra)

            except KeyError as e:
                predicciones.append(f"Error: columna faltante {e}")
            except Exception as e:
                predicciones.append(f"Error: {e}")
        
        resultados.append({
            'estudiante': estudiante.get('nombre', 'Desconocido'),
            'predicciones': predicciones
        })
    
    return resultados
