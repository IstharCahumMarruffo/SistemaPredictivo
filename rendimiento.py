
import matplotlib.pyplot as plt
import io
import base64

def grafico():
    # Crear una gráfica de ejemplo
    fig, ax = plt.subplots()
    categorias = ['Bajo', 'Medio', 'Alto']
    valores = [30, 50, 20]  # Datos simulados
    ax.bar(categorias, valores, color=['green', 'yellow', 'red'])
    ax.set_title('Distribución de Riesgo de Deserción')

    # Guardar la imagen en un buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Cerrar la figura
    plt.close(fig)

    # Codificar la imagen en base64
    return base64.b64encode(img.getvalue()).decode('utf-8')
