# Análisis de Noticias

Este proyecto analiza noticias utilizando la API de NewsAPI. El script realiza un análisis de sentimientos en los títulos de las noticias, clasifica las noticias en categorías y extrae palabras clave. También proporciona un informe detallado en formato JSON.

## Requisitos

- Python 3.7 o superior.
- Bibliotecas:
  - `requests`
  - `pandas`
  - `scikit-learn`
  - `textblob`
  - `tabulate`

  Puedes instalar las dependencias usando el siguiente comando:

  pip install -r requirements.txt

## Configuración

1. Obtener una clave de API de NewsAPI:
   - Regístrate en NewsAPI y obtén una clave de API.

2. Actualizar el script con tu clave de API:
   - Reemplaza 'API_KEY' en el script con tu clave de API de NewsAPI.

## Uso

1. Ejecutar el script:
   Asegúrate de estar en el directorio donde se encuentra el script y ejecuta el siguiente comando:
   (en el mismo directorio de desafio.py debe de estar model_data.py, es un archivo para usar de base con scikit-learn y generar un modelo para detectar las noticias)

   python deasafio.py



   El script:

   - Toma noticias desde NewsAPI.
   - Entrena un modelo de clasificación.
   - Analiza la info
   - Tiene un Menú para navegar donde se puede:
      - Mostrar Resultados en la consola. (Realiza un análisis, Extrae palabras clave, etc)
      - Genera un informe en formato JSON y lo guarda como reporte.json.
      - Salir
    

2. Revisar el informe:
   El archivo reporte.json contiene un informe detallado con los siguientes datos:
   - Total de artículos.
   - Total de fuentes.
   - Total de categorías.
   - Total de palabras clave.
   - Distribución de sentimientos.
   - Palabras clave más frecuentes.
   - Categorías más populares.
   - Distribución de categorías y fuentes.


## Ejemplo de Salida por Consola

Cuando el script se ejecute, se mostrará algo como esto en la consola:
### Sources:

| Source   | Count |
|----------|-------|
| Source A | 10    |
| Source B | 8     |

### Categories:

| Category      | Count |
|---------------|-------|
| Sports        | 15    |
| Entertainment | 12    |

### Keywords:

| Keyword   | Frequency |
|-----------|-----------|
| tesla     | 20        |
| electric  | 15        |

### Sentiments:

| Sentiment  | Count |
|------------|-------|
| Positive   | 60    |
| Neutral    | 30    |
| Negative   | 10    |

## Manejo de Errores

- Si el script no puede conectarse a NewsAPI, se lanzará una excepción con un mensaje de error.
- Asegúrate de que tu clave de API sea válida y que tu conexión a Internet esté activa.


## Licencia

Este proyecto está bajo la Licencia MIT.


