import re
import sys
import os
import requests
import json
import numpy as np
import pandas as pd
from collections import Counter
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from textblob import TextBlob
from model_data import data

# Función para obtener noticias utilizando la API de NewsAPI
def fetch_news(api_key):
    try:
        url = f'https://newsapi.org/v2/everything?q=news&sortBy=publishedAt&pageSize=100&apiKey={api_key}'
        response = requests.get(url)  # Hacer la solicitud a la API
        data = response.json()  # Parsear la respuesta en formato JSON
        if data['status'].lower() != 'ok':
            raise Exception('Error fetching data from NewsAPI')  # Lanzar excepción si hay un error
    except Exception as e:
        print(f"Error fetching data: {e}")  # Imprimir error en caso de falla
        sys.exit()  # Salir del programa si ocurre un error grave
    
    return data['articles']  # Devolver los artículos obtenidos

# Función para entrenar un modelo de clasificación
def train_classifier():
    df = pd.DataFrame(data)  # Convertir el conjunto de datos en un DataFrame de pandas
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
    # Crear un pipeline que vectoriza los textos y entrena un modelo Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.1))
    model.fit(X_train, y_train)  # Entrenar el modelo
    
    y_pred = model.predict(X_test)  # Hacer predicciones en el conjunto de prueba
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')  # Imprimir la precisión del modelo
    print(classification_report(y_test, y_pred, zero_division=1))  # Imprimir un reporte de clasificación
    
    return model  # Devolver el modelo entrenado

# Función para clasificar artículos usando el modelo entrenado
def categorize_articles(model, articles):
    # Combinar el título y contenido de cada artículo para formar el texto
    texts = [article['title'] + ' ' + (article['content'] or '') for article in articles]
    predictions = model.predict(texts)  # Hacer predicciones usando el modelo
    return predictions  # Devolver las predicciones

# Función para extraer las palabras clave más frecuentes
def extract_keywords(articles, num_keywords=10):
    # Combinar el título y contenido de cada artículo para formar el texto
    texts = [article['title'] + ' ' + (article['content'] or '') for article in articles]
    vectorizer = CountVectorizer(stop_words='english')  # Vectorizar el texto, excluyendo las palabras comunes
    X = vectorizer.fit_transform(texts)  # Transformar los textos en una matriz de cuentas
    
    word_counts = X.sum(axis=0)  # Sumar las ocurrencias de cada palabra
    words = vectorizer.get_feature_names_out()  # Obtener las palabras
    
    word_freq = dict(zip(words, word_counts.A1))  # Crear un diccionario de palabras y sus frecuencias
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)  # Ordenar por frecuencia
    
    return sorted_word_freq[:num_keywords]  # Devolver las palabras clave más frecuentes

# Función para analizar los datos obtenidos
def analyze_data(articles, model):
    # Contar la cantidad de artículos por fuente
    sources = Counter(article['source']['name'] for article in articles if article.get('source') and article['source'].get('name'))
    predictions = categorize_articles(model, articles)  # Obtener las categorías predichas para los artículos
    categories = Counter(predictions)  # Contar la cantidad de artículos por categoría
    keywords = extract_keywords(articles)  # Extraer las palabras clave más frecuentes
    sentiment_counts = analyze_sentiments(articles)  # Analizar los sentimientos de los artículos
    
    return sources, categories, keywords, sentiment_counts  # Devolver los resultados del análisis

# Función para crear un reporte basado en el análisis
def create_report(sources, categories, keywords, sentiment_counts):
    report = {
        'summary': {
            'total_articles': len(articles),  # Total de artículos
            'total_sources': len(sources),  # Total de fuentes
            'total_categories': len(categories),  # Total de categorías
            'total_keywords': len(keywords),  # Total de palabras clave
            'total_sentiments': len(sentiment_counts)  # Total de sentimientos
        },
        'top_keywords': [{'keyword': word, 'frequency': freq} for word, freq in keywords],  # Palabras clave principales
        'popular_categories': [{'category': category, 'count': count} for category, count in categories.most_common(5)],  # Categorías más comunes
        'category_distribution': dict(categories),  # Distribución de categorías
        'sources_distribution': dict(sources),  # Distribución de fuentes
        'sentiment_distribution': dict(sentiment_counts)  # Distribución de sentimientos
    }
    
    return report  # Devolver el reporte

# Función para serializar el reporte y poder guardarlo
def serialize_report(report):
    def convert(value):
        if isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, list):
            return [convert(i) for i in value]
        elif isinstance(value, dict):
            return {convert(k): convert(v) for k, v in value.items()}
        elif isinstance(value, Counter):
            return dict(value)
        else:
            raise TypeError(f'Unsupported type: {type(value)}')
    return convert(report)  # Devolver el reporte serializado

# Función para guardar el reporte en un archivo JSON
def save_report_to_json(report, file_path='reporte.json'):
    with open(file_path, 'w') as file:
        json.dump(serialize_report(report), file, indent=4)  # Guardar el reporte como un archivo JSON

# Función para mostrar los resultados en la consola
def display_results(sources, categories, keywords):
    source_list = [(name, count) for name, count in sources.items()]  # Lista de fuentes y sus conteos
    category_list = [(category, count) for category, count in categories.items()]  # Lista de categorías y sus conteos
    keyword_list = [(word, freq) for word, freq in keywords]  # Lista de palabras clave y sus frecuencias
    
    print("Sources:")
    print(tabulate(source_list, headers=['Source', 'Count'], tablefmt='grid'))  # Mostrar las fuentes en formato de tabla
    
    print("\nCategories:")
    print(tabulate(category_list, headers=['Category', 'Count'], tablefmt='grid'))  # Mostrar las categorías en formato de tabla
    
    print("\nKeywords:")
    print(tabulate(keyword_list, headers=['Keyword', 'Frequency'], tablefmt='grid'))  # Mostrar las palabras clave en formato de tabla

# Función para analizar los sentimientos de los artículos
def analyze_sentiments(articles):
    sentiments = []
    for article in articles:
        title = article.get('title', '')  # Obtener el título del artículo
        blob = TextBlob(title)  # Analizar el sentimiento del título
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            sentiment_label = 'positive'  # Sentimiento positivo
        elif sentiment < 0:
            sentiment_label = 'negative'  # Sentimiento negativo
        else:
            sentiment_label = 'neutral'  # Sentimiento neutral
        
        sentiments.append(sentiment_label)  # Agregar el sentimiento a la lista
    
    sentiment_counts = Counter(sentiments)  # Contar los sentimientos
    return sentiment_counts  # Devolver el conteo de sentimientos

# Función para detectar el sistema operativo
def detectar_so():
    sistema = os.name
    if sistema == 'nt':
        return "cls"  # Comando para limpiar la pantalla en Windows
    elif sistema == 'posix':
        return "clear"  # Comando para limpiar la pantalla en Linux/Unix/MacOS

# Lógica principal del programa
api_key = 'API_KEY'
articles = fetch_news(api_key)  # Obtener las noticias usando la API
model = train_classifier()  # Entrenar el modelo de clasificación
sources, categories, keywords, sentiment_counts = analyze_data(articles, model)  # Analizar los datos obtenidos
borrar = detectar_so()  # Detectar el comando para limpiar la pantalla

while True:
    print("1 - Mostrar Resultados")
    print("2 - Generar Reporte")
    print("3 - Salir")
    teclado = input(">>> ")
    if teclado == '1':
        display_results(sources, categories, keywords)  # Mostrar los resultados en la consola
        print("\nSentimientos en las noticias:")
        print(tabulate(sentiment_counts.items(), headers=['Sentiment', 'Count'], tablefmt='grid'))  # Mostrar los sentimientos
    elif teclado == '2':
        report = create_report(sources, categories, keywords, sentiment_counts)  # Crear el reporte
        save_report_to_json(report)  # Guardar el reporte en un archivo JSON
    elif teclado == '3':
        print("Gracias por usar nuestro software")
        sys.exit()  # Salir del programa
    else:
        print("Debe elegir una opción válida")  # Mensaje de error si la opción no es válida
    input("Presione ENTER para continuar...")
    os.system(borrar)  # Limpiar la pantalla según el sistema operativo
