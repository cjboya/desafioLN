from collections import Counter
import re
import requests
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
from tabulate import tabulate
from model_data import data
import json
import numpy as np
from textblob import TextBlob
import os


def fetch_news(api_key):
    try:
        url = f'https://newsapi.org/v2/everything?q=tesla&from=2024-07-08&sortBy=publishedAt&pageSize=100&apiKey={api_key}'
        response = requests.get(url)
        data = response.json()
        if data['status'].lower() != 'ok':
            raise Exception('Error al traer datos NewsAPI')
    except Exception as e:
        print(f"Sin datos, ocurrio un error: {e}")
        sys.exit()
    
    return data['articles']


# Entrenar un modelo de clasificación
def train_classifier():
    # Cargar un conjunto de datos etiquetado
    
    df = pd.DataFrame(data)
    
    # Dividir datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
    
    # Crear un pipeline de vectorización y clasificación
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
    
    return model

# Clasificar artículos usando el modelo entrenado
def categorize_articles(model, articles):
    texts = [article['title'] + ' ' + (article['content'] or '') for article in articles]
    predictions = model.predict(texts)
    
    return predictions

# Extraer palabras clave más frecuentes
def extract_keywords(articles, num_keywords=10):
    texts = [article['title'] + ' ' + (article['content'] or '') for article in articles]
    
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    word_counts = X.sum(axis=0)
    words = vectorizer.get_feature_names_out()
    
    word_freq = dict(zip(words, word_counts.A1))
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_word_freq[:num_keywords]

def analyze_data(articles, model):
    sources = Counter(article['source']['name'] for article in articles if article.get('source') and article['source'].get('name'))
    
    # Obtener categorías de artículos
    predictions = categorize_articles(model, articles)
    categories = Counter(predictions)
    
    # Extraer palabras clave
    keywords = extract_keywords(articles)
    
    # Analizar sentimientos
    sentiment_counts = analyze_sentiments(articles)
    
    return sources, categories, keywords, sentiment_counts


def create_report(sources, categories, keywords, sentiment_counts):
    report = {
        'summary': {
            'total_articles': int(len(articles)),
            'total_sources': int(len(sources)),
            'total_categories': int(len(categories)),
            'total_keywords': int(len(keywords)),
            'total_sentiments': int(len(sentiment_counts))
        },
        'top_keywords': [{'keyword': word, 'frequency': freq} for word, freq in keywords],
        'popular_categories': [{'category': category, 'count': count} for category, count in categories.most_common(5)],
        'category_distribution': dict(categories),
        'sources_distribution': dict(sources),
        'sentiment_distribution': dict(sentiment_counts)
    }
    
    return report

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

    return convert(report)

def save_report_to_json(report, file_path='reporte.json'):
    with open(file_path, 'w') as file:
        json.dump(serialize_report(report), file, indent=4)

def display_results(sources, categories, keywords):
    source_list = [(name, count) for name, count in sources.items()]
    category_list = [(category, count) for category, count in categories.items()]
    keyword_list = [(word, freq) for word, freq in keywords]

    print("Sources:")
    print(tabulate(source_list, headers=['Source', 'Count'], tablefmt='grid'))
    
    print("\nCategories:")
    print(tabulate(category_list, headers=['Category', 'Count'], tablefmt='grid'))
    
    print("\nKeywords:")
    print(tabulate(keyword_list, headers=['Keyword', 'Frequency'], tablefmt='grid'))


def analyze_sentiments(articles):
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        blob = TextBlob(title)
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            sentiment_label = 'positive'
        elif sentiment < 0:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        sentiments.append(sentiment_label)
    
    sentiment_counts = Counter(sentiments)
    
    return sentiment_counts


def detectar_so():
    sistema = os.name
    if sistema == 'nt':
        return "cls" #"Windows"
    elif sistema == 'posix':
        return "clear" #"Linux/Unix/MacOS"



# Credenciales , etc.
api_key = 'API_KEY'
articles = fetch_news(api_key)
model = train_classifier()
sources, categories, keywords, sentiment_counts = analyze_data(articles, model)  # Ajustado para 4 valores
borrar = detectar_so()

while True:
    print("1 - Mostrar Resultados")
    print("2 - Generar Reporte")
    print("3 - Salir")
    teclado = input(">>> ")
    if teclado == '1':
        display_results(sources, categories, keywords)
        print("\nSentimientos en las noticias:")
        print(tabulate(sentiment_counts.items(), headers=['Sentiment', 'Count'], tablefmt='grid'))
    elif teclado == '2':
        report = create_report(sources, categories, keywords, sentiment_counts)
        save_report_to_json(report)
    elif teclado == '3':
        print("Gracias por usar nuestro soft")
        sys.exit()
    else:
        print("Debe elegir una posibilidad")
    input("Toque ENTER para continuar...")
    os.system(borrar)
