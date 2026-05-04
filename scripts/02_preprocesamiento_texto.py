# =============================================================================
# 02_preprocesamiento_texto.py
# Preprocesamiento de texto para sentimiento (RoBERTuito) y tópicos (BERTopic)
# Input:  data_wo_na.csv
# Output: data_clean_text.csv
# =============================================================================

import pandas as pd
import re
import emoji
import spacy
from collections import Counter

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------

data_wo_na = pd.read_csv('data_wo_na.csv', low_memory=False)

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE SPACY Y STOPWORDS
# -----------------------------------------------------------------------------

nlp = spacy.load('es_core_news_lg')
stop_words = nlp.Defaults.stop_words.copy()

# Mantener negaciones (crítico para sentimiento)
negations = {'no', 'nunca', 'jamas', 'tampoco', 'nada', 'nadie', 'ningun', 'ninguna', 'sin'}
stop_words -= negations

# Eliminar términos ubicuos del dominio
domain = {'ia', 'inteligencia', 'artificial', 'ai'}
stop_words |= domain

# Stopwords adicionales tras análisis de frecuencias (primera iteración)
EXTRA_STOPWORDS = {
    # Verbos comodín
    'utilizar', 'crear', 'hablar', 'querer', 'creer', 'seguir',
    'transformar', 'mejorar', 'generar', 'llegar', 'presentar',
    'leer', 'aprender', 'descubrir',
    # Sustantivos genéricos
    'mundo', 'año', 'tiempo', 'vida', 'forma', 'persona',
    'noticia', 'artículo', 'información',
    # Hashtag mal procesado
    'inteligenciaartificial',
    # Residuos de emojis
    'light', 'skin', 'rocket', 'face', 'heart', 'fire', 'star'
}
stop_words |= EXTRA_STOPWORDS

# Stopwords adicionales tras análisis de frecuencias (segunda iteración)
EXTRA_STOPWORDS_V2 = {
    # Verbos comodín
    'permitir', 'buscar', 'contar', 'cambiar', 'impulsar',
    # Sustantivos genéricos
    'tema', 'proceso', 'clave', 'real', 'social',
    # Residuos
    'backhand', 'vía',
    # Participio mal lematizado
    'creado'
}
stop_words |= EXTRA_STOPWORDS_V2

# -----------------------------------------------------------------------------
# 3. FUNCIONES DE PREPROCESAMIENTO
# -----------------------------------------------------------------------------

def clean_text_sentiment(text):
    """Limpieza base para RoBERTuito — conserva emojis originales."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_bertopic(text):
    """Limpieza base para BERTopic — elimina emojis."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = emoji.replace_emoji(text, replace='')  # elimina emojis completamente
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------------------------------------------------------
# 4. APLICAR PREPROCESAMIENTO
# -----------------------------------------------------------------------------

data_wo_na['text_sentiment'] = data_wo_na['text'].apply(clean_text_sentiment)
data_wo_na['text_bertopic']       = data_wo_na['text'].apply(clean_text_bertopic)

# Verificar tokens más frecuentes
all_tokens = [token for doc in data_wo_na['text_lda'] for token in doc]
print(Counter(all_tokens).most_common(50))

# -----------------------------------------------------------------------------
# 5. GUARDADO
# -----------------------------------------------------------------------------

data_wo_na.to_csv('data_clean_text.csv', index=False)
print("Guardado: data_clean_text.csv")
