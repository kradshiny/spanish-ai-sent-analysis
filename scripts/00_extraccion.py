# =============================================================================
# 00_EXTRACCION.PY
# Extracción de tuits mediante la API de twitterapi.io
#
# REQUISITOS PREVIOS:
# - Registrarse en https://twitterapi.io y obtener una clave de API
# - Instalar dependencias: pip install requests pandas
#
# SALIDA:
# - data_clean_with_2022.csv: corpus bruto con los tuits extraídos
# =============================================================================

# %%
import requests
import pandas as pd
import calendar
from datetime import date

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Introduce tu clave de API de twitterapi.io
API_KEY = "TU_CLAVE_API"

# Periodo de extracción
ANIO_INICIO = 2022
MES_INICIO = 6
ANIO_FIN = 2025
MES_FIN = 12

# Número máximo de páginas por mes (cada página devuelve ~20 tuits)
MAX_PAGINAS = 70

# Palabras clave de búsqueda
QUERY_BASE = "(ChatGPT OR \"Inteligencia Artificial\" OR IA) lang:es -filter:retweets -filter:nativeretweets -filter:replies"

# Columnas a conservar en el CSV de salida
COLUMNAS = ['id', 'text', 'source', 'retweetCount', 'replyCount',
            'likeCount', 'quoteCount', 'viewCount', 'createdAt', 'lang',
            'author.location']

# =============================================================================
# GENERACIÓN DE RANGOS TEMPORALES MENSUALES
# =============================================================================

# %%
dates = []
for year in range(ANIO_INICIO, ANIO_FIN + 1):
    mes_inicio = MES_INICIO if year == ANIO_INICIO else 1
    mes_fin = MES_FIN if year == ANIO_FIN else 12
    for month in range(mes_inicio, mes_fin + 1):
        first = date(year, month, 1).strftime('%Y-%m-%d')
        last = date(year, month, calendar.monthrange(year, month)[1]).strftime('%Y-%m-%d')
        dates.append((first, last))

print(f"Periodos a extraer: {len(dates)} meses")

# =============================================================================
# EXTRACCIÓN
# =============================================================================

# %%
url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
headers = {"X-API-Key": API_KEY}

all_results = []

for start, end in dates:
    print(f"Extrayendo: {start} → {end}")
    cursor = ""

    for i in range(MAX_PAGINAS):
        query = {
            'queryType': 'Latest',
            'query': f"{QUERY_BASE} since:{start} until:{end}",
            'cursor': cursor
        }

        response = requests.get(url, headers=headers, params=query)

        if response.status_code != 200:
            print(f"  Error {response.status_code} en página {i+1}. Saltando.")
            break

        data = response.json()
        all_results.append(data)

        if data.get('has_next_page'):
            cursor = data.get('next_cursor', "")
        else:
            break

    print(f"  Páginas extraídas: {i+1}")

# =============================================================================
# PROCESAMIENTO Y GUARDADO
# =============================================================================

# %%
df = pd.json_normalize(all_results)
df_exploded = df.explode('tweets').reset_index(drop=True)
tweets_df = pd.json_normalize(df_exploded['tweets'])

# Seleccionar columnas disponibles
columnas_disponibles = [col for col in COLUMNAS if col in tweets_df.columns]
df_filtered = tweets_df[columnas_disponibles].copy()

# Eliminar filas completamente vacías
df_filtered = df_filtered.dropna(how='all')

print(f"\nTuits extraídos: {len(df_filtered)}")
print(df_filtered['createdAt'].apply(lambda x: str(x)[:4]).value_counts().sort_index())

df_filtered.to_csv('data_clean_with_2022.csv', index=False)
print("\nGuardado en data_clean_with_2022.csv")
