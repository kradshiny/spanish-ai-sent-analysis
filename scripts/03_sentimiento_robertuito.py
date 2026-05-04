# =============================================================================
# 03_sentimiento_robertuito.py
# Etiquetado automático de sentimiento con RoBERTuito
# Input:  data_clean_text.csv
# Output: data_sentiment.csv
# =============================================================================

import pandas as pd
from pysentimiento import create_analyzer
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

data_wo_na = pd.read_csv(DATA_PROCESSED / 'data_clean_text.csv', low_memory=False)

# -----------------------------------------------------------------------------
# 2. MODELO ROBERTUITO
# -----------------------------------------------------------------------------

analyzer = create_analyzer(task='sentiment', lang='es')

text_list = data_wo_na['text_sentiment'].to_list()

batch_size = 1000
all_results = []

for i in range(0, len(text_list), batch_size):
    batch = text_list[i:i + batch_size]
    batch_results = analyzer.predict(batch)
    all_results.extend(batch_results)
    print(f"Procesados {i + len(batch)} de {len(text_list)}")

# -----------------------------------------------------------------------------
# 3. ASIGNACIÓN DE RESULTADOS
# -----------------------------------------------------------------------------

data_wo_na.loc[:, "sentiment_robertuito"]  = [r.output       for r in all_results]
data_wo_na.loc[:, "neg_score_robertuito"]  = [r.probas['NEG'] for r in all_results]
data_wo_na.loc[:, "pos_score_robertuito"]  = [r.probas['POS'] for r in all_results]
data_wo_na.loc[:, "neu_score_robertuito"]  = [r.probas['NEU'] for r in all_results]

print(data_wo_na['sentiment_robertuito'].value_counts())

# -----------------------------------------------------------------------------
# 4. GUARDADO
# -----------------------------------------------------------------------------

data_wo_na.to_csv(DATA_PROCESSED / 'data_sentiment.csv', index=False)
print("Guardado: data_sentiment.csv")
