# Evolución del sentimiento hacia la inteligencia artificial en X: un estudio longitudinal en español (2022-2025)

**Trabajo de Fin de Grado — Grado en Análisis de Datos en la Empresa (Business Analytics)**  
Universidad Autónoma de Madrid · Curso 2025/2026

**Autora:** Isabela Bulgaru Merla  
**Tutor:** Ramón Mahía Casado

---

## Descripción

Este repositorio contiene el código y los datos del TFG, que analiza la evolución del sentimiento de la comunidad hispanohablante sobre la inteligencia artificial en X (anteriormente Twitter) entre junio de 2022 y diciembre de 2025.

El análisis combina:
- **Análisis de sentimiento** mediante RoBERTuito, modelo preentrenado en 500 millones de tuits en español.
- **Modelado de tópicos** mediante BERTopic, con embeddings multilingües, reducción dimensional UMAP y clustering HDBSCAN.
- **Análisis estadístico** de series temporales con el test de Pettitt y el test de Mann-Whitney U.

El corpus está compuesto por **58.496 tuits en español** extraídos de X entre junio de 2022 y diciembre de 2025.

---

## Estructura del repositorio

```
TFG/
├── scripts/
│   ├── 01_limpieza_datos.py          # Filtrado, normalización geográfica y feature engineering
│   ├── 02_preprocesamiento_texto.py  # Limpieza de texto para sentimiento y tópicos
│   ├── 03_sentimiento_robertuito.py  # Etiquetado automático con RoBERTuito
│   ├── 04_modelado_topicos.py        # Pipeline BERTopic: embeddings, UMAP, HDBSCAN
│   └── 05_analisis_resultados.py     # Generación de gráficos y tests estadísticos
├── outputs/
│   ├── figures/              # Figuras generadas
│   ├── modelo_bertopic       # Modelo BERTopic entrenado sobre el corpus
│   └── tabla_topicos.xlsx    # Salida de tópicos con categorías asignadas
├── requirements.txt
└── README.md
```

---

## Pipeline

El análisis sigue un pipeline secuencial de cinco fases. Cada script lee un CSV de entrada y genera un CSV de salida:

| Fase | Script | Entrada | Salida |
|------|--------|---------|--------|
| Extracción (opcional) | `00_extraccion.py` | - | Datos brutos |
| 1. Limpieza | `01_limpieza_datos.py` | Datos brutos | Corpus limpio |
| 2. Preprocesamiento | `02_preprocesamiento_texto.py` | Corpus limpio | Corpus preprocesado |
| 3. Sentimiento | `03_sentimiento_robertuito.py` | Corpus preprocesado | Corpus con etiquetas |
| 4. Tópicos | `04_modelado_topicos.py` | Corpus preprocesado | Corpus con tópicos |
| 5. Resultados | `05_analisis_resultados.py` | Corpus completo | Figuras y estadísticos |


**Nota sobre la extracción de datos**

Los datos fueron extraídos mediante [twitterapi.io](https://twitterapi.io/), una API de pago que permite acceso a datos históricos de X/Twitter. Para la replicación de la extracción, es necesario registrarse previamente y obtener las credenciales desde el panel de usuario.

## Instalación

```bash
git clone https://github.com/kradshiny/spanish-ai-sent-analysis.git
cd spanish-ai-sent-analysis

python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Parámetros principales

**BERTopic:**
- Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
- UMAP: 15 vecinos, 5 componentes
- HDBSCAN: tamaño mínimo de clúster 200, 10 muestras mínimas

**RoBERTuito:**
- Modelo: `pysentimiento/robertuito-sentiment-analysis`
- Inferencia en batches de 1.000 tuits

---

## Datos

La extracción se realizó mediante [twitterapi.io](https://twitterapi.io) con las palabras clave `"Inteligencia Artificial"`, `"ChatGPT"` e `"IA"`, filtrando por idioma español y excluyendo retuits y respuestas.

Los datos brutos no se incluyen en el repositorio por restricciones de tamaño. Están disponibles a través de Google Drive:

**[📁 Descargar datos completos](https://drive.google.com/file/d/1CdJLiI9bLCRpJUZMMlTWPqxwyOYaiLNz/view?usp=share_link)**

El archivo incluye:
- `data_clean_with_2022.csv` - Datos crudos extraídos de X
- `data_wo_na.csv` - Datos tras limpieza
- `data_clean_text.csv` - Datos con texto preprocesado
- `data_sentiment.csv` - Datos con sentimiento etiquetado
- `data_final.csv` - Datos con tópicos asignados


